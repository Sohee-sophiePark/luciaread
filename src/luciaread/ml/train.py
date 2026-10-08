"""Train, calibrate and evaluate the LuciaRead classifiers. Needs the `ml` dependency group."""

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from luciaread.ml import metrics as M
from luciaread.ml.split import load

ROOT = Path(__file__).resolve().parents[3]
TASKS = {
    "cxr": ["NORMAL", "PNEUMONIA"],
    "oct": ["CNV", "DME", "DRUSEN", "NORMAL"],
    "modality": ["cxr", "oct"],
}
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CXR_TARGET_SENSITIVITY = 0.95
MODALITY_PER_CLASS = {"train": 1500, "val": 300, "test": 500}


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def decode(path: Path) -> np.ndarray:
    """Grayscale uint8 SIZE x SIZE array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def images(rows: list[dict], raw: Path, cache: Path) -> np.ndarray:
    """Decoded images for rows, cached on disk by the hash of their paths."""
    key = hashlib.sha256("\n".join(r["path"] for r in rows).encode()).hexdigest()[:16]
    f = cache / f"{key}.npy"
    if f.exists():
        return np.load(f)
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(lambda r: decode(raw / r["path"]), rows)))
    cache.mkdir(parents=True, exist_ok=True)
    np.save(f, arr)
    return arr


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 (N,H,W) -> normalized float (N,3,H,W)."""
    x = x.float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


AUGMENT = v2.Compose(
    [
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
    ]
)


def build_model(task: str, pretrained: bool = True) -> nn.Module:
    return timm.create_model("resnet18", pretrained=pretrained, num_classes=len(TASKS[task]))


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, dev: torch.device, bs: int = 128) -> np.ndarray:
    model.eval()
    out = [
        model(normalize(torch.from_numpy(x[i : i + bs]).to(dev))).cpu()
        for i in range(0, len(x), bs)
    ]
    return torch.cat(out).numpy().astype(np.float64)


def fit(task, xtr, ytr, xva, yva, epochs, dev, seed=7, bs=64) -> nn.Module:
    """AdamW + cosine schedule; keeps the epoch with the lowest validation NLL."""
    torch.manual_seed(seed)
    model = build_model(task).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(xtr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    loss_fn, rng = nn.CrossEntropyLoss(), np.random.default_rng(seed)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        t0, perm = time.time(), rng.permutation(len(xtr))
        for i in range(0, len(xtr), bs):
            b = perm[i : i + bs]
            xb = torch.stack([AUGMENT(torch.from_numpy(x)[None]) for x in xtr[b]]).squeeze(1)
            loss = loss_fn(model(normalize(xb.to(dev))), torch.from_numpy(ytr[b]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        v = M.nll(M.softmax(predict_logits(model, xva, dev)), yva)
        print(
            f"[{task}] epoch {ep + 1}/{epochs} val_nll={v:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if v < best:
            best, best_state = (
                v,
                {k: t.detach().cpu().clone() for k, t in model.state_dict().items()},
            )
    model.load_state_dict(best_state)
    return model


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimizing NLL on the given (validation) logits."""
    z, t = torch.tensor(logits), torch.tensor(y)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def task_rows(task: str, rows: list[dict], split: str, seed: int = 7) -> list[dict]:
    """Rows for a task and split with an integer `y`; modality rows are a seeded sample."""
    classes = TASKS[task]
    if task == "modality":
        rng, out = random.Random(seed), []
        for m in classes:
            pool = [r for r in rows if r["split"] == split and r["modality"] == m]
            out += rng.sample(pool, min(MODALITY_PER_CLASS[split], len(pool)))
        return [{**r, "y": classes.index(r["modality"])} for r in out]
    return [
        {**r, "y": classes.index(r["label"])}
        for r in rows
        if r["split"] == split and r["modality"] == task
    ]


def evaluate(task, lv, yv, lt, yt, groups) -> dict:
    """Calibration fitted on validation; every reported number computed on test."""
    t = fit_temperature(lv, yv)
    raw, cal = M.softmax(lt), M.softmax(lt, t)
    acc = lambda i: float((cal[i].argmax(1) == yt[i]).mean())  # noqa: E731
    aucf = lambda i: M.macro_auc(cal[i], yt[i])  # noqa: E731
    out = {
        "classes": TASKS[task],
        "temperature": t,
        "n_test": int(len(yt)),
        "n_test_patients": int(len(np.unique(groups))),
        "accuracy": acc(np.arange(len(yt))),
        "accuracy_ci95": M.cluster_bootstrap(acc, groups),
        "auc": aucf(np.arange(len(yt))),
        "auc_ci95": M.cluster_bootstrap(aucf, groups),
        "ece_before": M.ece(raw, yt)[0],
        "ece_after": M.ece(cal, yt)[0],
        "nll_before": M.nll(raw, yt),
        "nll_after": M.nll(cal, yt),
        "reliability_bins": M.ece(cal, yt)[1],
        "recall_per_class": {
            c: float((cal[yt == k].argmax(1) == k).mean()) for k, c in enumerate(TASKS[task])
        },
    }
    if task == "cxr":
        pv = M.softmax(lv, t)[:, 1]
        thr = M.threshold_for_sensitivity(pv, yv == 1, CXR_TARGET_SENSITIVITY)
        sens, spec = M.sens_spec(cal[:, 1], yt == 1, thr)
        out |= {
            "threshold": thr,
            "sensitivity": sens,
            "specificity": spec,
            "sensitivity_ci95": M.cluster_bootstrap(
                lambda i: M.sens_spec(cal[i, 1], yt[i] == 1, thr)[0], groups
            ),
            "specificity_ci95": M.cluster_bootstrap(
                lambda i: M.sens_spec(cal[i, 1], yt[i] == 1, thr)[1], groups
            ),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--control", action="store_true", help="shuffled-label leakage control; no weights saved"
    )
    a = ap.parse_args(argv)
    raw, cache = ROOT / "data/raw", ROOT / "data/raw/cache"
    manifest = ROOT / "data/splits/manifest.csv"
    rows, dev = load(manifest), device()
    sets = {s: task_rows(a.task, rows, s) for s in ("train", "val", "test")}
    x = {s: images(r, raw, cache) for s, r in sets.items()}
    y = {s: np.array([r["y"] for r in rs]) for s, rs in sets.items()}
    if a.control:
        y["train"] = np.random.default_rng(7).permutation(y["train"])
    model = fit(a.task, x["train"], y["train"], x["val"], y["val"], a.epochs, dev)
    lv, lt = predict_logits(model, x["val"], dev), predict_logits(model, x["test"], dev)
    res = evaluate(
        a.task, lv, y["val"], lt, y["test"], np.array([r["patient"] for r in sets["test"]])
    )
    res |= {
        "epochs": a.epochs,
        "backbone": "resnet18 (timm, ImageNet)",
        "input": f"{SIZE}px grayscale",
        "n_train": len(y["train"]),
        "n_val": len(y["val"]),
        "torch": torch.__version__,
        "device": str(dev),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()[:12],
    }
    key = f"{a.task}_control" if a.control else a.task
    if not a.control:
        (ROOT / "weights").mkdir(exist_ok=True)
        torch.save(model.state_dict(), ROOT / f"weights/{a.task}.pt")
        res["weights_sha256"] = hashlib.sha256(
            (ROOT / f"weights/{a.task}.pt").read_bytes()
        ).hexdigest()
    card_path = ROOT / "reports/model_card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    card[key] = res
    card_path.parent.mkdir(exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2) + "\n")
    keys = (
        "accuracy",
        "accuracy_ci95",
        "auc",
        "auc_ci95",
        "ece_before",
        "ece_after",
        "sensitivity",
        "specificity",
    )
    print(json.dumps({k: res[k] for k in keys if k in res}, indent=2))


if __name__ == "__main__":
    main()
