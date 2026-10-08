"""Model-card audits for a trained task: original-test-folder subset metrics and the Grad-CAM
randomization sanity check. Needs the `ml` dependency group and `weights/<task>.pt`."""

import argparse
import json

import numpy as np
import torch

from luciaread.ml import metrics as M
from luciaread.ml.split import load
from luciaread.ml.train import ROOT, build_model, images, predict_logits, task_rows
from luciaread.tools.model import TorchClassifier


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.ravel().argsort().argsort(), b.ravel().argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["cxr", "oct"], required=True)
    ap.add_argument("--n-cam", type=int, default=40)
    a = ap.parse_args(argv)
    rows = task_rows(a.task, load(ROOT / "data/splits/manifest.csv"), "test")
    x = images(rows, ROOT / "data/raw", ROOT / "data/raw/cache")
    y = np.array([r["y"] for r in rows])
    clf = TorchClassifier(a.task, ROOT / "weights")
    probs = M.softmax(predict_logits(clf.model, x, clf.dev), clf.temperature)
    orig = np.array(["/test/" in r["path"] for r in rows])
    pred = (
        (probs[:, 1] >= clf.threshold).astype(int) if clf.threshold is not None else probs.argmax(1)
    )
    subset = {
        "n": int(orig.sum()),
        "accuracy": float((pred[orig] == y[orig]).mean()),
        "auc": M.macro_auc(probs[orig], y[orig]),
    }
    rng = np.random.default_rng(7)
    idx = rng.choice(len(rows), size=min(a.n_cam, len(rows)), replace=False)
    trained = [clf.gradcam(x[i], int(pred[i])) for i in idx]
    torch.manual_seed(7)
    clf.model.load_state_dict(build_model(a.task, pretrained=False).state_dict())
    clf.model.to(clf.dev).eval()
    random = [clf.gradcam(x[i], int(pred[i])) for i in idx]
    corr = [spearman(t[::4, ::4], r[::4, ::4]) for t, r in zip(trained, random, strict=True)]
    audit = {
        "original_test_folder_subset": subset,
        "gradcam_randomization": {
            "n_images": len(idx),
            "mean_spearman_trained_vs_random": float(np.nanmean(corr)),
            "note": "low correlation = heatmap depends on learned weights (Adebayo et al. 2018)",
        },
    }
    card_path = ROOT / "reports/model_card.json"
    card = json.loads(card_path.read_text())
    card[a.task]["audit"] = audit
    card_path.write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
