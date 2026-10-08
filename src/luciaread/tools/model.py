"""Classifier, modality check and Grad-CAM tools over a Scan; the model backend is injected."""

import hashlib
import io
import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from luciaread.config import RulesCfg
from luciaread.models import CLASSES, DISPLAY, Flag, Metric, ToolResult
from luciaread.tools.image import Scan


class Classifier(Protocol):
    classes: list[str]
    temperature: float
    threshold: float | None

    def logits(self, gray: np.ndarray) -> np.ndarray: ...

    def gradcam(self, gray: np.ndarray, k: int) -> np.ndarray: ...


class TorchClassifier:
    """ResNet-18 weights from `weights/<task>.pt` with calibration from `weights/<task>.json`."""

    def __init__(self, task: str, weights: Path) -> None:
        import torch

        from luciaread.ml.train import build_model, device

        meta = json.loads((weights / f"{task}.json").read_text())
        self.classes, self.temperature = meta["classes"], meta["temperature"]
        self.threshold = meta.get("threshold")
        self.torch, self.dev = torch, device()
        self.model = build_model(task, pretrained=False)
        state = torch.load(weights / f"{task}.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.dev).eval()

    def _input(self, gray: np.ndarray):
        from luciaread.ml.train import normalize

        return normalize(self.torch.from_numpy(gray[None]).to(self.dev))

    def logits(self, gray: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.model(self._input(gray))[0].cpu().double().numpy()

    def gradcam(self, gray: np.ndarray, k: int) -> np.ndarray:
        """Plain Grad-CAM on the last conv stage, upsampled to the input size, scaled to [0, 1]."""
        store = {}
        layer = self.model.layer4
        h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
        h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
        try:
            self.model.zero_grad()
            self.model(self._input(gray))[0, k].backward()
        finally:
            h1.remove()
            h2.remove()
        w = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = self.torch.relu((w * store["a"]).sum(1, keepdim=True))
        cam = self.torch.nn.functional.interpolate(cam, size=gray.shape, mode="bilinear")
        cam = cam[0, 0].detach().cpu().double().numpy()
        return cam / cam.max() if cam.max() > 0 else cam


def calibrated(clf: Classifier, gray: np.ndarray) -> np.ndarray:
    z = clf.logits(gray) / clf.temperature
    e = np.exp(z - z.max())
    return e / e.sum()


def modality(scan: Scan, clf: Classifier) -> ToolResult:
    """Probability the image is a chest X-ray vs a retinal OCT (modality CNN)."""
    p = calibrated(clf, scan.gray)
    metrics = [
        Metric(
            key=f"modality.{c}.pct",
            value=round(100 * float(p[i]), 1),
            unit="pct",
            label=f"Modality model: {DISPLAY[c]}",
            source_tool="check_modality",
        )
        for i, c in enumerate(clf.classes)
    ]
    return ToolResult(
        tool="check_modality", metrics=metrics, data={"modality": clf.classes[int(p.argmax())]}
    )


def classify(scan: Scan, clf: Classifier, modality: str, rules: RulesCfg) -> ToolResult:
    """Calibrated class probabilities and the model label; CXR uses the validation threshold."""
    tool, classes = "classify", CLASSES[modality]
    p = calibrated(clf, scan.gray)
    if clf.threshold is not None:
        k = int(p[1] >= clf.threshold)
    else:
        k = int(p.argmax())
    metrics = [
        Metric(
            key=f"prob.{c}.pct",
            value=round(100 * float(p[i]), 1),
            unit="pct",
            label=f"Calibrated probability: {DISPLAY[c]}",
            source_tool=tool,
        )
        for i, c in enumerate(classes)
    ]
    flags = []
    if p[k] < rules.low_confidence:
        flags.append(
            Flag(
                flag_id="low_confidence",
                severity="review",
                metric_refs=[f"prob.{classes[k]}.pct"],
                message="Model confidence in its label is below the review threshold",
            )
        )
    return ToolResult(
        tool=tool, metrics=metrics, flags=flags, data={"label": classes[k], "label_index": k}
    )


def anatomy_mask(gray: np.ndarray, modality: str) -> np.ndarray:
    """Coarse expected-anatomy region: CXR central lung box; OCT bright retinal band rows."""
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=bool)
    if modality == "cxr":
        mask[int(0.08 * h) : int(0.85 * h), int(0.1 * w) : int(0.9 * w)] = True
        return mask
    rows = np.convolve(gray.astype(float).mean(axis=1), np.ones(9) / 9, mode="same")
    bright = np.flatnonzero(rows > rows.mean())
    if len(bright):
        mask[max(0, bright[0] - 10) : min(h, bright[-1] + 11), :] = True
    return mask


def explain(scan: Scan, clf: Classifier, modality: str, k: int, rules: RulesCfg):
    """Grad-CAM for class k: share of heat inside the expected anatomy; returns (result, cam)."""
    tool = "explain_gradcam"
    cam = clf.gradcam(scan.gray, k)
    total = cam.sum()
    inside = float((cam * anatomy_mask(scan.gray, modality)).sum() / total) if total else 0.0
    key = "attention.inside.pct"
    metrics = [
        Metric(
            key=key,
            value=round(100 * inside, 1),
            unit="pct",
            label="Grad-CAM heat inside expected anatomy",
            source_tool=tool,
        )
    ]
    flags = []
    if inside < rules.attention_inside_min:
        flags.append(
            Flag(
                flag_id="attention_off_target",
                severity="review",
                metric_refs=[key],
                message="Model attention falls largely outside the expected anatomy",
            )
        )
    return ToolResult(tool=tool, metrics=metrics, flags=flags), cam


def heatmap_png(cam: np.ndarray) -> bytes:
    """RGBA heatmap (yellow-red, alpha follows intensity) for overlay on the scan."""
    c = np.clip(cam, 0, 1)
    rgba = np.stack([np.full_like(c, 255), 255 * (1 - c) ** 0.7, np.zeros_like(c), 200 * c], -1)
    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(buf, format="PNG")
    return buf.getvalue()


class RecordingClassifier:
    """Wraps a classifier and stores its outputs by image hash for torch-free replay."""

    def __init__(self, inner: Classifier, store: dict) -> None:
        self.inner, self.store = inner, store
        self.classes, self.temperature = inner.classes, inner.temperature
        self.threshold = inner.threshold
        store.setdefault("meta", {})[",".join(inner.classes)] = {
            "temperature": inner.temperature,
            "threshold": inner.threshold,
        }

    def logits(self, gray: np.ndarray) -> np.ndarray:
        out = self.inner.logits(gray)
        self.store[f"{_key(gray, self.classes)}:logits"] = out.tolist()
        return out

    def gradcam(self, gray: np.ndarray, k: int) -> np.ndarray:
        cam = self.inner.gradcam(gray, k)
        self.store[f"{_key(gray, self.classes)}:cam{k}"] = np.round(cam, 3).tolist()
        return cam


class ReplayClassifier:
    """Serves recorded logits and Grad-CAM maps; raises KeyError on an unrecorded image."""

    def __init__(self, classes: list[str], store: dict) -> None:
        meta = store["meta"][",".join(classes)]
        self.classes, self.store = classes, store
        self.temperature, self.threshold = meta["temperature"], meta["threshold"]

    def logits(self, gray: np.ndarray) -> np.ndarray:
        return np.array(self.store[f"{_key(gray, self.classes)}:logits"])

    def gradcam(self, gray: np.ndarray, k: int) -> np.ndarray:
        return np.array(self.store[f"{_key(gray, self.classes)}:cam{k}"])


def _key(gray: np.ndarray, classes: list[str]) -> str:
    return f"{','.join(classes)}:{hashlib.sha256(gray.tobytes()).hexdigest()[:16]}"
