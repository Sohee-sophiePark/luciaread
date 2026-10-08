"""Input gate (G0), metadata stripping, preprocessing and image-quality metrics."""

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from luciaread.config import RulesCfg
from luciaread.models import Flag, GateResult, Metric, ToolResult

SIZE = 224
DISPLAY_MAX = 1024
FORMATS = {"JPEG", "PNG"}


@dataclass(frozen=True)
class Scan:
    png: bytes  # metadata-free RGB PNG, longest side <= DISPLAY_MAX
    gray: np.ndarray  # uint8 SIZE x SIZE, model input
    width: int
    height: int
    saturation: float  # mean HSV saturation in [0, 1]
    sha256: str


def load_scan(data: bytes, rules: RulesCfg) -> tuple[Scan | None, GateResult]:
    """G0: size, format, pixel count and decode checks; returns the cleaned scan or violations."""
    if len(data) > rules.max_upload_mb * 1024 * 1024:
        return None, GateResult(gate="G0", passed=False, violations=["file too large"])
    try:
        im = Image.open(io.BytesIO(data))
        if im.format not in FORMATS:
            return None, GateResult(gate="G0", passed=False, violations=["not a JPEG or PNG"])
        if im.width * im.height > rules.max_pixels:
            return None, GateResult(gate="G0", passed=False, violations=["too many pixels"])
        rgb = im.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None, GateResult(gate="G0", passed=False, violations=["image could not be decoded"])
    if min(rgb.size) < rules.min_side_px:
        return None, GateResult(gate="G0", passed=False, violations=["image too small"])
    w, h = rgb.size
    rgb.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    png = buf.getvalue()
    sat = float(np.asarray(rgb.convert("HSV"))[..., 1].mean() / 255)
    gray = np.asarray(rgb.convert("L").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)
    scan = Scan(png, gray, w, h, sat, hashlib.sha256(png).hexdigest())
    return scan, GateResult(gate="G0", passed=True)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian."""
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def quality(scan: Scan, modality: str, rules: RulesCfg) -> ToolResult:
    """Contrast, brightness, sharpness and resolution; `low_quality` flag when out of range."""
    tool = "image_quality"
    g = scan.gray.astype(np.float64)
    vals = {
        "quality.contrast.score": (g.std(), "score", "Contrast (pixel std)"),
        "quality.brightness.score": (g.mean(), "score", "Mean brightness (0-255)"),
        "quality.sharpness.score": (sharpness(scan.gray), "score", "Sharpness (Laplacian var.)"),
        "quality.min_side.px": (min(scan.width, scan.height), "px", "Shortest image side"),
    }
    metrics = [
        Metric(key=k, value=round(float(v), 1), unit=u, label=lbl, source_tool=tool)
        for k, (v, u, lbl) in vals.items()
    ]
    qc = rules.quality[modality]
    lo, hi = qc.brightness_range
    problems = {
        "quality.contrast.score": vals["quality.contrast.score"][0] < qc.min_contrast_std,
        "quality.brightness.score": not lo <= vals["quality.brightness.score"][0] <= hi,
        "quality.sharpness.score": vals["quality.sharpness.score"][0] < qc.min_sharpness,
    }
    bad = [k for k, b in problems.items() if b]
    flags = (
        [
            Flag(
                flag_id="low_quality",
                severity="review",
                metric_refs=bad,
                message="Image quality is outside the range seen in training data",
            )
        ]
        if bad
        else []
    )
    return ToolResult(tool=tool, metrics=metrics, flags=flags)
