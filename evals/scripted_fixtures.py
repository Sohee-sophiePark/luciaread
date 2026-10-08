"""Scripted LLM responses, fake classifiers and synthetic images for tests and golden evals."""

import io
import json

import numpy as np
from PIL import Image

from luciaread.harness.orchestrator import Classifiers
from luciaread.llm.base import LLMResponse
from luciaread.llm.scripted import ScriptedClient
from luciaread.models import CLASSES


def resp(parsed: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(parsed), parsed=parsed, tokens_in=120, tokens_out=60)


def route(modality: str) -> LLMResponse:
    return resp({"modality": modality, "reason": "Grayscale medical image."})


def visual(concern: bool = False, text: str = "") -> LLMResponse:
    return resp(
        {
            "view": "frontal",
            "observations": ["Both lung fields are included in the image."],
            "artifacts": [],
            "text_in_image": text,
            "concern": concern,
        }
    )


def findings(*points: tuple[str, list[str]]) -> LLMResponse:
    """Specialist findings: (text, metric_refs) pairs."""
    return resp({"findings": [{"text": t, "metric_refs": refs} for t, refs in points]})


def model_findings(label: str) -> LLMResponse:
    p, a = f"prob.{label}.pct", "attention.inside.pct"
    return findings(
        (f"The model output is {label} at {{{{m:{p}}}}}.", [p]),
        (f"Grad-CAM places {{{{m:{a}}}}} of its heat in the expected anatomy.", [a]),
    )


QUALITY_FINDINGS = findings(
    ("Contrast is {{m:quality.contrast.score}}.", ["quality.contrast.score"])
)


def report(label: str, review_flags: tuple[str, ...] = (), raw_number: bool = False) -> LLMResponse:
    p = f"prob.{label}.pct"
    points = [{"text": f"Model output {label} at {{{{m:{p}}}}}.", "metric_refs": [p]}]
    points += [{"text": "This needs a closer look.", "flag_refs": [f]} for f in review_flags]
    note = "Human review is needed for the open flags." if review_flags else "No review flags."
    return resp(
        {
            "headline": f"Model output {label}" + (" at 97 percent" if raw_number else ""),
            "summary": "The model output suggests this label; the reviewer should check the image.",
            "key_points": points,
            "review_note": note,
        }
    )


def evaluation(passed: bool = True) -> LLMResponse:
    return resp(
        {
            "verdict": "pass" if passed else "revise",
            "checks": {
                "faithful_to_findings": True,
                "hedged_appropriately": passed,
                "addresses_review_flags": True,
                "no_diagnostic_claims": True,
                "clinician_voice": True,
            },
            "scores": {"clarity": 4, "usefulness_for_reviewer": 4, "tone": 5},
            "issues": [] if passed else ["Hedge the summary."],
        }
    )


class FakeClassifier:
    """Fixed logits; Grad-CAM is a centred blob, or a corner blob when `off_target`."""

    def __init__(self, classes, logits, threshold=None, off_target=False) -> None:
        self.classes, self.temperature, self.threshold = classes, 1.0, threshold
        self._logits, self.off_target = np.array(logits, dtype=float), off_target

    def logits(self, gray: np.ndarray) -> np.ndarray:
        return self._logits

    def gradcam(self, gray: np.ndarray, k: int) -> np.ndarray:
        yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]] / gray.shape[0]
        cy, cx = (0.97, 0.97) if self.off_target else (0.5, 0.5)
        return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 0.01)


def classifiers(modality: str = "cxr", logits=None, off_target=False) -> Classifiers:
    """Modality model picks `modality`; the task model returns `logits` (confident by default)."""
    mod = FakeClassifier(["cxr", "oct"], [4.0, -4.0] if modality == "cxr" else [-4.0, 4.0])
    cxr = FakeClassifier(CLASSES["cxr"], logits or [-4.0, 4.0], 0.3, off_target)
    oct_ = FakeClassifier(CLASSES["oct"], logits or [5.0, 0.0, 0.0, 0.0], None, off_target)
    return Classifiers(mod, cxr, oct_)


def scan_png(colour: bool = False, flat: bool = False, seed: int = 0, size: int = 512) -> bytes:
    """Synthetic grayscale scan-like PNG; `colour` adds saturation, `flat` removes contrast."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    g = 128 + 60 * np.sin(12 * xx) * np.cos(9 * yy) + rng.normal(0, 18, (size, size))
    g = np.full((size, size), 128.0) if flat else g
    rgb = np.stack([g, g, g], -1)
    if colour:
        rgb[..., 0] += 90
        rgb[..., 2] -= 90
    buf = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def script_read(
    llm: ScriptedClient,
    modality: str = "cxr",
    label: str = "PNEUMONIA",
    reports=None,
    evals=None,
    vis: LLMResponse | None = None,
) -> None:
    """Queue router, describer, writer and evaluator responses for one read."""
    llm.add("router", route(modality))
    llm.add("analyst:model", model_findings(label))
    llm.add("analyst:quality", QUALITY_FINDINGS)
    llm.add("analyst:visual", vis or visual())
    llm.add("writer", *(reports or [report(label)]))
    llm.add("evaluator", *(evals or [evaluation()]))
