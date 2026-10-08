"""Fill {{m:<key>}} placeholders from the metric dictionary and build the final read."""

import re

from luciaread.models import DISPLAY, Metric, Rendered, Report, Triage

PLACEHOLDER = re.compile(r"\{\{m:([^}]+)\}\}")
DISCLAIMER = (
    "Research demo, not medical advice. Model outputs come from classifiers trained on one public "
    "pediatric dataset (Kermany 2018), are not externally validated and may reflect dataset "
    "shortcuts. Not a diagnosis; a qualified clinician must review every image."
)


def format_value(value: float, unit: str) -> str:
    if unit == "pct":
        return f"{value:.1f}%"
    if unit == "px":
        return f"{value:.0f} px"
    return f"{value:.1f}"


def render_text(text: str, metrics: dict[str, Metric]) -> str:
    """Substitute metric placeholders; raises KeyError for an unknown key."""
    return PLACEHOLDER.sub(lambda m: format_value(metrics[m[1]].value, metrics[m[1]].unit), text)


def render(r: Report, modality: str, label: str, triage: Triage, metrics: dict[str, Metric]):
    """Final read: model label and probability from code, text with placeholders filled."""
    return Rendered(
        modality=modality,
        label=label,
        label_display=DISPLAY[label],
        probability_pct=metrics[f"prob.{label}.pct"].value,
        triage=triage,
        headline=render_text(r.headline, metrics),
        summary=render_text(r.summary, metrics),
        key_points=[render_text(p.text, metrics) for p in r.key_points],
        review_note=render_text(r.review_note, metrics),
        disclaimer=DISCLAIMER,
    )
