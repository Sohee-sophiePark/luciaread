"""Deterministic gates: G1 modality agreement, G4 specialist findings, G5 report checks."""

import re

from pydantic import ValidationError

from luciaread.config import RulesCfg
from luciaread.harness.injection import detect
from luciaread.models import (
    CLASSES,
    Flag,
    GateResult,
    Metric,
    Report,
    RouteDecision,
    SpecialistFindings,
    ToolResult,
)
from luciaread.render import PLACEHOLDER, render_text

PROHIBITED = re.compile(
    r"\bdiagnos\w*|\bdefinite(ly)?\b|\bcertain(ly)?\b|\bconfirm(s|ed)?\b|\brule[sd]? out\b"
    r"|\bno need\b|\btreat\w*|\bprescri\w*|\bantibiotic\w*|\bsurgery\b|\bguarantee\w*",
    re.IGNORECASE,
)
SECOND_PERSON = re.compile(r"\byou(r|rs|rself)?\b", re.IGNORECASE)
UNIT_AFTER = re.compile(r"\}\}\s*(%|(percent|px|pixels?)\b)", re.IGNORECASE)
LIMITS = {"headline": 16, "summary": 90, "review_note": 50, "key_point": 30, "key_points": 4}


def has_raw_digits(text: str) -> bool:
    """True if any digit remains after removing {{m:...}} placeholders."""
    return bool(re.search(r"\d", PLACEHOLDER.sub("", text)))


def modality_gate(saturation: float, cnn: str | None, route: RouteDecision | None, rules: RulesCfg):
    """G1: colour photos are rejected before any LLM call; otherwise router and CNN must agree."""
    if saturation > rules.colour_max_saturation:
        return GateResult(
            gate="G1", passed=False, violations=["colour image, not a grayscale scan"]
        )
    if route is None:
        return GateResult(gate="G1", passed=True)
    if route.modality == "other":
        return GateResult(gate="G1", passed=False, violations=["router: not a supported scan"])
    if route.modality != cnn:
        return GateResult(
            gate="G1", passed=False, violations=["router and modality model disagree"]
        )
    return GateResult(gate="G1", passed=True)


def _words(text: str) -> int:
    return len(text.split())


def output_gates(
    raw: dict, modality: str, label: str, metrics: dict[str, Metric], flags: dict[str, Flag]
) -> tuple[Report | None, GateResult]:
    """G5: schema, numbers via placeholders, refs, label, review flags, language, length, echo."""
    try:
        r = Report.model_validate(raw)
    except ValidationError as e:
        return None, GateResult(gate="G5", passed=False, violations=[f"schema: {e.error_count()}"])
    texts = [r.headline, r.summary, r.review_note, *(p.text for p in r.key_points)]
    v = []
    if any(has_raw_digits(t) for t in texts):
        v.append("G5.2 raw number outside a {{m:...}} placeholder")
    unknown = {k for t in texts for k in PLACEHOLDER.findall(t)} - metrics.keys()
    unknown |= {k for p in r.key_points for k in p.metric_refs} - metrics.keys()
    if unknown:
        v.append(f"G5.3 unknown metric keys: {sorted(unknown)}")
    if bad := {f for p in r.key_points for f in p.flag_refs} - flags.keys():
        v.append(f"G5.4 unknown flag refs: {sorted(bad)}")
    others = [c for c in CLASSES[modality] if c != label]
    if not re.search(rf"\b{label}\b", r.headline, re.IGNORECASE):
        v.append(f"G5.5 headline must name the model label {label}")
    if hit := [c for c in others if re.search(rf"\b{c}\b", r.headline, re.IGNORECASE)]:
        v.append(f"G5.5 headline names a label the model did not output: {hit}")
    review = sorted(k for k, f in flags.items() if f.severity == "review")
    cited = {f for p in r.key_points for f in p.flag_refs}
    if missing := [f for f in review if f not in cited]:
        v.append(f"G5.6 review flags not cited in key_points: {missing}")
    if review and re.search(r"\broutine\b", " ".join(texts), re.IGNORECASE):
        v.append("G5.6 says routine while review flags are open")
    if hits := sorted({m.group(0).lower() for t in texts for m in PROHIBITED.finditer(t)}):
        v.append(f"G5.7 prohibited wording: {hits}")
    if any(SECOND_PERSON.search(t) for t in texts):
        v.append("G5.7 second person; write for the reviewing clinician")
    for field in ("headline", "summary", "review_note"):
        if _words(getattr(r, field)) > LIMITS[field]:
            v.append(f"G5.8 {field} over {LIMITS[field]} words")
    if len(r.key_points) > LIMITS["key_points"] or any(
        _words(p.text) > LIMITS["key_point"] for p in r.key_points
    ):
        v.append("G5.8 key_points over limit")
    if any(UNIT_AFTER.search(t) for t in texts):
        v.append("G5.10 unit word after a placeholder; placeholders render with their unit")
    if not unknown and any(detect(render_text(t, metrics)) for t in texts):
        v.append("G5.9 echoes an embedded instruction")
    return r, GateResult(gate="G5", passed=not v, violations=v)


MAX_SPECIALIST_FINDINGS = 3


def findings_gate(
    raw: dict, results: list[ToolResult]
) -> tuple[SpecialistFindings | None, GateResult]:
    """G4: schema, count, numbers via placeholders, refs limited to the specialist's own tools,
    prohibited wording. Uncited flags are not a violation; the caller adds code findings."""
    try:
        out = SpecialistFindings.model_validate(raw)
    except ValidationError as e:
        return None, GateResult(gate="G4", passed=False, violations=[f"schema: {e.error_count()}"])
    keys = {m.key for r in results for m in r.metrics}
    flag_ids = {f.flag_id for r in results for f in r.flags}
    v = []
    if not out.findings or len(out.findings) > MAX_SPECIALIST_FINDINGS:
        v.append(f"G4.1 write 1 to {MAX_SPECIALIST_FINDINGS} findings")
    for i, p in enumerate(out.findings, start=1):
        if has_raw_digits(p.text):
            v.append(f"G4.2 finding {i}: raw number outside a {{{{m:...}}}} placeholder")
        if bad := (set(PLACEHOLDER.findall(p.text)) | set(p.metric_refs)) - keys:
            v.append(f"G4.3 finding {i}: unknown metric keys {sorted(bad)}")
        if bad := set(p.flag_refs) - flag_ids:
            v.append(f"G4.3 finding {i}: unknown flag refs {sorted(bad)}")
        if hits := sorted({m.group(0).lower() for m in PROHIBITED.finditer(p.text)}):
            v.append(f"G4.4 finding {i}: prohibited wording {hits}")
    return out, GateResult(gate="G4", passed=not v, violations=v)
