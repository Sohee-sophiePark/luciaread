"""Shared data models: tool outputs, specialist reports, drafts, verdicts and run state."""

import datetime as dt
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

Modality = Literal["cxr", "oct"]
Unit = Literal["pct", "px", "score"]
Triage = Literal["routine", "needs_review"]

CLASSES: dict[str, list[str]] = {
    "cxr": ["NORMAL", "PNEUMONIA"],
    "oct": ["CNV", "DME", "DRUSEN", "NORMAL"],
}
DISPLAY = {
    "cxr": "chest X-ray",
    "oct": "retinal OCT",
    "NORMAL": "no pattern detected",
    "PNEUMONIA": "pneumonia pattern",
    "CNV": "choroidal neovascularization (CNV) pattern",
    "DME": "diabetic macular edema (DME) pattern",
    "DRUSEN": "drusen pattern",
}
REVIEW_FLAGS = (
    "low_confidence",
    "low_quality",
    "attention_off_target",
    "injection_flagged",
    "visual_concern",
)


class Metric(BaseModel):
    key: str
    value: float
    unit: Unit
    label: str
    source_tool: str


class Flag(BaseModel):
    flag_id: str
    severity: Literal["info", "review"]
    metric_refs: list[str] = []
    message: str


class ToolResult(BaseModel):
    tool: str
    metrics: list[Metric] = []
    flags: list[Flag] = []
    data: dict[str, Any] = {}


class Point(BaseModel):
    text: str = Field(description="One sentence; numbers only as {{m:<metric_key>}} placeholders")
    metric_refs: list[str] = []
    flag_refs: list[str] = []


class Finding(Point):
    id: str
    agent: str


class SpecialistFindings(BaseModel):
    findings: list[Point] = Field(description="At most 3 findings about your own tool results")


class RouteDecision(BaseModel):
    modality: Literal["cxr", "oct", "other"]
    reason: str = Field(description="One short sentence on what the image shows; no numbers")


class VisualDescription(BaseModel):
    view: str = Field(description="Projection or scan orientation, e.g. 'frontal' or 'B-scan'")
    observations: list[str] = Field(description="Up to 3 neutral visual observations, no diagnosis")
    artifacts: list[str] = Field(description="Positioning, exposure, cropping or device artifacts")
    text_in_image: str = Field(description="Any text visible in the image, verbatim; '' if none")
    concern: bool = Field(description="True if image quality or artifacts may limit a reading")


class Report(BaseModel):
    headline: str = Field(description="At most 16 words; names the model output label")
    summary: str = Field(description="At most 90 words, clinician-facing, hedged")
    key_points: list[Point] = Field(description="At most 4, each citing metric or flag refs")
    review_note: str = Field(description="Why routine or why human review is needed; flag refs")


class EvalChecks(BaseModel):
    faithful_to_findings: bool
    hedged_appropriately: bool
    addresses_review_flags: bool
    no_diagnostic_claims: bool
    clinician_voice: bool


class EvalScores(BaseModel):
    clarity: int
    usefulness_for_reviewer: int
    tone: int


class EvalResponse(BaseModel):
    verdict: Literal["pass", "revise"]
    checks: EvalChecks
    scores: EvalScores
    issues: list[str]


class EvalVerdict(EvalResponse):
    passed: bool = False  # decided by code from checks and scores


class GateResult(BaseModel):
    gate: str
    passed: bool
    violations: list[str] = []


class Rendered(BaseModel):
    modality: Modality
    label: str
    label_display: str
    probability_pct: float
    triage: Triage
    headline: str
    summary: str
    key_points: list[str]
    review_note: str
    disclaimer: str


class SignOff(BaseModel):
    decision: Literal["sign_off", "return"]
    note: str = Field(default="", max_length=1000)
    at: dt.datetime | None = None


class RunStatus(StrEnum):
    CREATED = "CREATED"
    INPUT_GATED = "INPUT_GATED"
    REJECTED = "REJECTED"
    ROUTED = "ROUTED"
    ANALYZING = "ANALYZING"
    WRITING = "WRITING"
    AWAITING_SIGNOFF = "AWAITING_SIGNOFF"
    NEEDS_CLINICIAN_REVIEW = "NEEDS_CLINICIAN_REVIEW"
    SIGNED_OFF = "SIGNED_OFF"
    RETURNED = "RETURNED"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"


class RunState(BaseModel):
    run_id: str
    case_id: str
    image_sha256: str
    status: RunStatus = RunStatus.CREATED
    modality: Modality | None = None
    route: RouteDecision | None = None
    label: str | None = None
    triage: Triage | None = None
    metrics: dict[str, Metric] = {}
    flags: dict[str, Flag] = {}
    findings: list[Finding] = []
    visual: VisualDescription | None = None
    drafts: list[Report] = []
    gate_results: list[GateResult] = []
    eval_verdicts: list[EvalVerdict] = []
    revision_count: int = 0
    final: Rendered | None = None
    signoff: SignOff | None = None
    message: str | None = None
    error: str | None = None
    budget: dict[str, int] = {}
    meta: dict[str, Any] = {}
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))
