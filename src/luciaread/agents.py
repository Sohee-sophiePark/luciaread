"""LLM agents: router, model/quality/visual specialists, writer, evaluator; code fallbacks."""

import json
from statistics import mean

from pydantic import BaseModel, ValidationError

from luciaread.config import LoopCfg, Role
from luciaread.harness.deps import Deps
from luciaread.harness.gates import findings_gate
from luciaread.harness.injection import detect, redact, wrap
from luciaread.llm.base import LLMRequest, Message
from luciaread.models import (
    DISPLAY,
    EvalResponse,
    EvalVerdict,
    Finding,
    Flag,
    GateResult,
    Metric,
    Report,
    RouteDecision,
    SpecialistFindings,
    ToolResult,
    VisualDescription,
)
from luciaread.render import render_text


def load_prompt(name: str, deps: Deps) -> str:
    return (deps.settings.path("prompts") / name).read_text(encoding="utf-8")


def request(deps: Deps, role: Role, prompt: str, text: str, schema: type[BaseModel], **kw):
    s = deps.settings
    return LLMRequest(
        model=s.models[role],
        system=load_prompt(prompt, deps),
        messages=[Message(role="user", text=text)],
        temperature=s.temperature[role],
        thinking_level=s.thinking_level[role],
        max_output_tokens=s.max_output_tokens[role],
        response_schema=schema,
        purpose=kw.pop("purpose", role),
        **kw,
    )


async def route(png: bytes, deps: Deps) -> RouteDecision | None:
    """Image type only (chest X-ray / retinal OCT / other); None if the reply is unparseable."""
    req = request(deps, "router", "router.md", "Classify the image type.", RouteDecision, image=png)
    try:
        return RouteDecision.model_validate((await deps.llm_call(req)).parsed or {})
    except ValidationError:
        return None


async def describe(png: bytes, modality: str, deps: Deps) -> VisualDescription | None:
    """Neutral visual description; the describer never sees the classifier output."""
    text = f"The image is a {DISPLAY[modality]}. Describe it."
    req = request(
        deps,
        "analyst",
        "visual_describer.md",
        text,
        VisualDescription,
        image=png,
        purpose="analyst:visual",
    )
    try:
        return VisualDescription.model_validate((await deps.llm_call(req)).parsed or {})
    except ValidationError:
        return None


PROMPTS = {"model": "model_analyst.md", "quality": "quality_analyst.md"}


def tool_block(results: list[ToolResult]) -> str:
    """Tool results as plain lines: metrics with values, flags, and the model label if any."""
    lines = []
    for r in results:
        lines.append(f'<tool name="{r.tool}">')
        lines += [f"- metric {m.key}: {m.label} = {m.value} ({m.unit})" for m in r.metrics]
        lines += [f"- flag {f.flag_id} [{f.severity}]: {f.message}" for f in r.flags]
        lines += [f"- model label: {r.data['label']}"] if "label" in r.data else []
        lines.append("</tool>")
    return "\n".join(lines)


async def specialist(
    agent: str, results: list[ToolResult], modality: str, deps: Deps, fallback: list[Finding]
) -> tuple[list[Finding], bool, list[GateResult]]:
    """LLM specialist over its own tool results. G4 with one repair round; on a second failure
    the code `fallback` findings are used (degraded). Uncited flags get a code finding."""
    text = (
        f"Image type: {DISPLAY[modality]}.\n<tool_results>\n{tool_block(results)}\n</tool_results>"
    )
    gates = []
    for _ in range(2):
        req = request(
            deps, "analyst", PROMPTS[agent], text, SpecialistFindings, purpose=f"analyst:{agent}"
        )
        raw = (await deps.llm_call(req)).parsed or {}
        out, g = findings_gate(raw, results)
        gates.append(g)
        deps.trace.emit("gate_result", agent, g.model_dump(), "info" if g.passed else "warn")
        if g.passed:
            tag = agent.upper()
            found = [
                Finding(id=f"{tag}-{i}", agent=agent, **p.model_dump())
                for i, p in enumerate(out.findings, start=1)
            ]
            cited = {f for x in found for f in x.flag_refs}
            for fl in (f for r in results for f in r.flags if f.flag_id not in cited):
                found.append(
                    Finding(
                        id=f"{tag}-{len(found) + 1}",
                        agent=agent,
                        text=fl.message,
                        metric_refs=fl.metric_refs,
                        flag_refs=[fl.flag_id],
                    )
                )
            return found, False, gates
        text += f"\n<previous_findings>\n{json.dumps(raw)}\n</previous_findings>"
        text += f"\n<gate_feedback>\n{json.dumps(g.violations)}\n</gate_feedback>"
    return fallback, True, gates


def model_findings(cls: ToolResult, att: ToolResult, label: str) -> list[Finding]:
    """Fallback for the model specialist: findings templated from tool metrics."""
    p, a = f"prob.{label}.pct", "attention.inside.pct"
    out = [
        Finding(
            id="MODEL-1",
            agent="model",
            metric_refs=[p],
            text=f"Model output {label} ({DISPLAY[label]}) at {{{{m:{p}}}}} calibrated "
            "probability.",
        ),
        Finding(
            id="MODEL-2",
            agent="model",
            metric_refs=[a],
            text=f"Grad-CAM places {{{{m:{a}}}}} of its heat inside the expected anatomy.",
        ),
    ]
    for f in cls.flags + att.flags:
        out.append(
            Finding(
                id=f"MODEL-{len(out) + 1}",
                agent="model",
                text=f.message,
                metric_refs=f.metric_refs,
                flag_refs=[f.flag_id],
            )
        )
    return out


def quality_findings(q: ToolResult) -> list[Finding]:
    """Fallback for the quality specialist: findings templated from tool metrics."""
    keys = [m.key for m in q.metrics if m.unit == "score"]
    refs = ", ".join(f"{{{{m:{k}}}}}" for k in keys)
    if q.flags:
        f = q.flags[0]
        return [
            Finding(
                id="QUAL-1",
                agent="quality",
                text=f"{f.message}: {refs}.",
                metric_refs=keys,
                flag_refs=[f.flag_id],
            )
        ]
    text = f"Contrast, brightness and sharpness ({refs}) are within the training range."
    return [Finding(id="QUAL-1", agent="quality", text=text, metric_refs=keys)]


def visual_findings(v: VisualDescription, redact_on: bool) -> tuple[list[Finding], list[Flag]]:
    """Describer output as findings; image text is untrusted and checked for instructions."""
    flags, findings = [], []
    for i, obs in enumerate([*v.observations, *v.artifacts][:4], start=1):
        if not any(c.isdigit() for c in obs):
            findings.append(Finding(id=f"VIS-{i}", agent="visual", text=obs))
    if v.concern:
        flags.append(
            Flag(
                flag_id="visual_concern",
                severity="review",
                message="Visual describer reports quality or artifact concerns",
            )
        )
    if v.text_in_image.strip() and detect(v.text_in_image):
        flags.append(
            Flag(
                flag_id="injection_flagged",
                severity="review",
                message="Text in the image looks like an instruction; it was ignored",
            )
        )
        if redact_on:
            v.text_in_image = redact(v.text_in_image)
    for f in flags:
        findings.append(
            Finding(
                id=f"VIS-{len(findings) + 1}", agent="visual", text=f.message, flag_refs=[f.flag_id]
            )
        )
    return findings, flags


def writer_input(
    modality, label, triage, metrics: dict[str, Metric], flags, findings, visual
) -> str:
    md = "\n".join(f"- {k}: {m.label} = {m.value} ({m.unit})" for k, m in sorted(metrics.items()))
    fl = "\n".join(f"- {f.flag_id} [{f.severity}]: {f.message}" for f in flags.values()) or "none"
    fd = "\n".join(f"- {f.id} ({f.agent}): {f.text}" for f in findings)
    parts = [
        f'<case modality="{modality}" model_label="{label}" triage="{triage}" />',
        f"<metric_dictionary>\n{md}\n</metric_dictionary>",
        f"<flags>\n{fl}\n</flags>",
        f"<findings>\n{fd}\n</findings>",
    ]
    if visual and visual.text_in_image.strip():
        parts.append(wrap(visual.text_in_image, "image_text", "visual"))
    return "\n".join(parts)


async def write(text: str, deps: Deps, feedback: dict | None) -> dict:
    """Single writer: the report draft with numbers as placeholders."""
    if feedback:
        text += f"\n<previous_draft>\n{json.dumps(feedback['draft'])}\n</previous_draft>"
        text += f'\n<revision_feedback source="{feedback["source"]}">\n'
        text += f"{json.dumps(feedback['items'])}\n</revision_feedback>"
    req = request(deps, "writer", "writer.md", text, Report)
    return (await deps.llm_call(req)).parsed or {}


def passes(v: EvalResponse, cfg: LoopCfg) -> bool:
    scores = list(v.scores.model_dump().values())
    return (
        all(v.checks.model_dump().values())
        and min(scores) >= cfg.pass_min_score
        and mean(scores) >= cfg.pass_mean_score
    )


async def evaluate(case_text: str, draft: Report, metrics, deps: Deps) -> EvalVerdict:
    """Separate evaluator on the rendered draft; code decides pass from checks and scores."""
    rendered = draft.model_copy(
        update={
            "headline": render_text(draft.headline, metrics),
            "summary": render_text(draft.summary, metrics),
            "review_note": render_text(draft.review_note, metrics),
            "key_points": [
                p.model_copy(update={"text": render_text(p.text, metrics)})
                for p in draft.key_points
            ],
        }
    )
    text = f"{case_text}\n<draft>\n{rendered.model_dump_json()}\n</draft>"
    resp = await deps.llm_call(request(deps, "evaluator", "evaluator.md", text, EvalResponse))
    try:
        r = EvalResponse.model_validate(resp.parsed or {})
    except ValidationError as e:
        return EvalVerdict(
            verdict="revise",
            checks={k: False for k in EvalResponse.model_fields["checks"].annotation.model_fields},
            scores={"clarity": 1, "usefulness_for_reviewer": 1, "tone": 1},
            issues=[f"unparseable evaluator verdict: {e.error_count()} errors"],
        )
    return EvalVerdict(**r.model_dump(), passed=passes(r, deps.settings.loop))
