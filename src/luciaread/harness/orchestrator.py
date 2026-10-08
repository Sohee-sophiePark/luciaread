"""Read pipeline: G0 → G1 modality → parallel specialists → writer ⇄ G5 + evaluator → sign-off."""

import asyncio
import datetime as dt
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from luciaread import agents
from luciaread.config import RateLimitsCfg, Settings
from luciaread.harness.budget import BudgetExhausted, RunBudget
from luciaread.harness.deps import Deps
from luciaread.harness.gates import modality_gate, output_gates
from luciaread.harness.trace import TraceBus
from luciaread.llm.base import LLMClient
from luciaread.llm.cassette import CassetteClient
from luciaread.llm.limiter import RateLimiter
from luciaread.models import (
    CLASSES,
    DISPLAY,
    GateResult,
    RunState,
    RunStatus,
    SignOff,
)
from luciaread.render import render
from luciaread.tools import image, model
from luciaread.tools.model import Classifier

TASKS = {"modality": ["cxr", "oct"], **CLASSES}
REJECT_MESSAGE = (
    "LuciaRead reads frontal pediatric chest X-rays and retinal OCT B-scans only. This image was "
    "not accepted: {reasons}."
)


@dataclass
class Classifiers:
    modality: Classifier
    cxr: Classifier
    oct: Classifier
    store: dict | None = None  # recorded outputs (record mode)
    path: Path | None = None

    def save(self) -> None:
        if self.store is not None and self.path:
            self.path.write_text(json.dumps(self.store) + "\n", encoding="utf-8")


def make_classifiers(settings: Settings, scenario: str | None = None) -> Classifiers:
    """Live: torch weights. Record: torch + store. Replay: `cassettes/<scenario>.models.json`."""
    path = settings.path("cassettes") / f"{scenario}.models.json" if scenario else None
    if settings.run_mode == "replay":
        store = json.loads(path.read_text(encoding="utf-8"))
        return Classifiers(*(model.ReplayClassifier(c, store) for c in TASKS.values()))
    torch_clfs = [model.TorchClassifier(t, settings.path("weights")) for t in TASKS]
    if settings.run_mode == "record" and path:
        store: dict = {}
        recs = [model.RecordingClassifier(c, store) for c in torch_clfs]
        return Classifiers(*recs, store=store, path=path)
    return Classifiers(*torch_clfs)


def make_deps(
    run_id: str, settings: Settings, llm: LLMClient | None = None, scenario: str | None = None
) -> Deps:
    """LLM client for `settings.run_mode`; record/replay use `cassettes/<scenario>.jsonl`."""
    if llm is None:
        if settings.run_mode == "live":
            from luciaread.llm.gemini import GeminiClient

            llm = GeminiClient()
        else:
            inner = None
            if settings.run_mode == "record":
                from luciaread.llm.gemini import GeminiClient

                inner = GeminiClient()
            path = settings.path("cassettes") / f"{scenario or run_id}.jsonl"
            llm = CassetteClient(path, settings.run_mode, inner)
    limits = settings.rate_limits
    if settings.run_mode == "replay":  # nothing reaches the API, so no pacing
        limits = RateLimitsCfg.model_validate({m: {"rpm": 10**6} for m in settings.models.values()})
    limiter = RateLimiter(limits, settings.retry, settings.budget.per_call_timeout_seconds)
    run_dir = settings.path("runs") / run_id
    trace = TraceBus(run_id, run_dir / "trace.jsonl")
    return Deps(settings, llm, limiter, RunBudget(settings.budget), trace, run_dir)


def new_run_id(case_id: str) -> str:
    return f"{case_id}-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def run_metadata(settings: Settings) -> dict:
    """Provenance: models, prompt and settings hashes, run mode."""
    prompts = b"".join(p.read_bytes() for p in sorted(settings.path("prompts").glob("*.md")))
    return {
        "models": dict(settings.models),
        "prompts_sha": hashlib.sha256(prompts).hexdigest()[:12],
        "settings_sha": hashlib.sha256(settings.model_dump_json().encode()).hexdigest()[:12],
        "run_mode": settings.run_mode,
    }


def save_state(state: RunState, run_dir: Path) -> None:
    """Atomic write of `<run_dir>/state.json`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "state.tmp"
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, run_dir / "state.json")


def _gate(state: RunState, deps: Deps, g: GateResult) -> bool:
    state.gate_results.append(g)
    deps.trace.emit("gate_result", "system", g.model_dump(), "info" if g.passed else "warn")
    return g.passed


def _reject(state: RunState, deps: Deps, g: GateResult) -> RunState:
    state.status = RunStatus.REJECTED
    state.message = REJECT_MESSAGE.format(reasons="; ".join(g.violations))
    deps.trace.emit("run_rejected", "system", {"reasons": g.violations}, "warn")
    return state


async def run_read(
    state: RunState, data: bytes, deps: Deps, clfs: Classifiers, run_dir: Path
) -> RunState:
    """Run the pipeline; never raises. Terminal state saved to `<run_dir>/state.json`."""
    state.meta = run_metadata(deps.settings)
    try:
        await _steps(state, data, deps, clfs, run_dir)
    except BudgetExhausted as e:
        state.status, state.error = RunStatus.DEGRADED, str(e)
        deps.trace.emit("budget_exhausted", "system", {"which": e.which}, "error")
    except Exception as e:  # noqa: BLE001 — every failure ends as a FAILED state
        state.status, state.error = RunStatus.FAILED, f"{type(e).__name__}: {e}"
        deps.trace.emit("run_failed", "system", {"error": state.error}, "error")
    finally:
        state.budget = deps.budget.snapshot()
        deps.trace.emit("run_completed", "system", {"status": state.status, **deps.trace.totals()})
        deps.trace.close()
        save_state(state, run_dir)
        clfs.save()
        if close := getattr(deps.llm, "aclose", None):
            await close()
    return state


async def _steps(state, data, deps: Deps, clfs: Classifiers, run_dir: Path) -> None:
    rules, t = deps.settings.rules, deps.trace
    t.emit("run_started", "system", {"case_id": state.case_id})
    scan, g0 = image.load_scan(data, rules)
    if not _gate(state, deps, g0):
        _reject(state, deps, g0)
        return
    state.image_sha256 = scan.sha256
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "image.png").write_bytes(scan.png)
    state.status = RunStatus.INPUT_GATED
    if not (g := modality_gate(scan.saturation, None, None, rules)).passed:
        _gate(state, deps, g)
        _reject(state, deps, g)
        return
    mod = await asyncio.to_thread(model.modality, scan, clfs.modality)
    t.emit("tool_call", "router", {"tool": mod.tool, "data": mod.data})
    route = await agents.route(scan.png, deps)
    state.route = route
    t.emit("route_decided", "router", route.model_dump() if route else {"modality": None})
    if not _gate(state, deps, modality_gate(scan.saturation, mod.data["modality"], route, rules)):
        _reject(state, deps, state.gate_results[-1])
        return
    state.modality, state.status = route.modality, RunStatus.ANALYZING
    await _analyse(state, scan, deps, clfs, run_dir)
    await _write(state, deps)


async def _analyse(state: RunState, scan, deps: Deps, clfs: Classifiers, run_dir: Path) -> None:
    """Three read-only LLM specialists in parallel. Model and quality run their tools in code
    first and write findings about them; the visual describer reads the image."""
    rules, t, m = deps.settings.rules, deps.trace, state.modality
    clf = getattr(clfs, m)

    def model_tools():
        cls = model.classify(scan, clf, m, rules)
        att, cam = model.explain(scan, clf, m, cls.data["label_index"], rules)
        return cls, att, cam

    async def model_specialist():
        cls, att, cam = await asyncio.to_thread(model_tools)
        fallback = agents.model_findings(cls, att, cls.data["label"])
        return (cls, att, cam), await agents.specialist("model", [cls, att], m, deps, fallback)

    async def quality_specialist():
        q = await asyncio.to_thread(image.quality, scan, m, rules)
        return q, await agents.specialist("quality", [q], m, deps, agents.quality_findings(q))

    for agent in ("model", "quality", "visual"):
        t.emit("agent_started", agent, {})
    ((cls, att, cam), mod_out), (q, qual_out), visual = await asyncio.gather(
        model_specialist(), quality_specialist(), agents.describe(scan.png, m, deps)
    )
    (run_dir / "heatmap.png").write_bytes(model.heatmap_png(cam))
    state.label = cls.data["label"]
    for agent, r in (("model", cls), ("model", att), ("quality", q)):
        t.emit("tool_call", agent, {"tool": r.tool, "flags": [f.flag_id for f in r.flags]})
        state.metrics |= {x.key: x for x in r.metrics}
        state.flags |= {f.flag_id: f for f in r.flags}
    degraded = {"model": mod_out[1], "quality": qual_out[1], "visual": visual is None}
    state.gate_results += mod_out[2] + qual_out[2]
    state.findings = mod_out[0] + qual_out[0]
    if visual:
        state.visual = visual
        vf, vflags = agents.visual_findings(visual, deps.settings.injection.redact)
        state.findings += vf
        state.flags |= {f.flag_id: f for f in vflags}
        for f in vflags:
            t.emit(f.flag_id, "visual", {"message": f.message}, "warn")
    review = any(f.severity == "review" for f in state.flags.values())
    state.triage = "needs_review" if review else "routine"
    for agent in ("model", "quality", "visual"):
        n = sum(f.agent == agent for f in state.findings)
        t.emit("agent_finished", agent, {"findings": n, "degraded": degraded[agent]})
    t.emit("triage_set", "system", {"triage": state.triage, "flags": sorted(state.flags)})


async def _write(state: RunState, deps: Deps) -> None:
    """Writer drafts; G5 then the evaluator review; at most `max_revisions` revisions."""
    s, t = deps.settings, deps.trace
    state.status = RunStatus.WRITING
    case = agents.writer_input(
        state.modality,
        state.label,
        state.triage,
        state.metrics,
        state.flags,
        state.findings,
        state.visual,
    )
    feedback, last_ok = None, None
    for attempt in range(s.loop.max_revisions + 1):
        if attempt:
            state.revision_count = attempt
            t.emit("revision_requested", "writer", {"source": feedback["source"], "n": attempt})
        raw = await agents.write(case, deps, feedback)
        draft, g5 = output_gates(raw, state.modality, state.label, state.metrics, state.flags)
        t.emit("draft_created", "writer", {"attempt": attempt + 1})
        if not _gate(state, deps, g5):
            feedback = {"source": "gates", "items": g5.violations, "draft": raw}
            continue
        state.drafts.append(draft)
        last_ok = draft
        verdict = await agents.evaluate(case, draft, state.metrics, deps)
        state.eval_verdicts.append(verdict)
        t.emit("evaluator_verdict", "evaluator", verdict.model_dump())
        if verdict.passed:
            state.final = render(draft, state.modality, state.label, state.triage, state.metrics)
            state.status = RunStatus.AWAITING_SIGNOFF
            t.emit("awaiting_signoff", "system", {"triage": state.triage})
            return
        feedback = {"source": "evaluator", "items": verdict.issues, "draft": raw}
    if last_ok:
        state.final = render(last_ok, state.modality, state.label, state.triage, state.metrics)
    state.status = RunStatus.NEEDS_CLINICIAN_REVIEW
    state.message = (
        "The draft did not pass review within the revision limit; read the image directly."
    )
    t.emit("needs_clinician_review", "system", {"revisions": state.revision_count}, "warn")


def apply_signoff(state: RunState, decision: SignOff, run_dir: Path, trace: TraceBus) -> RunState:
    """Human step: sign off (writes signed_report.json) or return; only from a reviewable state."""
    if state.status not in (RunStatus.AWAITING_SIGNOFF, RunStatus.NEEDS_CLINICIAN_REVIEW):
        raise ValueError(f"run is {state.status}, not awaiting sign-off")
    state.signoff = decision.model_copy(update={"at": dt.datetime.now(dt.UTC)})
    trace.emit("signoff_recorded", "human", {"decision": decision.decision, "note": decision.note})
    if decision.decision == "sign_off":
        report = {
            "run_id": state.run_id,
            "image_sha256": state.image_sha256,
            "read": state.final.model_dump() if state.final else None,
            "signoff": state.signoff.model_dump(mode="json"),
        }
        (run_dir / "signed_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        trace.emit("signed_report_written", "human", {"label": DISPLAY.get(state.label or "", "")})
        state.status = RunStatus.SIGNED_OFF
    else:
        state.status = RunStatus.RETURNED
    save_state(state, run_dir)
    return state
