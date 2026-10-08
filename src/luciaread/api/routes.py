"""HTTP API: demo cases, start a read (case or live upload), state, SSE trace, images, sign-off."""

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request, UploadFile
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from luciaread import replay
from luciaread.harness.orchestrator import (
    apply_signoff,
    make_classifiers,
    make_deps,
    new_run_id,
    run_read,
)
from luciaread.harness.trace import load_events
from luciaread.models import RunState, SignOff

SafeId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,80}$")]  # ids become file paths
router = APIRouter(prefix="/api")


@router.get("/health")
def health(request: Request) -> dict:
    s = request.app.state.settings
    return {"status": "ok", "run_mode": s.run_mode, "models": s.models}


@router.get("/cases")
def list_cases(request: Request) -> list[dict]:
    return [c for c in replay.cases(request.app.state.settings) if not c.get("hidden")]


@router.get("/cases/{case_id}/image.png")
def case_image(case_id: SafeId, request: Request) -> FileResponse:
    s = request.app.state.settings
    try:
        return FileResponse(s.path("samples") / replay.case(s, case_id)["file"])
    except StopIteration as e:
        raise HTTPException(404, "unknown case") from e


def _start(request: Request, case_id: str, data: bytes, scenario: str | None) -> dict:
    app = request.app.state
    s = app.settings
    if s.run_mode == "live":
        app.clfs = app.clfs or make_classifiers(s)
        clfs = app.clfs
    else:
        clfs = make_classifiers(s, scenario)
    run_id = new_run_id(case_id)
    deps = make_deps(run_id, s, app.llm, scenario)
    state = RunState(run_id=run_id, case_id=case_id, image_sha256="")
    task = asyncio.create_task(run_read(state, data, deps, clfs, deps.run_dir))
    app.runs[run_id] = (state, deps, task)
    return {"run_id": run_id}


@router.post("/reads/case/{case_id}")
async def read_case(case_id: SafeId, request: Request) -> dict:
    s = request.app.state.settings
    try:
        data = replay.case_image(s, case_id)
    except StopIteration as e:
        raise HTTPException(404, "unknown case") from e
    return _start(request, case_id, data, None if s.run_mode == "live" else case_id)


@router.post("/reads/upload")
async def read_upload(file: UploadFile, request: Request) -> dict:
    s = request.app.state.settings
    if s.run_mode != "live":
        raise HTTPException(409, "uploads need RUN_MODE=live on your own laptop")
    data = await file.read(int(s.rules.max_upload_mb * 1024 * 1024) + 1)
    return _start(request, "UPLOAD", data, None)


def _state(run_id: str, request: Request) -> RunState:
    app = request.app.state
    if run_id in app.runs:
        return app.runs[run_id][0]
    path = app.settings.path("runs") / run_id / "state.json"
    if not path.exists():
        raise HTTPException(404, "unknown run")
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))


@router.get("/runs/{run_id}")
def get_run(run_id: SafeId, request: Request) -> RunState:
    return _state(run_id, request)


@router.get("/runs/{run_id}/events")
async def run_events(run_id: SafeId, request: Request) -> EventSourceResponse:
    app = request.app.state
    path = app.settings.path("runs") / run_id / "trace.jsonl"
    if run_id not in app.runs and not path.exists():
        raise HTTPException(404, "unknown run")

    async def gen() -> AsyncIterator[dict]:
        if run_id in app.runs:
            q = app.runs[run_id][1].trace.subscribe()
            while (e := await q.get()) is not None:
                yield {"id": str(e.seq), "data": e.model_dump_json()}
        else:
            for e in load_events(path):
                yield {"id": str(e.seq), "data": e.model_dump_json()}

    return EventSourceResponse(gen())


@router.get("/runs/{run_id}/{name}.png")
def run_image(run_id: SafeId, name: str, request: Request) -> FileResponse:
    path = request.app.state.settings.path("runs") / run_id / f"{name}.png"
    if name not in ("image", "heatmap") or not path.exists():
        raise HTTPException(404, "no such image")
    return FileResponse(path)


@router.post("/runs/{run_id}/signoff")
def signoff(run_id: SafeId, decision: SignOff, request: Request) -> RunState:
    app = request.app.state
    state = _state(run_id, request)
    run_dir = app.settings.path("runs") / run_id
    trace = app.runs[run_id][1].trace if run_id in app.runs else None
    if trace is None:
        raise HTTPException(409, "run is no longer active")
    try:
        return apply_signoff(state, decision, run_dir, trace)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


def loopback_only(request: Request) -> None:
    if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
        raise HTTPException(403, "developer console is local only")


dev = APIRouter(prefix="/api/dev", dependencies=[Depends(loopback_only)])


@dev.get("/runs")
def dev_runs(request: Request) -> list[dict]:
    runs = request.app.state.settings.path("runs")
    states = sorted(runs.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in states[:50]:
        s = RunState.model_validate_json(p.read_text(encoding="utf-8"))
        out.append(
            {
                "run_id": s.run_id,
                "case_id": s.case_id,
                "status": s.status,
                "label": s.label,
                "triage": s.triage,
                "budget": s.budget,
            }
        )
    return out


@dev.get("/runs/{run_id}")
def dev_run(run_id: SafeId, request: Request) -> dict:
    s = request.app.state.settings
    path = s.path("runs") / run_id / "trace.jsonl"
    events = load_events(path) if path.exists() else []
    done = [e.payload for e in events if e.type == "llm_call_finished"]
    cost = sum(s.pricing.cost(p["model"], p["tokens_in"], p["tokens_out"]) for p in done)
    return {"state": _state(run_id, request), "events": events, "cost_usd": cost}
