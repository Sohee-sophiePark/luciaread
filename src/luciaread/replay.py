"""Demo cases (samples/cases.yaml), recorded replays, and the static export for the public demo."""

import base64
import datetime as dt
import json
import shutil
from pathlib import Path

import yaml

from luciaread.config import Settings
from luciaread.harness.trace import TraceEvent
from luciaread.models import RunState


def cases(settings: Settings) -> list[dict]:
    """Case list: id, title, file, source, changes, approve, hidden."""
    return yaml.safe_load((settings.path("samples") / "cases.yaml").read_text(encoding="utf-8"))


def case(settings: Settings, case_id: str) -> dict:
    return next(c for c in cases(settings) if c["id"] == case_id)


def case_image(settings: Settings, case_id: str) -> bytes:
    return (settings.path("samples") / case(settings, case_id)["file"]).read_bytes()


def write_replay(settings: Settings, c: dict, state: RunState, events: list[TraceEvent]) -> Path:
    """`replays/<case>.json`: case, recorded events and terminal state, plus image and heatmap."""
    run_dir = settings.path("runs") / state.run_id
    heat = run_dir / "heatmap.png"
    out = {
        "case": c,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "events": [e.model_dump(mode="json") for e in events],
        "state": state.model_dump(mode="json"),
        "heatmap_png_b64": base64.b64encode(heat.read_bytes()).decode() if heat.exists() else None,
    }
    path = settings.path("replays") / f"{c['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return path


def export_static(settings: Settings, web_public: Path) -> int:
    """Write `static-data.json` and copy case images and heatmaps into `web_public/cases/`."""
    out_dir = web_public / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded = []
    for c in cases(settings):
        path = settings.path("replays") / f"{c['id']}.json"
        if c.get("hidden") or not path.exists():
            continue
        rep = json.loads(path.read_text(encoding="utf-8"))
        shutil.copyfile(settings.path("samples") / c["file"], out_dir / c["file"])
        heat = None
        if rep["heatmap_png_b64"]:
            heat = f"{c['id']}_heatmap.png"
            (out_dir / heat).write_bytes(base64.b64decode(rep["heatmap_png_b64"]))
        recorded.append(
            {
                "case": c,
                "image": f"cases/{c['file']}",
                "heatmap": heat and f"cases/{heat}",
                "events": rep["events"],
                "state": rep["state"],
            }
        )
    (web_public / "static-data.json").write_text(json.dumps({"cases": recorded}), encoding="utf-8")
    return len(recorded)
