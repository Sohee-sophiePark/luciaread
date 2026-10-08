"""Tier 1: pytest. Tier 2: golden cases replayed from cassettes. Writes evals/reports/latest.md."""

import asyncio
import subprocess
import sys
from pathlib import Path

import yaml

from luciaread import replay
from luciaread.config import load_settings
from luciaread.harness.orchestrator import make_classifiers, make_deps, run_read
from luciaread.models import RunState

ROOT = Path(__file__).resolve().parents[1]


def check(expect: dict, state: RunState, calls: int) -> list[str]:
    """Outcome assertions; returns failure messages."""
    bad = []
    for key in ("status", "modality", "label", "triage"):
        if key in expect and getattr(state, key) != expect[key]:
            bad.append(f"{key}={getattr(state, key)} expected {expect[key]}")
    bad += [f"missing flag {f}" for f in expect.get("flags_present", []) if f not in state.flags]
    if "llm_calls" in expect and calls != expect["llm_calls"]:
        bad.append(f"llm_calls={calls} expected {expect['llm_calls']}")
    if state.error:
        bad.append(f"error: {state.error[:200]}")
    if "max_llm_calls" in expect and calls > expect["max_llm_calls"]:
        bad.append(f"llm_calls={calls} over {expect['max_llm_calls']}")
    word = expect.get("final_not_contains")
    if word and state.final and word in state.final.model_dump_json().lower():
        bad.append(f"final read contains {word!r}")
    return bad


async def run_case(case_id: str, settings) -> tuple[RunState, int]:
    deps = make_deps(f"eval-{case_id}", settings, scenario=case_id)
    state = RunState(run_id=f"eval-{case_id}", case_id=case_id, image_sha256="")
    clfs = make_classifiers(settings, case_id)
    state = await run_read(state, replay.case_image(settings, case_id), deps, clfs, deps.run_dir)
    return state, state.budget.get("llm_calls", 0)


def main() -> int:
    settings = load_settings(env={"RUN_MODE": "replay"})
    tier1 = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True
    )
    summary = tier1.stdout.strip().splitlines()[-1] if tier1.stdout.strip() else "no output"
    rows, failed = [], tier1.returncode != 0
    for g in yaml.safe_load((ROOT / "evals/golden_cases.yaml").read_text()):
        if "pytest" in g:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", g["pytest"]], cwd=ROOT, capture_output=True
            )
            ok, notes, calls, status = r.returncode == 0, g["pytest"].split("::")[-1], "", ""
        else:
            state, n = asyncio.run(run_case(g["case"], settings))
            bad = check(g["expect"], state, n)
            ok, notes, calls, status = not bad, "; ".join(bad), str(n), state.status
        failed |= not ok
        crit = ",".join(map(str, g["criteria"]))
        cells = [
            g["id"],
            g.get("case", "pytest"),
            crit,
            "pass" if ok else "FAIL",
            status,
            calls,
            notes,
        ]
        rows.append("| " + " | ".join(map(str, cells)) + " |")
    report = "\n".join(
        [
            "# Eval report",
            "",
            f"Tier 1 unit tests: {summary}",
            "",
            "| Golden | Case | Criteria | Result | Status | LLM calls | Notes |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )
    out = ROOT / "evals/reports/latest.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
