"""No API key material in the repo; .env, internal docs, runs, weights and raw data ignored."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
KEY = re.compile(r"AIza[0-9A-Za-z_\-]{30,}|GEMINI_API_KEY\s*=\s*\S{8,}")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def test_no_key_material_in_repo() -> None:
    files = git("ls-files", "--cached", "--others", "--exclude-standard").stdout.split()
    hits = [
        f
        for f in files
        if (ROOT / f).is_file() and KEY.search((ROOT / f).read_text(errors="ignore"))
    ]
    assert hits == []


def test_env_and_internal_docs_are_ignored() -> None:
    for name in (".env", "INTERNAL.md", "CLAUDE.md", "runs/x", "weights/x.pt", "data/raw/x"):
        assert git("check-ignore", "-q", name).returncode == 0, name
