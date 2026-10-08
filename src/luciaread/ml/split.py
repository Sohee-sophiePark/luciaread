"""Patient-level train/val/test split manifest for the Kermany 2018 images."""

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
OCT_CAPS = {"train": 2000, "val": 300, "test": 500}  # images per class
FIELDS = ("path", "modality", "label", "patient", "split")


def patient_id(modality: str, name: str) -> str:
    """Patient key parsed from a Kermany filename; raises on an unknown pattern."""
    pats = {
        "cxr": r"^(person\d+)_|^((?:NORMAL\d-)?IM-\d+)-",
        "oct": r"^[A-Z]+-(\d+)-\d+\.jpeg$",
    }
    m = re.match(pats[modality], name)
    if not m:
        raise ValueError(f"unparsed filename: {name}")
    return f"{modality}-{next(g for g in m.groups() if g)}"


def scan(raw: Path) -> list[dict]:
    """All images under data/raw with modality, label and patient; official split ignored."""
    roots = {"cxr": raw / "cxr" / "chest_xray", "oct": raw / "oct" / "OCT2017"}
    rows = []
    for modality, root in roots.items():
        for p in sorted(root.glob("*/*/*.jpeg")):
            rows.append(
                {
                    "path": p.relative_to(raw).as_posix(),
                    "modality": modality,
                    "label": p.parent.name,
                    "patient": patient_id(modality, p.name),
                }
            )
    return rows


def assign(rows: list[dict], seed: int = 7) -> list[dict]:
    """Assign whole patients to splits, stratified by each patient's majority label."""
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient"]].append(r)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for pid, rs in by_patient.items():
        labels = [r["label"] for r in rs]
        strata[(rs[0]["modality"], max(set(labels), key=labels.count))].append(pid)
    rng = random.Random(seed)
    for key in sorted(strata):
        pids = sorted(strata[key])
        rng.shuffle(pids)
        start = 0
        for i, (split, frac) in enumerate(SPLITS):
            end = len(pids) if i == len(SPLITS) - 1 else start + round(frac * len(pids))
            for pid in pids[start:end]:
                for r in by_patient[pid]:
                    r["split"] = split
            start = end
    return rows


def cap_oct(rows: list[dict], seed: int = 7) -> list[dict]:
    """Keep all CXR rows; sample OCT rows down to OCT_CAPS per split and class."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    out = []
    for r in rows:
        (groups[(r["split"], r["label"])] if r["modality"] == "oct" else out).append(r)
    for (split, _), rs in sorted(groups.items()):
        out += rng.sample(rs, min(OCT_CAPS[split], len(rs)))
    return sorted(out, key=lambda r: r["path"])


def build(raw: Path, out: Path) -> list[dict]:
    """Write the manifest CSV and return its rows."""
    rows = cap_oct(assign(scan(raw)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))
