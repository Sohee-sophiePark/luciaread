"""Patient-level split (incl. the committed manifest) and metric functions."""

from pathlib import Path

import numpy as np
import pytest

from luciaread.ml import metrics as M
from luciaread.ml.split import assign, load, patient_id

MANIFEST = Path(__file__).parents[1] / "data/splits/manifest.csv"


@pytest.mark.parametrize(
    ("modality", "name", "pid"),
    [
        ("cxr", "person1_virus_6.jpeg", "cxr-person1"),
        ("cxr", "person12_bacteria_47.jpeg", "cxr-person12"),
        ("cxr", "IM-0031-0001.jpeg", "cxr-IM-0031"),
        ("cxr", "NORMAL2-IM-0272-0001-0002.jpeg", "cxr-NORMAL2-IM-0272"),
        ("oct", "CNV-1234-5.jpeg", "oct-1234"),
    ],
)
def test_patient_id(modality, name, pid) -> None:
    assert patient_id(modality, name) == pid


def test_patient_id_rejects_unknown_pattern() -> None:
    with pytest.raises(ValueError):
        patient_id("oct", "scan.jpeg")


def test_assign_keeps_patients_whole() -> None:
    rows = [
        {"modality": "oct", "label": "CNV", "patient": f"p{i % 40}", "path": str(i)}
        for i in range(400)
    ]
    seen: dict[str, set] = {}
    for r in assign(rows):
        seen.setdefault(r["patient"], set()).add(r["split"])
    assert all(len(s) == 1 for s in seen.values())


def test_committed_manifest_is_patient_disjoint() -> None:
    seen: dict[str, set] = {}
    for r in load(MANIFEST):
        seen.setdefault(r["patient"], set()).add(r["split"])
    assert seen and all(len(s) == 1 for s in seen.values())


def test_auc_and_threshold() -> None:
    s, pos = np.array([0.1, 0.4, 0.35, 0.8]), np.array([False, False, True, True])
    assert M.auc(s, pos) == pytest.approx(0.75)
    p = np.linspace(0.01, 1, 100)
    t = M.threshold_for_sensitivity(p, np.ones(100, bool), 0.95)
    assert (p >= t).mean() >= 0.95


def test_ece_zero_when_calibrated_and_temperature_free_softmax() -> None:
    probs = np.array([[0.0, 1.0]] * 10)
    assert M.ece(probs, np.ones(10, int))[0] == pytest.approx(0.0)
    z = np.array([[1.0, 3.0]])
    assert M.softmax(z, 2.0) == pytest.approx(M.softmax(z / 2.0))
