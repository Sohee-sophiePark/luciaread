"""Acceptance criterion 1: committed model-card metrics meet the floors agreed on 2026-10-08."""

import json
from pathlib import Path

CARD = json.loads((Path(__file__).parents[1] / "reports/model_card.json").read_text())


def test_cxr_floors() -> None:
    c = CARD["cxr"]
    assert c["auc"] >= 0.99 and c["sensitivity"] >= 0.93 and c["ece_after"] <= 0.05


def test_oct_floors() -> None:
    c = CARD["oct"]
    assert c["accuracy"] >= 0.90 and c["auc"] >= 0.98 and c["ece_after"] <= 0.05


def test_shuffled_label_controls_near_chance() -> None:
    assert CARD["cxr_control"]["auc"] <= 0.6 and CARD["oct_control"]["auc"] <= 0.6


def test_gradcam_randomization_sanity_check_passes() -> None:
    for t in ("cxr", "oct"):
        assert CARD[t]["audit"]["gradcam_randomization"]["mean_spearman_trained_vs_random"] < 0.5
