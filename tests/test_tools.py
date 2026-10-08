"""Input gate, quality, classifier tools, anatomy mask and record/replay of model outputs."""

import io

import numpy as np
import pytest
import scripted_fixtures as sf
from PIL import Image

from luciaread.config import load_settings
from luciaread.tools import image, model

RULES = load_settings(env={}).rules


def png_with_exif() -> bytes:
    im = Image.open(io.BytesIO(sf.scan_png()))
    exif = Image.Exif()
    exif[0x010E] = "PATIENT NAME"  # ImageDescription
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_load_scan_strips_metadata_and_downsizes() -> None:
    scan, g = image.load_scan(png_with_exif(), RULES)
    assert g.passed and b"PATIENT NAME" not in scan.png and scan.gray.shape == (224, 224)
    assert Image.open(io.BytesIO(scan.png)).getexif() == {}


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"x" * 11 * 1024 * 1024, "file too large"),
        (b"GIF89a....", "image could not be decoded"),
        (sf.scan_png(size=64), "image too small"),
    ],
)
def test_load_scan_rejects(data, reason) -> None:
    scan, g = image.load_scan(data, RULES)
    assert scan is None and g.violations == [reason]


def test_quality_flags_flat_image_only() -> None:
    good, _ = image.load_scan(sf.scan_png(), RULES)
    flat, _ = image.load_scan(sf.scan_png(flat=True), RULES)
    assert image.quality(good, "cxr", RULES).flags == []
    assert image.quality(flat, "cxr", RULES).flags[0].flag_id == "low_quality"


def test_classify_uses_cxr_threshold_and_flags_low_confidence() -> None:
    scan, _ = image.load_scan(sf.scan_png(), RULES)
    clf = sf.FakeClassifier(["NORMAL", "PNEUMONIA"], [0.2, -0.2], threshold=0.3)
    r = model.classify(scan, clf, "cxr", RULES)
    assert r.data["label"] == "PNEUMONIA"  # p(PNEUMONIA) ≈ 0.40 ≥ threshold 0.3
    assert [f.flag_id for f in r.flags] == ["low_confidence"]
    assert sum(m.value for m in r.metrics) == pytest.approx(100, abs=0.2)


def test_explain_flags_attention_outside_anatomy() -> None:
    scan, _ = image.load_scan(sf.scan_png(), RULES)
    inside, _ = model.explain(scan, sf.FakeClassifier(["a", "b"], [0, 1]), "cxr", 1, RULES)
    corner, cam = model.explain(
        scan, sf.FakeClassifier(["a", "b"], [0, 1], off_target=True), "cxr", 1, RULES
    )
    assert inside.flags == [] and corner.flags[0].flag_id == "attention_off_target"
    assert model.heatmap_png(cam).startswith(b"\x89PNG")


def test_oct_mask_covers_bright_band() -> None:
    g = np.zeros((224, 224), np.uint8)
    g[100:130] = 200
    m = model.anatomy_mask(g, "oct")
    assert m[110].all() and not m[10].any()


def test_recorded_outputs_replay_identically() -> None:
    scan, _ = image.load_scan(sf.scan_png(), RULES)
    store: dict = {}
    rec = model.RecordingClassifier(sf.FakeClassifier(["NORMAL", "PNEUMONIA"], [1, 2], 0.3), store)
    a = model.classify(scan, rec, "cxr", RULES)
    a_att, _ = model.explain(scan, rec, "cxr", 1, RULES)
    rep = model.ReplayClassifier(["NORMAL", "PNEUMONIA"], store)
    assert model.classify(scan, rep, "cxr", RULES) == a
    assert model.explain(scan, rep, "cxr", 1, RULES)[0] == a_att


def test_weights_must_match_model_card(tmp_path) -> None:
    import hashlib
    import json

    (tmp_path / "weights").mkdir()
    (tmp_path / "reports").mkdir()
    pt = tmp_path / "weights" / "cxr.pt"
    pt.write_bytes(b"trained")
    entry = {
        "classes": ["NORMAL", "PNEUMONIA"],
        "temperature": 0.8,
        "threshold": 0.7,
        "weights_sha256": hashlib.sha256(b"trained").hexdigest(),
    }
    (tmp_path / "reports" / "model_card.json").write_text(json.dumps({"cxr": entry}))
    assert model.card_entry("cxr", tmp_path / "weights")["threshold"] == 0.7
    pt.write_bytes(b"retrained")
    with pytest.raises(ValueError, match="does not match"):
        model.card_entry("cxr", tmp_path / "weights")
