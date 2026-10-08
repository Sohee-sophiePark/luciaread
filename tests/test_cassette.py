"""Cassette round trip: record through a scripted client, replay identical, miss raises."""

from pathlib import Path

import pytest

from luciaread.llm.base import LLMRequest, LLMResponse, Message
from luciaread.llm.cassette import CassetteClient, CassetteMiss
from luciaread.llm.scripted import ScriptedClient


def req(purpose: str, text: str) -> LLMRequest:
    return LLMRequest(
        model="m", system="s", messages=[Message(role="user", text=text)], purpose=purpose
    )


async def test_record_then_replay_identical_and_order_independent(tmp_path: Path) -> None:
    scripted = ScriptedClient()
    scripted.add("a", LLMResponse(text="A", tokens_in=1, tokens_out=2))
    scripted.add("b", LLMResponse(text="B"))
    rec = CassetteClient(tmp_path / "c.jsonl", "record", scripted)
    ra = await rec.generate(req("a", "x"))
    await rec.generate(req("b", "y"))
    rep = CassetteClient(tmp_path / "c.jsonl", "replay")
    pb, pa = await rep.generate(req("b", "y")), await rep.generate(req("a", "x"))
    assert (pa.text, pa.tokens_in, pa.tokens_out, pa.source) == ("A", 1, 2, "cassette")
    assert pb.text == "B" and ra.source == "scripted"


async def test_miss_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(CassetteMiss, match="make record"):
        await CassetteClient(tmp_path / "none.jsonl", "replay").generate(req("a", "x"))


async def test_scripted_requires_a_queued_response() -> None:
    with pytest.raises(AssertionError):
        await ScriptedClient().generate(req("a", "x"))


async def test_identical_requests_replay_in_recorded_order(tmp_path: Path) -> None:
    scripted = ScriptedClient()
    scripted.add("e", LLMResponse(text="first"), LLMResponse(text="second"))
    rec = CassetteClient(tmp_path / "c.jsonl", "record", scripted)
    for _ in range(2):
        await rec.generate(req("e", "same"))
    rep = CassetteClient(tmp_path / "c.jsonl", "replay")
    assert [(await rep.generate(req("e", "same"))).text for _ in range(3)] == [
        "first",
        "second",
        "second",
    ]


def test_image_key_ignores_png_encoding() -> None:
    """Same pixels, different PNG bytes (as across zlib builds) → same cassette key."""
    import io

    from PIL import Image

    from luciaread.llm.cassette import request_key

    im = Image.open(io.BytesIO(open("samples/S1.png", "rb").read())).convert("RGB")
    a, b = io.BytesIO(), io.BytesIO()
    im.save(a, format="PNG", compress_level=1)
    im.save(b, format="PNG", compress_level=9, optimize=True)
    assert a.getvalue() != b.getvalue()
    req = lambda img: LLMRequest(model="m", system="s", messages=[], purpose="p", image=img)  # noqa: E731
    assert request_key(req(a.getvalue())) == request_key(req(b.getvalue()))
