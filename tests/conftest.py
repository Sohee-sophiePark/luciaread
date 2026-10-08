"""Shared fixtures: scripted LLM deps with unlimited rpm, so tests never sleep or call a network."""

import pytest

from luciaread.config import RateLimitsCfg, load_settings
from luciaread.harness.budget import RunBudget
from luciaread.harness.deps import Deps
from luciaread.harness.trace import TraceBus
from luciaread.llm.limiter import RateLimiter
from luciaread.llm.scripted import ScriptedClient


@pytest.fixture
def settings():
    return load_settings(env={})


@pytest.fixture
def deps(settings) -> Deps:
    models = [*settings.models.values(), *settings.fallback_models]
    limits = RateLimitsCfg.model_validate({m: {"rpm": 10**6} for m in models})
    limiter = RateLimiter(limits, settings.retry, 5.0, jitter=lambda: 0.0)
    return Deps(settings, ScriptedClient(), limiter, RunBudget(settings.budget), TraceBus("t"))
