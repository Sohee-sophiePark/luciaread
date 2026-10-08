"""Settings loader. `config/settings.yaml` is the single source of truth; env overrides run_mode."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"

RunMode = Literal["live", "record", "replay"]
Role = Literal["router", "analyst", "writer", "evaluator"]


class RateLimitsCfg(BaseModel):
    """`max_concurrency` plus one `{rpm: N}` entry per model id, stored as extra keys."""

    model_config = ConfigDict(extra="allow")
    max_concurrency: int = 2

    def rpm_for(self, model_id: str) -> int:
        entry = (self.model_extra or {}).get(model_id)
        if not isinstance(entry, dict) or "rpm" not in entry:
            raise KeyError(f"rate_limits has no rpm entry for model {model_id!r}")
        return int(entry["rpm"])


class RetryCfg(BaseModel):
    max_attempts: int
    base_seconds: float
    cap_seconds: float
    max_wait_seconds: float = 120.0  # a longer server-requested wait fails fast


class BudgetCfg(BaseModel):
    max_llm_calls: int
    max_total_tokens: int
    max_wall_seconds: int
    per_call_timeout_seconds: int


class LoopCfg(BaseModel):
    max_revisions: int
    pass_min_score: int
    pass_mean_score: float


class QualityCfg(BaseModel):
    min_contrast_std: float
    brightness_range: tuple[float, float]
    min_sharpness: float


class RulesCfg(BaseModel):
    """Deterministic thresholds for gates and review flags."""

    low_confidence: float
    max_upload_mb: float
    max_pixels: int
    min_side_px: int
    colour_max_saturation: float
    quality: dict[str, QualityCfg]  # per modality, from training-split percentiles
    attention_inside_min: float


class PricingCfg(BaseModel):
    """`free_tier` plus one `{input, output}` USD-per-million entry per model id (extra keys)."""

    model_config = ConfigDict(extra="allow")
    free_tier: bool = True

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        p = (self.model_extra or {}).get(model)
        if self.free_tier or not isinstance(p, dict):
            return 0.0
        return (tokens_in * p["input"] + tokens_out * p["output"]) / 1_000_000


class InjectionCfg(BaseModel):
    redact: bool


class TraceCfg(BaseModel):
    store_prompts: bool


class PathsCfg(BaseModel):
    weights: str
    prompts: str
    runs: str
    cassettes: str
    replays: str
    samples: str


class ApiCfg(BaseModel):
    port_min: int
    port_max: int


class Settings(BaseModel):
    run_mode: RunMode
    models: dict[Role, str]
    fallback_models: list[str] = []  # live only: tried in order on quota or overload
    temperature: dict[Role, float]
    max_output_tokens: dict[Role, int]
    thinking_level: dict[Role, str | int]  # Gemini 3: MINIMAL|LOW|MEDIUM|HIGH; 2.5: token budget
    rate_limits: RateLimitsCfg
    retry: RetryCfg
    budget: BudgetCfg
    loop: LoopCfg
    rules: RulesCfg
    pricing: PricingCfg
    injection: InjectionCfg
    trace: TraceCfg
    paths: PathsCfg
    api: ApiCfg

    def path(self, name: str) -> Path:
        """Absolute path for `paths.<name>`, resolved against the repo root."""
        return ROOT / getattr(self.paths, name)


def load_settings(path: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Parse the YAML file. A non-empty `RUN_MODE` in `env` (os.environ) wins."""
    environ = os.environ if env is None else env
    raw = yaml.safe_load((path or SETTINGS_FILE).read_text(encoding="utf-8"))
    if value := environ.get("RUN_MODE", "").strip():
        raw["run_mode"] = value
    return Settings.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton; the only cached global state in the package."""
    return load_settings()


def gemini_api_key() -> str:
    """Read `GEMINI_API_KEY` from the environment at call time. Never stored or logged."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set; live and record modes need it")
    return key
