"""Settings: config/default.toml, overridden by config/local.toml if present."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Thresholds:
    calibrated: bool
    yellow_below: float
    red_below: float
    ocr_agree_min_ratio: float
    vendor_match_min_ratio: float

    @classmethod
    def load(cls, path: Path) -> Thresholds:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


class Settings:
    """Thin wrapper over the merged TOML dict, with paths resolved against the repo root."""

    def __init__(self, data: dict[str, Any]):
        self.data = data

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.data[section]

    def path(self, key: str) -> Path:
        p = Path(self.data["paths"][key])
        return p if p.is_absolute() else REPO_ROOT / p

    def thresholds(self) -> Thresholds:
        return Thresholds.load(self.path("thresholds"))

    @classmethod
    def load(cls, extra: Path | None = None) -> Settings:
        data = tomllib.loads((CONFIG_DIR / "default.toml").read_text(encoding="utf-8"))
        for override in (CONFIG_DIR / "local.toml", extra):
            if override and override.exists():
                data = _merge(data, tomllib.loads(override.read_text(encoding="utf-8")))
        return cls(data)


def apply_gpu_limits(settings: Settings) -> None:
    """Cap Paddle's GPU memory and set CUDA module loading. Must run before paddle is first
    imported; environment variables already set by the user win."""
    gpu = settings.data.get("gpu", {})
    limit = int(gpu.get("paddle_memory_limit_mb", 0))
    if limit > 0:
        os.environ.setdefault("FLAGS_gpu_memory_limit_mb", str(limit))
    if gpu.get("cuda_module_loading"):
        os.environ.setdefault("CUDA_MODULE_LOADING", gpu["cuda_module_loading"])
