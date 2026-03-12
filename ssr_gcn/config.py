"""Config loading helpers for SSR GCN."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ssr_gcn.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT


def resolve_path(path_like: str | None, base: Path | None = None) -> Path | None:
    """Resolve a path relative to project root or the provided base."""
    if not path_like:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    if base is not None:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (PROJECT_ROOT / path).resolve()


def _deep_update(target: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_cfg(
    config_path: str | Path | None = None,
    runtime_profile: str | None = None,
) -> dict[str, Any]:
    """Load YAML config and apply runtime profile overrides."""
    path = resolve_path(str(config_path), base=Path.cwd()) if config_path else DEFAULT_CONFIG_PATH
    if path is None or not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path or DEFAULT_CONFIG_PATH}")

    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}

    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = str(path)

    profile_name = runtime_profile or cfg.get("runtime", {}).get("profile")
    runtime_profiles = cfg.get("runtime_profile") or {}
    if profile_name and profile_name in runtime_profiles:
        profile_cfg = copy.deepcopy(runtime_profiles[profile_name])
        _deep_update(cfg, profile_cfg)
        cfg.setdefault("runtime", {})["profile"] = profile_name

    return cfg
