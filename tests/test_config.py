"""Tests for ssr_gcn.config — no torch required."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssr_gcn.config import _deep_update, load_cfg, resolve_path
from ssr_gcn.constants import DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_none_returns_none(self):
        assert resolve_path(None) is None

    def test_empty_string_returns_none(self):
        assert resolve_path("") is None

    def test_absolute_path_returned_as_is(self):
        p = Path(__file__).resolve()
        result = resolve_path(str(p))
        assert result == p

    def test_relative_resolved_against_base(self):
        base = Path(__file__).resolve().parent
        name = Path(__file__).name
        result = resolve_path(name, base=base)
        assert result == Path(__file__).resolve()


# ---------------------------------------------------------------------------
# _deep_update
# ---------------------------------------------------------------------------

class TestDeepUpdate:
    def test_flat_override(self):
        target = {"a": 1, "b": 2}
        _deep_update(target, {"b": 99, "c": 3})
        assert target == {"a": 1, "b": 99, "c": 3}

    def test_nested_preserves_unmentioned_keys(self):
        target = {"training": {"lr": 0.001, "epochs": 50}}
        _deep_update(target, {"training": {"lr": 0.01}})
        assert target["training"]["lr"] == pytest.approx(0.01)
        assert target["training"]["epochs"] == 50  # untouched

    def test_adds_new_nested_key(self):
        target = {"training": {"lr": 0.001}}
        _deep_update(target, {"training": {"weight_decay": 1e-4}})
        assert "weight_decay" in target["training"]

    def test_non_dict_src_replaces_target(self):
        target = {"training": {"lr": 0.001}}
        _deep_update(target, {"training": 42})
        assert target["training"] == 42


# ---------------------------------------------------------------------------
# load_cfg
# ---------------------------------------------------------------------------

def _write_tmp_cfg(cfg: dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
        return f.name


class TestLoadCfg:
    def test_loads_basic_keys(self):
        tmp = _write_tmp_cfg({"training": {"epochs": 5}, "model": {"hidden_channels": 16}})
        cfg = load_cfg(tmp)
        assert cfg["training"]["epochs"] == 5
        assert cfg["model"]["hidden_channels"] == 16
        Path(tmp).unlink()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_cfg("/nonexistent/path/to/config.yaml")

    def test_runtime_profile_applied_from_yaml(self):
        raw = {
            "training": {"epochs": 50, "lr": 0.001},
            "runtime": {"profile": "fast"},
            "runtime_profile": {
                "fast": {"training": {"epochs": 2, "lr": 0.01}},
            },
        }
        tmp = _write_tmp_cfg(raw)
        cfg = load_cfg(tmp)
        assert cfg["training"]["epochs"] == 2
        assert cfg["training"]["lr"] == pytest.approx(0.01)
        assert cfg["training"].get("epochs") == 2  # override applied
        Path(tmp).unlink()

    def test_runtime_profile_cli_override(self):
        raw = {
            "training": {"epochs": 50},
            "runtime_profile": {
                "debug": {"training": {"epochs": 1}},
            },
        }
        tmp = _write_tmp_cfg(raw)
        cfg = load_cfg(tmp, runtime_profile="debug")
        assert cfg["training"]["epochs"] == 1
        Path(tmp).unlink()

    def test_profile_preserves_unmentioned_keys(self):
        raw = {
            "training": {"epochs": 50, "lr": 0.001},
            "runtime_profile": {
                "quick": {"training": {"epochs": 2}},
            },
        }
        tmp = _write_tmp_cfg(raw)
        cfg = load_cfg(tmp, runtime_profile="quick")
        assert cfg["training"]["epochs"] == 2
        assert cfg["training"]["lr"] == pytest.approx(0.001)  # preserved
        Path(tmp).unlink()

    def test_meta_config_path_set(self):
        tmp = _write_tmp_cfg({"training": {}})
        cfg = load_cfg(tmp)
        assert "_meta" in cfg
        assert "config_path" in cfg["_meta"]
        Path(tmp).unlink()

    def test_default_config_loads_if_exists(self):
        if not DEFAULT_CONFIG_PATH.exists():
            pytest.skip("Default config not found on disk")
        cfg = load_cfg()
        assert "training" in cfg
        assert "model" in cfg

    def test_default_config_has_expected_training_keys(self):
        if not DEFAULT_CONFIG_PATH.exists():
            pytest.skip("Default config not found on disk")
        cfg = load_cfg()
        training = cfg.get("training", {})
        for key in ("epochs", "lr", "per_gpu_batch_size"):
            assert key in training, f"Expected key '{key}' in training config"
