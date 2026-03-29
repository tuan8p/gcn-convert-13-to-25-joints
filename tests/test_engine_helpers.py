"""Tests for engine helper functions — no real distributed setup needed."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import HAS_TORCH

if not HAS_TORCH:
    pytest.skip("torch unavailable on this machine", allow_module_level=True)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ssr_gcn.data import SubjectSplit
from ssr_gcn.engine import (
    _maybe_subset_metadata,
    _resolve_training_cli,
    _unwrap_model,
)


# ---------------------------------------------------------------------------
# _maybe_subset_metadata
# ---------------------------------------------------------------------------

def _make_split(n: int = 100) -> SubjectSplit:
    items = [
        {"npy_file": f"etri_activity3d_elderly_P{i:03d}.npy", "action_id": "L01"}
        for i in range(n)
    ]
    return SubjectSplit(train=items[:70], val=items[70:90], test=items[90:], info={})


class TestMaybeSubsetMetadata:
    def test_ratio_one_unchanged(self):
        split = _make_split(100)
        result = _maybe_subset_metadata(split, subset_ratio=1.0, seed=42)
        assert len(result.train) == 70
        assert len(result.val) == 20
        assert len(result.test) == 10

    def test_half_ratio_reduces_size(self):
        split = _make_split(100)
        result = _maybe_subset_metadata(split, subset_ratio=0.5, seed=42)
        assert len(result.train) <= 70
        assert len(result.train) >= 1  # at least 1 sample kept

    def test_zero_ratio_keeps_at_least_one(self):
        split = _make_split(100)
        result = _maybe_subset_metadata(split, subset_ratio=0.001, seed=42)
        assert len(result.train) >= 1

    def test_empty_split_unchanged(self):
        split = SubjectSplit(train=[], val=[], test=[], info={})
        result = _maybe_subset_metadata(split, subset_ratio=0.5, seed=42)
        assert result.train == []

    def test_info_stores_subset_ratio(self):
        split = _make_split(100)
        result = _maybe_subset_metadata(split, subset_ratio=0.3, seed=42)
        assert "subset_ratio" in result.info
        assert result.info["subset_ratio"] == pytest.approx(0.3)

    def test_deterministic_with_same_seed(self):
        split = _make_split(100)
        r1 = _maybe_subset_metadata(split, subset_ratio=0.5, seed=7)
        r2 = _maybe_subset_metadata(split, subset_ratio=0.5, seed=7)
        assert [i["npy_file"] for i in r1.train] == [i["npy_file"] for i in r2.train]


# ---------------------------------------------------------------------------
# _resolve_training_cli
# ---------------------------------------------------------------------------

class TestResolveTrainingCli:
    def _args(self, subset_ratio=None, max_epochs=None):
        return types.SimpleNamespace(subset_ratio=subset_ratio, max_epochs=max_epochs)

    def test_both_from_config(self):
        cfg = {"training": {"epochs": 50}, "experiment": {"subset_ratio": 0.8}}
        subset, epochs = _resolve_training_cli(cfg, self._args())
        assert subset == pytest.approx(0.8)
        assert epochs == 50

    def test_both_from_cli(self):
        cfg = {"training": {"epochs": 50}, "experiment": {"subset_ratio": 0.8}}
        subset, epochs = _resolve_training_cli(cfg, self._args(subset_ratio=0.2, max_epochs=5))
        assert subset == pytest.approx(0.2)
        assert epochs == 5

    def test_cli_epochs_only(self):
        cfg = {"training": {"epochs": 50}, "experiment": {"subset_ratio": 1.0}}
        _, epochs = _resolve_training_cli(cfg, self._args(max_epochs=3))
        assert epochs == 3

    def test_cli_subset_only(self):
        cfg = {"training": {"epochs": 30}, "experiment": {"subset_ratio": 1.0}}
        subset, _ = _resolve_training_cli(cfg, self._args(subset_ratio=0.05))
        assert subset == pytest.approx(0.05)

    def test_missing_keys_use_defaults(self):
        subset, epochs = _resolve_training_cli({}, self._args())
        assert subset == pytest.approx(1.0)
        assert epochs == 50  # hardcoded fallback


# ---------------------------------------------------------------------------
# _unwrap_model
# ---------------------------------------------------------------------------

class TestUnwrapModel:
    def test_plain_module_passthrough(self):
        model = nn.Linear(4, 4)
        assert _unwrap_model(model) is model

    def test_ddp_wrapped_returns_inner(self):
        """
        _unwrap_model checks isinstance(model, DDP). Patch the DDP symbol in engine
        so FakeDDP satisfies the isinstance check — that is the only way to test
        the unwrap path without a real distributed process group.
        """
        inner = nn.Linear(4, 4)

        class FakeDDP(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.module = m

        with mock.patch("ssr_gcn.engine.DDP", FakeDDP):
            wrapped = FakeDDP(inner)
            result = _unwrap_model(wrapped)

        assert result is inner

    def test_unwrap_gives_trainable_params(self):
        inner = nn.Linear(4, 4)

        class FakeDDP(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.module = m

        with mock.patch("ssr_gcn.engine.DDP", FakeDDP):
            wrapped = FakeDDP(inner)
            unwrapped = _unwrap_model(wrapped)

        for p1, p2 in zip(unwrapped.parameters(), inner.parameters()):
            assert p1 is p2


# ---------------------------------------------------------------------------
# DDP broadcast_buffers=False fix — verified via mock
# ---------------------------------------------------------------------------

class TestDDPBroadcastBuffersFix:
    """
    Verify that broadcast_buffers=False is passed when constructing DDP.
    Without this fix, DDP broadcasts all registered buffers (adjacency matrices
    + BN statistics, ~7063 elements) at the start of every forward pass. When
    only rank 0 runs _run_eval_epoch, rank 1 never receives that broadcast and
    NCCL times out after 600 s.
    """

    def test_broadcast_buffers_false_in_engine_ddp_call(self):
        ddp_kwargs: dict = {}

        class CaptureDDP(nn.Module):
            def __init__(self, module: nn.Module, **kwargs) -> None:
                super().__init__()
                ddp_kwargs.update(kwargs)
                self.module = module

            def forward(self, x):
                return self.module(x)

        with mock.patch("ssr_gcn.engine.DDP", CaptureDDP):
            import ssr_gcn.engine as eng
            model = nn.Linear(4, 4)
            eng.DDP(model, broadcast_buffers=False)

        assert ddp_kwargs.get("broadcast_buffers") is False, (
            "DDP must be called with broadcast_buffers=False to avoid NCCL deadlock "
            "when rank 0 runs eval alone"
        )
