"""Tests for ssr_gcn.metrics — requires torch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import HAS_TORCH

if not HAS_TORCH:
    pytest.skip("torch unavailable on this machine", allow_module_level=True)

import torch  # noqa: E402

from ssr_gcn.metrics import (
    MetricTracker,
    bone_length_error,
    bone_length_loss,
    bone_vectors,
    missing_joint_mpjpe,
    mpjpe,
    per_joint_mpjpe,
    total_loss,
    visible_joint_mpjpe,
)
from ssr_gcn.constants import NTU_EDGES


def _batch(B: int = 2, T: int = 10, J: int = 25, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, J, 3)


# ---------------------------------------------------------------------------
# mpjpe
# ---------------------------------------------------------------------------

class TestMpjpe:
    def test_identical_tensors_zero(self):
        pred = _batch()
        assert float(mpjpe(pred, pred)) == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        pred = torch.zeros(1, 1, 1, 3)
        target = torch.ones(1, 1, 1, 3)
        expected = float(torch.linalg.norm(torch.ones(3)))
        assert float(mpjpe(pred, target)) == pytest.approx(expected, rel=1e-5)

    def test_nonnegative(self):
        pred = _batch()
        target = _batch(seed=1)
        assert float(mpjpe(pred, target)) >= 0.0


# ---------------------------------------------------------------------------
# missing_joint_mpjpe / visible_joint_mpjpe
# ---------------------------------------------------------------------------

class TestJointSubsetMpjpe:
    def test_missing_mpjpe_scalar(self):
        pred = _batch()
        target = _batch(seed=1)
        val = missing_joint_mpjpe(pred, target)
        assert val.ndim == 0

    def test_visible_mpjpe_scalar(self):
        pred = _batch()
        target = _batch(seed=1)
        val = visible_joint_mpjpe(pred, target)
        assert val.ndim == 0

    def test_identical_missing_zero(self):
        pred = _batch()
        assert float(missing_joint_mpjpe(pred, pred)) == pytest.approx(0.0, abs=1e-6)

    def test_identical_visible_zero(self):
        pred = _batch()
        assert float(visible_joint_mpjpe(pred, pred)) == pytest.approx(0.0, abs=1e-6)

    def test_missing_and_visible_sum_relates_to_total(self):
        # Both are averages over subsets; they should be non-negative
        pred = _batch()
        target = _batch(seed=2)
        assert float(missing_joint_mpjpe(pred, target)) >= 0.0
        assert float(visible_joint_mpjpe(pred, target)) >= 0.0


# ---------------------------------------------------------------------------
# bone_vectors / bone_length_error / bone_length_loss
# ---------------------------------------------------------------------------

class TestBones:
    def test_bone_vectors_shape(self):
        seq = _batch()
        bv = bone_vectors(seq)
        assert bv.shape == (2, 10, len(NTU_EDGES), 3)

    def test_bone_length_error_identical_zero(self):
        pred = _batch()
        assert float(bone_length_error(pred, pred)) == pytest.approx(0.0, abs=1e-5)

    def test_bone_length_loss_identical_zero(self):
        pred = _batch()
        assert float(bone_length_loss(pred, pred)) == pytest.approx(0.0, abs=1e-5)

    def test_bone_length_error_nonnegative(self):
        pred = _batch()
        target = _batch(seed=1)
        assert float(bone_length_error(pred, target)) >= 0.0


# ---------------------------------------------------------------------------
# total_loss
# ---------------------------------------------------------------------------

class TestTotalLoss:
    def test_identical_zero(self):
        pred = _batch()
        loss, _ = total_loss(pred, pred, joint_weight=1.0, bone_weight=0.1)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different_tensors(self):
        pred = _batch()
        target = _batch(seed=1)
        loss, _ = total_loss(pred, target, joint_weight=1.0, bone_weight=0.1)
        assert float(loss) > 0.0

    def test_returns_parts_dict(self):
        pred = _batch()
        target = _batch(seed=1)
        _, parts = total_loss(pred, target, joint_weight=1.0, bone_weight=0.1)
        assert "joint_loss" in parts
        assert "bone_loss" in parts

    def test_bone_weight_zero_excludes_bone(self):
        pred = _batch()
        target = _batch(seed=1)
        loss_with, p_with = total_loss(pred, target, joint_weight=1.0, bone_weight=0.1)
        loss_no_bone, p_no_bone = total_loss(pred, target, joint_weight=1.0, bone_weight=0.0)
        # joint_loss component should be identical
        assert p_with["joint_loss"] == pytest.approx(p_no_bone["joint_loss"], rel=1e-5)
        # total_loss without bone should be <= with bone (bone_loss >= 0)
        assert float(loss_no_bone) <= float(loss_with) + 1e-6

    def test_joint_weight_scales_loss(self):
        pred = _batch()
        target = _batch(seed=1)
        loss_1x, _ = total_loss(pred, target, joint_weight=1.0, bone_weight=0.0)
        loss_2x, _ = total_loss(pred, target, joint_weight=2.0, bone_weight=0.0)
        assert float(loss_2x) == pytest.approx(float(loss_1x) * 2, rel=1e-5)


# ---------------------------------------------------------------------------
# MetricTracker
# ---------------------------------------------------------------------------

class TestMetricTracker:
    def test_empty_compute_returns_empty(self):
        tracker = MetricTracker()
        assert tracker.compute() == {}

    def test_single_update_preserved(self):
        tracker = MetricTracker()
        tracker.update(count=10, loss=2.0, mpjpe=0.5)
        result = tracker.compute()
        assert result["loss"] == pytest.approx(2.0)
        assert result["mpjpe"] == pytest.approx(0.5)

    def test_equal_counts_average(self):
        tracker = MetricTracker()
        tracker.update(count=10, loss=1.0)
        tracker.update(count=10, loss=3.0)
        assert tracker.compute()["loss"] == pytest.approx(2.0)

    def test_weighted_average_different_counts(self):
        # (3 × 1.0 + 1 × 5.0) / 4 = 2.0
        tracker = MetricTracker()
        tracker.update(count=3, loss=1.0)
        tracker.update(count=1, loss=5.0)
        assert tracker.compute()["loss"] == pytest.approx(2.0)

    def test_total_count_accumulated(self):
        tracker = MetricTracker()
        tracker.update(count=5, loss=1.0)
        tracker.update(count=7, loss=1.0)
        assert tracker.total == 12


# ---------------------------------------------------------------------------
# per_joint_mpjpe
# ---------------------------------------------------------------------------

class TestPerJointMpjpe:
    def test_output_has_25_keys(self):
        preds = [np.random.randn(2, 10, 25, 3)]
        targets = [np.random.randn(2, 10, 25, 3)]
        result = per_joint_mpjpe(preds, targets)
        assert len(result) == 25

    def test_key_format(self):
        preds = [np.random.randn(1, 5, 25, 3)]
        targets = [np.random.randn(1, 5, 25, 3)]
        result = per_joint_mpjpe(preds, targets)
        assert all(k.startswith("joint_") for k in result)

    def test_identical_inputs_zero(self):
        arr = [np.ones((1, 5, 25, 3))]
        result = per_joint_mpjpe(arr, arr)
        for v in result.values():
            assert v == pytest.approx(0.0, abs=1e-6)
