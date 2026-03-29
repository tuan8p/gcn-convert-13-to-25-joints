"""Integration-style tests: full prepare → model → restore pipeline — requires torch."""
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

from ssr_gcn.constants import TOYOTA_TO_NTU_25_MAP
from ssr_gcn.data import (
    prepare_inference_input,
    prepare_sequence_pair,
    restore_prediction,
)
from ssr_gcn.model import SSRGCN


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model() -> SSRGCN:
    model = SSRGCN(hidden_channels=16, num_blocks=2, temporal_kernel=3, dropout=0.0)
    model.eval()
    return model


def _etri_seq(T: int = 80, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, 25, 3)).astype(np.float32)


def _toyota_seq(T: int = 80, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, 13, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Training data preparation (ETRI 25D → input_13 / target_25 pair)
# ---------------------------------------------------------------------------

class TestTrainPreparePipeline:
    def test_model_accepts_prepared_input(self, small_model):
        prepared = prepare_sequence_pair(
            _etri_seq(), fixed_len=30, apply_augmentation=False,
            rotation_deg=0.0, jitter_std=0.0, seed=0,
        )
        inp = torch.from_numpy(prepared["input_13"]).unsqueeze(0)  # (1, 30, 13, 3)
        with torch.no_grad():
            out = small_model(inp)
        assert out.shape == (1, 30, 25, 3)

    def test_input13_target25_visible_joint_consistency(self):
        """Core data integrity check: input_13[t,i] == target_25[t, NTU_IDX[i]]."""
        prepared = prepare_sequence_pair(
            _etri_seq(seed=7), fixed_len=30, apply_augmentation=False,
            rotation_deg=0.0, jitter_std=0.0, seed=0,
        )
        for i, ntu_idx in enumerate(TOYOTA_TO_NTU_25_MAP):
            np.testing.assert_array_almost_equal(
                prepared["input_13"][:, i, :],
                prepared["target_25"][:, ntu_idx, :],
                decimal=5,
                err_msg=f"Toyota joint {i} ↔ NTU joint {ntu_idx} inconsistent",
            )

    def test_jitter_does_not_modify_target(self):
        """
        After jitter augmentation, target_25 should be unchanged (jitter only on input_13).
        This is the fix for the original bug where both arrays received independent noise.
        """
        no_aug = prepare_sequence_pair(
            _etri_seq(seed=3), fixed_len=30, apply_augmentation=False,
            rotation_deg=0.0, jitter_std=0.0, seed=0,
        )
        with_jitter = prepare_sequence_pair(
            _etri_seq(seed=3), fixed_len=30, apply_augmentation=True,
            rotation_deg=0.0, jitter_std=0.05, seed=99,
        )
        np.testing.assert_array_almost_equal(
            no_aug["target_25"],
            with_jitter["target_25"],
            decimal=5,
            err_msg="target_25 must not change when jitter is applied",
        )

    def test_scale_and_root_center_valid(self):
        prepared = prepare_sequence_pair(
            _etri_seq(), fixed_len=30, apply_augmentation=False,
            rotation_deg=0.0, jitter_std=0.0, seed=0,
        )
        assert float(prepared["scale"]) > 0.0
        assert np.isfinite(prepared["root_center"]).all()


# ---------------------------------------------------------------------------
# Inference pipeline (Toyota 13D → model → restore to world coords)
# ---------------------------------------------------------------------------

class TestInferencePipeline:
    def test_full_pipeline_from_13_joints(self, small_model):
        seq_13 = _toyota_seq()
        prepared = prepare_inference_input(seq_13, fixed_len=30)
        inp = torch.from_numpy(prepared["input_13"]).unsqueeze(0)
        with torch.no_grad():
            pred_norm = small_model(inp).squeeze(0).numpy()
        restored = restore_prediction(pred_norm, prepared["root_center"], prepared["scale"])
        assert restored.shape == (30, 25, 3)
        assert not np.isnan(restored).any()
        assert not np.isinf(restored).any()

    def test_full_pipeline_from_25_joints(self, small_model):
        seq_25 = _etri_seq()
        prepared = prepare_inference_input(seq_25, fixed_len=30)
        inp = torch.from_numpy(prepared["input_13"]).unsqueeze(0)
        with torch.no_grad():
            pred_norm = small_model(inp).squeeze(0).numpy()
        restored = restore_prediction(pred_norm, prepared["root_center"], prepared["scale"])
        assert restored.shape == (30, 25, 3)

    def test_restore_undoes_normalization(self):
        seq_13 = _toyota_seq()
        prepared = prepare_inference_input(seq_13, fixed_len=30)
        root = prepared["root_center"]
        scale = float(prepared["scale"])
        # Use normalized input as mock prediction to verify math
        norm_pred = prepared["input_13"]  # (30, 13, 3) — subset
        # Build a dummy 25-joint zero pred and set joint 0 manually
        dummy_pred = np.zeros((30, 25, 3), dtype=np.float32)
        dummy_pred[:, 0, :] = norm_pred[:, 0, :]
        restored = restore_prediction(dummy_pred, root, scale)
        # joint 0: norm_pred[:,0,:] * scale + root
        expected_j0 = norm_pred[:, 0, :] * scale + root
        np.testing.assert_array_almost_equal(restored[:, 0, :], expected_j0, decimal=5)

    def test_4d_toyota_input_inferred(self, small_model):
        seq_4d = np.random.default_rng(0).standard_normal((50, 2, 13, 3)).astype(np.float32)
        prepared = prepare_inference_input(seq_4d, fixed_len=30)
        inp = torch.from_numpy(prepared["input_13"]).unsqueeze(0)
        with torch.no_grad():
            pred_norm = small_model(inp).squeeze(0).numpy()
        restored = restore_prediction(pred_norm, prepared["root_center"], prepared["scale"])
        assert restored.shape == (30, 25, 3)

    def test_batch_inference_consistent(self, small_model):
        """Same sequence inferred individually and in batch should give same result."""
        seq_a = _toyota_seq(seed=1)
        seq_b = _toyota_seq(seed=2)
        prep_a = prepare_inference_input(seq_a, fixed_len=20)
        prep_b = prepare_inference_input(seq_b, fixed_len=20)
        inp_a = torch.from_numpy(prep_a["input_13"])  # (20, 13, 3)
        inp_b = torch.from_numpy(prep_b["input_13"])

        with torch.no_grad():
            out_a_single = small_model(inp_a.unsqueeze(0)).squeeze(0)
            out_b_single = small_model(inp_b.unsqueeze(0)).squeeze(0)
            batch = torch.stack([inp_a, inp_b])
            out_batch = small_model(batch)

        assert torch.allclose(out_a_single, out_batch[0], atol=1e-5)
        assert torch.allclose(out_b_single, out_batch[1], atol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end: simulate one training step (forward + backward)
# ---------------------------------------------------------------------------

class TestTrainingStep:
    def test_forward_backward_no_nan(self):
        from ssr_gcn.metrics import total_loss as compute_loss

        model = SSRGCN(hidden_channels=16, num_blocks=2, temporal_kernel=3, dropout=0.0)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Simulate one batch
        inp = torch.randn(2, 10, 13, 3)
        tgt = torch.randn(2, 10, 25, 3)

        optimizer.zero_grad()
        pred = model(inp)
        loss, parts = compute_loss(pred, tgt, joint_weight=1.0, bone_weight=0.1)
        loss.backward()

        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"

        optimizer.step()

        assert loss.item() > 0.0
        assert torch.isfinite(pred).all()
