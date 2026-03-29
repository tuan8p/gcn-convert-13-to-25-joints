"""Tests for ssr_gcn.data preprocessing — no torch required."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from ssr_gcn.constants import TOYOTA_TO_NTU_25_MAP
from ssr_gcn.data import (
    JOINT_NAMES_TOYOTA_13_INDEX,
    SubjectSplitter,
    apply_sequence_transform,
    augment_pair,
    build_subject_id,
    compute_root_and_scale,
    extract_toyota_13,
    prepare_inference_input,
    prepare_sequence_pair,
    restore_prediction,
    resample_sequence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq25(T: int = 20, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, 25, 3)).astype(np.float32)


def _seq13(T: int = 20, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, 13, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# resample_sequence
# ---------------------------------------------------------------------------

class TestResampleSequence:
    def test_same_length_unchanged(self):
        seq = _seq25(20)
        out = resample_sequence(seq, 20)
        np.testing.assert_array_almost_equal(out, seq)

    def test_upsample_shape(self):
        out = resample_sequence(_seq25(10), 50)
        assert out.shape == (50, 25, 3)
        assert out.dtype == np.float32

    def test_downsample_shape(self):
        out = resample_sequence(_seq25(100), 30)
        assert out.shape == (30, 25, 3)

    def test_single_frame_repeats(self):
        seq = _seq25(1)
        out = resample_sequence(seq, 10)
        assert out.shape == (10, 25, 3)
        for i in range(10):
            np.testing.assert_array_equal(out[i], out[0])

    def test_output_dtype_float32(self):
        seq = _seq25(20).astype(np.float64)
        out = resample_sequence(seq, 15)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# extract_toyota_13
# ---------------------------------------------------------------------------

class TestExtractToyota13:
    def test_shape(self):
        seq25 = _seq25(20)
        out = extract_toyota_13(seq25)
        assert out.shape == (20, 13, 3)
        assert out.dtype == np.float32

    def test_correct_joint_mapping(self):
        seq25 = _seq25(20)
        out = extract_toyota_13(seq25)
        for toyota_idx, ntu_idx in enumerate(TOYOTA_TO_NTU_25_MAP):
            np.testing.assert_array_equal(
                out[:, toyota_idx, :],
                seq25[:, ntu_idx, :],
                err_msg=f"Toyota idx {toyota_idx} should map to NTU idx {ntu_idx}",
            )


# ---------------------------------------------------------------------------
# compute_root_and_scale
# ---------------------------------------------------------------------------

class TestComputeRootAndScale:
    def _make_seq_with_known_hips(self, T: int = 10):
        seq = _seq13(T)
        seq[:, JOINT_NAMES_TOYOTA_13_INDEX["Rhip"], :] = np.array([2.0, 0.0, 0.0])
        seq[:, JOINT_NAMES_TOYOTA_13_INDEX["Lhip"], :] = np.array([0.0, 0.0, 0.0])
        seq[:, JOINT_NAMES_TOYOTA_13_INDEX["Rsho"], :] = np.array([2.0, 1.0, 0.0])
        seq[:, JOINT_NAMES_TOYOTA_13_INDEX["Lsho"], :] = np.array([0.0, 1.0, 0.0])
        return seq

    def test_root_center_is_mean_of_hips(self):
        seq = self._make_seq_with_known_hips()
        root_center, _ = compute_root_and_scale(seq)
        np.testing.assert_array_almost_equal(root_center[0], [1.0, 0.0, 0.0])

    def test_scale_positive(self):
        seq = self._make_seq_with_known_hips()
        _, scale = compute_root_and_scale(seq)
        assert scale > 0.0

    def test_scale_equals_torso_length(self):
        # shoulder center = (2+0)/2, 1, 0 = (1, 1, 0)
        # hip center = (1, 0, 0)
        # torso = sqrt((1-1)^2 + (1-0)^2 + 0) = 1.0
        seq = self._make_seq_with_known_hips()
        _, scale = compute_root_and_scale(seq)
        assert scale == pytest.approx(1.0, abs=1e-5)

    def test_output_shapes(self):
        seq = self._make_seq_with_known_hips(15)
        root_center, scale = compute_root_and_scale(seq)
        assert root_center.shape == (15, 3)
        assert isinstance(scale, float)


# ---------------------------------------------------------------------------
# prepare_sequence_pair
# ---------------------------------------------------------------------------

class TestPrepareSequencePair:
    def test_output_shapes(self):
        result = prepare_sequence_pair(
            _seq25(50), fixed_len=30, apply_augmentation=False,
            rotation_deg=0.0, jitter_std=0.0, seed=42,
        )
        assert result["input_13"].shape == (30, 13, 3)
        assert result["target_25"].shape == (30, 25, 3)
        assert result["root_center"].shape == (30, 3)
        assert result["observed_mask"].shape == (25,)

    def test_observed_mask_has_13_ones(self):
        result = prepare_sequence_pair(
            _seq25(50), fixed_len=30, apply_augmentation=False,
            rotation_deg=0.0, jitter_std=0.0, seed=0,
        )
        mask = result["observed_mask"]
        assert int(mask.sum()) == 13
        for idx in TOYOTA_TO_NTU_25_MAP:
            assert mask[idx] == 1.0

    def test_visible_joints_consistent_between_input_and_target(self):
        """input_13[t, i] should equal target_25[t, TOYOTA_TO_NTU_25_MAP[i]] (same joint, same normalization)."""
        result = prepare_sequence_pair(
            _seq25(50), fixed_len=30, apply_augmentation=False,
            rotation_deg=0.0, jitter_std=0.0, seed=7,
        )
        for i, ntu_idx in enumerate(TOYOTA_TO_NTU_25_MAP):
            np.testing.assert_array_almost_equal(
                result["input_13"][:, i, :],
                result["target_25"][:, ntu_idx, :],
                decimal=5,
                err_msg=f"Toyota joint {i} vs NTU joint {ntu_idx} mismatch",
            )

    def test_handles_4d_input(self):
        seq_4d = np.random.default_rng(0).standard_normal((50, 2, 25, 3)).astype(np.float32)
        result = prepare_sequence_pair(
            seq_4d, fixed_len=30, apply_augmentation=False,
            rotation_deg=0.0, jitter_std=0.0, seed=0,
        )
        assert result["input_13"].shape == (30, 13, 3)

    def test_rejects_wrong_shape(self):
        bad = np.random.randn(50, 17, 3).astype(np.float32)
        with pytest.raises(ValueError):
            prepare_sequence_pair(bad, fixed_len=30, apply_augmentation=False,
                                  rotation_deg=0.0, jitter_std=0.0, seed=0)

    def test_augmentation_changes_values(self):
        seq = _seq25(50)
        no_aug = prepare_sequence_pair(seq, 30, False, 0.0, 0.0, 0)
        with_aug = prepare_sequence_pair(seq, 30, True, 30.0, 0.0, 0)
        assert not np.allclose(no_aug["input_13"], with_aug["input_13"])

    def test_scale_greater_than_zero(self):
        result = prepare_sequence_pair(_seq25(50), 30, False, 0.0, 0.0, 0)
        assert float(result["scale"]) > 0.0

    def test_jitter_only_modifies_input_not_target(self):
        """
        Jitter simulates sensor noise → applied only to input_13, target_25 stays clean.
        Correct SSR behavior: model learns noisy-input → clean-target mapping.
        Bug that was fixed: previously both arrays received independent noise, making
        the visible joint positions inconsistent between input and target supervision.
        """
        seq = _seq25(50, seed=1)
        no_aug = prepare_sequence_pair(seq, 30, False, 0.0, 0.0, seed=0)
        with_jitter = prepare_sequence_pair(seq, 30, True, 0.0, 0.05, seed=99)

        # target_25 must be identical regardless of jitter (clean ground truth)
        np.testing.assert_array_almost_equal(
            no_aug["target_25"],
            with_jitter["target_25"],
            decimal=5,
            err_msg="target_25 must not change when jitter is applied to input",
        )

        # input_13 must differ (jitter was actually applied)
        assert not np.allclose(no_aug["input_13"], with_jitter["input_13"]), (
            "input_13 should differ after jitter augmentation"
        )


# ---------------------------------------------------------------------------
# prepare_inference_input
# ---------------------------------------------------------------------------

class TestPrepareInferenceInput:
    def test_from_13_joints(self):
        result = prepare_inference_input(_seq13(40), fixed_len=30)
        assert result["input_13"].shape == (30, 13, 3)
        assert result["root_center"].shape == (30, 3)

    def test_from_25_joints(self):
        result = prepare_inference_input(_seq25(40), fixed_len=30)
        assert result["input_13"].shape == (30, 13, 3)

    def test_from_4d_input(self):
        seq_4d = np.random.randn(40, 2, 25, 3).astype(np.float32)
        result = prepare_inference_input(seq_4d, fixed_len=30)
        assert result["input_13"].shape == (30, 13, 3)

    def test_rejects_wrong_joint_count(self):
        bad = np.random.randn(40, 17, 3).astype(np.float32)
        with pytest.raises(ValueError):
            prepare_inference_input(bad, fixed_len=30)


# ---------------------------------------------------------------------------
# restore_prediction
# ---------------------------------------------------------------------------

class TestRestorePrediction:
    def test_zero_prediction_returns_root(self):
        T, J = 30, 25
        pred_norm = np.zeros((T, J, 3), dtype=np.float32)
        root = np.full((T, 3), 2.0, dtype=np.float32)
        restored = restore_prediction(pred_norm, root, scale=3.0)
        # 0 * 3 + 2 = 2 everywhere
        np.testing.assert_array_almost_equal(restored, np.full((T, J, 3), 2.0))

    def test_inverse_of_normalization(self):
        T, J = 20, 25
        original = np.random.randn(T, J, 3).astype(np.float32)
        root = np.ones((T, 3), dtype=np.float32) * 0.5
        scale = 2.0
        normalized = (original - root[:, None, :]) / scale
        restored = restore_prediction(normalized, root, scale)
        np.testing.assert_array_almost_equal(restored, original, decimal=5)

    def test_accepts_numpy_scale(self):
        pred_norm = np.zeros((10, 25, 3), dtype=np.float32)
        root = np.zeros((10, 3), dtype=np.float32)
        scale = np.array(1.5, dtype=np.float32)
        restored = restore_prediction(pred_norm, root, scale)
        assert restored.shape == (10, 25, 3)


# ---------------------------------------------------------------------------
# augment_pair
# ---------------------------------------------------------------------------

class TestAugmentPair:
    def test_no_rotation_no_jitter_unchanged(self):
        rng = np.random.default_rng(0)
        inp = np.ones((10, 13, 3), dtype=np.float32)
        tgt = np.ones((10, 25, 3), dtype=np.float32)
        aug_inp, aug_tgt = augment_pair(inp, tgt, rotation_deg=0.0, jitter_std=0.0, rng=rng)
        np.testing.assert_array_equal(aug_inp, inp)
        np.testing.assert_array_equal(aug_tgt, tgt)

    def test_rotation_changes_values(self):
        rng = np.random.default_rng(1)
        inp = np.random.randn(10, 13, 3).astype(np.float32)
        tgt = np.random.randn(10, 25, 3).astype(np.float32)
        aug_inp, aug_tgt = augment_pair(inp, tgt, rotation_deg=45.0, jitter_std=0.0, rng=rng)
        assert not np.allclose(aug_inp, inp) or not np.allclose(aug_tgt, tgt)

    def test_original_arrays_not_modified(self):
        rng = np.random.default_rng(2)
        inp = np.random.randn(10, 13, 3).astype(np.float32)
        inp_copy = inp.copy()
        tgt = np.random.randn(10, 25, 3).astype(np.float32)
        tgt_copy = tgt.copy()
        augment_pair(inp, tgt, rotation_deg=30.0, jitter_std=0.01, rng=rng)
        np.testing.assert_array_equal(inp, inp_copy)
        np.testing.assert_array_equal(tgt, tgt_copy)


# ---------------------------------------------------------------------------
# SubjectSplitter
# ---------------------------------------------------------------------------

def _make_meta(n_subjects: int = 10, n_actions: int = 3, n_per_subject: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = [f"L{i + 1:02d}" for i in range(n_actions)]
    meta = []
    for person in range(1, n_subjects + 1):
        for _ in range(n_per_subject):
            meta.append({
                "npy_file": f"etri_activity3d_elderly_P{person:03d}A001.npy",
                "action_id": str(rng.choice(labels)),
                "person": person,
                "n_frames": 100,
            })
    return meta


class TestSubjectSplitter:
    def test_split_sizes_sum_to_total(self):
        meta = _make_meta(10)
        split = SubjectSplitter(0.7, 0.2, 0.1, seed=42).split(meta)
        total = len(split.train) + len(split.val) + len(split.test)
        assert total <= len(meta)

    def test_no_person_overlap_across_splits(self):
        meta = _make_meta(12, n_actions=4)
        split = SubjectSplitter(0.7, 0.2, 0.1, seed=42).split(meta)

        def persons(items):
            return {item["person"] for item in items}

        assert persons(split.train) & persons(split.val) == set()
        assert persons(split.train) & persons(split.test) == set()
        assert persons(split.val) & persons(split.test) == set()

    def test_val_labels_subset_of_train_labels(self):
        meta = _make_meta(12, n_actions=4)
        split = SubjectSplitter(0.7, 0.2, 0.1, seed=42).split(meta)
        train_labels = {item["action_id"] for item in split.train}
        val_labels = {item["action_id"] for item in split.val}
        assert val_labels <= train_labels

    def test_empty_input(self):
        split = SubjectSplitter().split([])
        assert split.train == []
        assert split.val == []
        assert split.test == []

    def test_info_dict_present(self):
        split = SubjectSplitter().split(_make_meta(6))
        assert "n_train" in split.info
        assert "n_val" in split.info

    def test_deterministic_with_same_seed(self):
        meta = _make_meta(10, n_actions=3)
        s1 = SubjectSplitter(seed=7).split(meta)
        s2 = SubjectSplitter(seed=7).split(meta)
        assert [i["npy_file"] for i in s1.train] == [i["npy_file"] for i in s2.train]


# ---------------------------------------------------------------------------
# JOINT_NAMES_TOYOTA_13_INDEX & build_subject_id
# ---------------------------------------------------------------------------

class TestJointNamesIndex:
    EXPECTED_KEYS = {"Rank", "Lank", "Rkne", "Lkne", "Rhip", "Lhip",
                     "Rwri", "Lwri", "Relb", "Lelb", "Rsho", "Lsho", "Head"}

    def test_has_all_keys(self):
        assert set(JOINT_NAMES_TOYOTA_13_INDEX.keys()) == self.EXPECTED_KEYS

    def test_values_in_range(self):
        for name, idx in JOINT_NAMES_TOYOTA_13_INDEX.items():
            assert 0 <= idx < 13, f"Index for {name} = {idx} out of range"

    def test_values_unique(self):
        values = list(JOINT_NAMES_TOYOTA_13_INDEX.values())
        assert len(set(values)) == len(values)


class TestBuildSubjectId:
    def test_etri_elderly(self):
        item = {"npy_file": "etri_activity3d_elderly_P005.npy", "person": 5}
        assert build_subject_id(item) == "etri_elderly_005"

    def test_toyota(self):
        item = {"npy_file": "toyota_smarthome_P003.npy", "person": 3}
        assert build_subject_id(item) == "toyota_003"

    def test_no_person_returns_none(self):
        item = {"npy_file": "etri_activity3d_elderly_P001.npy"}
        assert build_subject_id(item) is None
