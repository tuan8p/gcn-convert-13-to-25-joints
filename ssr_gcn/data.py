"""Data pipeline for SSR 13->25 on ETRI elderly."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - allows preprocessing smoke tests without torch
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """Fallback Dataset stub when torch is unavailable."""

        pass

from ssr_gcn.constants import ETRI_ELDERLY_PREFIX, TOYOTA_PREFIX, TOYOTA_TO_NTU_25_MAP


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def resolve_npy_merged_path(cfg: dict[str, Any]) -> Path:
    dataset_cfg = cfg.get("dataset") or {}
    raw_path = dataset_cfg.get("npy_merged_path")
    candidates = [
        Path(raw_path).resolve() if raw_path else None,
        (Path.cwd() / "data" / "processed" / "npy_merged").resolve(),
        (Path.cwd().parent / "data" / "processed" / "npy_merged").resolve(),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    checked = [str(c) for c in candidates if c is not None]
    raise FileNotFoundError(f"Unable to resolve npy_merged path. Checked: {checked}")


def infer_subdir_for_npy(npy_file: str) -> str:
    if npy_file.startswith(ETRI_ELDERLY_PREFIX):
        return "etri_activity3d_elderly"
    if npy_file.startswith(TOYOTA_PREFIX):
        return "toyota_smarthome"
    raise ValueError(f"Unsupported prefix for file: {npy_file}")


def resolve_npy_file(root: Path, npy_file: str) -> Path:
    return root / infer_subdir_for_npy(npy_file) / npy_file


def filter_etri_elderly(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in metadata if item.get("npy_file", "").startswith(ETRI_ELDERLY_PREFIX)]


def build_subject_id(item: dict[str, Any]) -> str | None:
    person = item.get("person")
    if person is None:
        return None
    npy_file = item.get("npy_file", "")
    if npy_file.startswith(ETRI_ELDERLY_PREFIX):
        return f"etri_elderly_{int(person):03d}"
    if npy_file.startswith(TOYOTA_PREFIX):
        return f"toyota_{int(person):03d}"
    return None


def _filter_eval_to_train_labels(
    train_meta: list[dict[str, Any]],
    eval_meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    train_labels = {item.get("action_id") for item in train_meta if item.get("action_id")}
    return [item for item in eval_meta if item.get("action_id") in train_labels]


def _build_class_maps(
    by_subject: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
    subject_to_classes: dict[str, set[str]] = {}
    class_to_subjects: dict[str, set[str]] = {}
    for sid, items in by_subject.items():
        labels = {item.get("action_id") for item in items if item.get("action_id")}
        if not labels:
            continue
        subject_to_classes[sid] = labels
        for label in labels:
            class_to_subjects.setdefault(label, set()).add(sid)
    excluded = {label for label, subjects in class_to_subjects.items() if len(subjects) < 2}
    for label in excluded:
        class_to_subjects.pop(label, None)
    return class_to_subjects, subject_to_classes, excluded


def _stratified_subject_split(
    by_subject: dict[str, list[dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], set[str]]:
    class_to_subjects, subject_to_classes, excluded = _build_class_maps(by_subject)
    all_sids = sorted(by_subject.keys())
    n_total = len(all_sids)
    n_train = max(1, int(n_total * train_ratio)) if train_ratio > 0 else 0
    n_val = max(0, int(n_total * val_ratio))
    n_test = max(0, n_total - n_train - n_val)

    rng = random.Random(seed)
    assigned: dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    target_counts = {"train": n_train, "val": n_val, "test": n_test}

    def sort_subject(subject_id: str) -> tuple[int, float, str]:
        return (len(subject_to_classes.get(subject_id, set())), rng.random(), subject_id)

    for label in sorted(class_to_subjects, key=lambda key: (len(class_to_subjects[key]), key)):
        candidates = sorted(class_to_subjects[label], key=sort_subject)
        slots = ["train", "val", "test"] if len(candidates) >= 3 else ["train", "test"]
        for slot in slots:
            if counts[slot] >= target_counts[slot]:
                continue
            for sid in candidates:
                if sid not in assigned:
                    assigned[sid] = slot
                    counts[slot] += 1
                    break

    leftovers = [sid for sid in all_sids if sid not in assigned]
    rng.shuffle(leftovers)
    for sid in leftovers:
        if counts["train"] < target_counts["train"]:
            slot = "train"
        elif counts["val"] < target_counts["val"]:
            slot = "val"
        else:
            slot = "test"
        assigned[sid] = slot
        counts[slot] += 1

    train_sids = [sid for sid, slot in assigned.items() if slot == "train"]
    val_sids = [sid for sid, slot in assigned.items() if slot == "val"]
    test_sids = [sid for sid, slot in assigned.items() if slot == "test"]
    return train_sids, val_sids, test_sids, excluded


@dataclass(slots=True)
class SubjectSplit:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    info: dict[str, Any]


class SubjectSplitter:
    """Split metadata by subject with optional label-aware stratification."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
        stratified: bool = True,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.stratified = stratified

    def split(self, metadata: list[dict[str, Any]]) -> SubjectSplit:
        by_subject: dict[str, list[dict[str, Any]]] = {}
        for item in metadata:
            subject_id = build_subject_id(item)
            if subject_id is None:
                continue
            by_subject.setdefault(subject_id, []).append(item)

        if not by_subject:
            return SubjectSplit([], [], [], {"n_train": 0, "n_val": 0, "n_test": 0})

        if self.stratified:
            train_sids, val_sids, test_sids, excluded = _stratified_subject_split(
                by_subject=by_subject,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )
        else:
            rng = random.Random(self.seed)
            subject_ids = sorted(by_subject)
            rng.shuffle(subject_ids)
            n_total = len(subject_ids)
            n_train = max(1, int(n_total * self.train_ratio))
            n_val = max(0, int(n_total * self.val_ratio))
            train_sids = subject_ids[:n_train]
            val_sids = subject_ids[n_train : n_train + n_val]
            test_sids = subject_ids[n_train + n_val :]
            excluded = set()

        train_meta = [
            item
            for sid in train_sids
            for item in by_subject[sid]
            if item.get("action_id") not in excluded
        ]
        val_meta = [
            item
            for sid in val_sids
            for item in by_subject[sid]
            if item.get("action_id") not in excluded
        ]
        test_meta = [
            item
            for sid in test_sids
            for item in by_subject[sid]
            if item.get("action_id") not in excluded
        ]

        val_meta = _filter_eval_to_train_labels(train_meta, val_meta)
        test_meta = _filter_eval_to_train_labels(train_meta, test_meta)

        info = {
            "n_train": len(train_meta),
            "n_val": len(val_meta),
            "n_test": len(test_meta),
            "n_train_subjects": len(train_sids),
            "n_val_subjects": len(val_sids),
            "n_test_subjects": len(test_sids),
            "n_train_labels": len({item.get("action_id") for item in train_meta}),
            "n_val_labels": len({item.get("action_id") for item in val_meta}),
            "n_test_labels": len({item.get("action_id") for item in test_meta}),
            "excluded_labels": sorted(excluded),
        }
        return SubjectSplit(train_meta, val_meta, test_meta, info)


def resample_sequence(sequence: np.ndarray, fixed_len: int) -> np.ndarray:
    """Resample a `(T, J, 3)` sequence to a fixed length with linear interpolation."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape[0] == fixed_len:
        return sequence.astype(np.float32, copy=True)
    if sequence.shape[0] == 1:
        return np.repeat(sequence, fixed_len, axis=0).astype(np.float32)

    old_steps = np.linspace(0.0, 1.0, num=sequence.shape[0], dtype=np.float32)
    new_steps = np.linspace(0.0, 1.0, num=fixed_len, dtype=np.float32)
    output = np.empty((fixed_len, sequence.shape[1], sequence.shape[2]), dtype=np.float32)
    for joint_idx in range(sequence.shape[1]):
        for coord_idx in range(sequence.shape[2]):
            output[:, joint_idx, coord_idx] = np.interp(
                new_steps,
                old_steps,
                sequence[:, joint_idx, coord_idx],
            )
    return output


def extract_toyota_13(sequence_25: np.ndarray) -> np.ndarray:
    """Select the Toyota-compatible 13 joints from an NTU-25 sequence."""
    return np.asarray(sequence_25[:, TOYOTA_TO_NTU_25_MAP, :], dtype=np.float32)


def compute_root_and_scale(sequence_13: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate root center and a stable torso scale from 13 visible joints."""
    hip_center = 0.5 * (
        sequence_13[:, JOINT_NAMES_TOYOTA_13_INDEX["Rhip"], :]
        + sequence_13[:, JOINT_NAMES_TOYOTA_13_INDEX["Lhip"], :]
    )
    shoulder_center = 0.5 * (
        sequence_13[:, JOINT_NAMES_TOYOTA_13_INDEX["Rsho"], :]
        + sequence_13[:, JOINT_NAMES_TOYOTA_13_INDEX["Lsho"], :]
    )
    torso = np.linalg.norm(shoulder_center - hip_center, axis=-1)
    torso = torso[torso > 1e-6]
    if torso.size == 0:
        scale = 1.0
    else:
        scale = float(np.mean(torso))
    return hip_center.astype(np.float32), max(scale, 1e-6)


JOINT_NAMES_TOYOTA_13_INDEX = {
    "Rank": 0,
    "Lank": 1,
    "Rkne": 2,
    "Lkne": 3,
    "Rhip": 4,
    "Lhip": 5,
    "Rwri": 6,
    "Lwri": 7,
    "Relb": 8,
    "Lelb": 9,
    "Rsho": 10,
    "Lsho": 11,
    "Head": 12,
}


def apply_sequence_transform(
    sequence_13: np.ndarray,
    sequence_25: np.ndarray,
    root_center: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    root_center = root_center[:, None, :]
    normalized_13 = (sequence_13 - root_center) / scale
    normalized_25 = (sequence_25 - root_center) / scale
    return normalized_13.astype(np.float32), normalized_25.astype(np.float32)


def random_y_rotation(rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    theta = math.radians(float(rng.uniform(-max_degrees, max_degrees)))
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    return np.asarray(
        [
            [cos_theta, 0.0, sin_theta],
            [0.0, 1.0, 0.0],
            [-sin_theta, 0.0, cos_theta],
        ],
        dtype=np.float32,
    )


def augment_pair(
    input_13: np.ndarray,
    target_25: np.ndarray,
    rotation_deg: float,
    jitter_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rotation_deg > 0:
        rotation = random_y_rotation(rng, rotation_deg)
        input_13 = input_13 @ rotation.T
        target_25 = target_25 @ rotation.T
    if jitter_std > 0:
        noise_13 = rng.normal(0.0, jitter_std, size=input_13.shape).astype(np.float32)
        noise_25 = rng.normal(0.0, jitter_std, size=target_25.shape).astype(np.float32)
        input_13 = input_13 + noise_13
        target_25 = target_25 + noise_25
    return input_13.astype(np.float32), target_25.astype(np.float32)


def prepare_sequence_pair(
    sequence_25: np.ndarray,
    fixed_len: int,
    apply_augmentation: bool,
    rotation_deg: float,
    jitter_std: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    if sequence_25.ndim == 4:
        sequence_25 = sequence_25[:, 0, :, :]
    if sequence_25.ndim != 3 or sequence_25.shape[1:] != (25, 3):
        raise ValueError(f"Expected (T, 25, 3), got {sequence_25.shape}")

    resampled = resample_sequence(sequence_25, fixed_len=fixed_len)
    input_13 = extract_toyota_13(resampled)
    root_center, scale = compute_root_and_scale(input_13)
    input_13_norm, target_25_norm = apply_sequence_transform(input_13, resampled, root_center, scale)

    if apply_augmentation:
        rng = np.random.default_rng(seed)
        input_13_norm, target_25_norm = augment_pair(
            input_13=input_13_norm,
            target_25=target_25_norm,
            rotation_deg=rotation_deg,
            jitter_std=jitter_std,
            rng=rng,
        )

    observed_mask = np.zeros(25, dtype=np.float32)
    observed_mask[TOYOTA_TO_NTU_25_MAP] = 1.0
    return {
        "input_13": input_13_norm.astype(np.float32),
        "target_25": target_25_norm.astype(np.float32),
        "root_center": root_center.astype(np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
        "observed_mask": observed_mask,
    }


def prepare_inference_input(
    sequence: np.ndarray,
    fixed_len: int,
) -> dict[str, np.ndarray | float]:
    """Prepare a 13-joint or 25-joint sequence for SSR inference."""
    if sequence.ndim == 4:
        sequence = sequence[:, 0, :, :]
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError(f"Expected (T, J, 3), got {sequence.shape}")

    if sequence.shape[1] == 25:
        sequence_25 = resample_sequence(sequence, fixed_len=fixed_len)
        input_13 = extract_toyota_13(sequence_25)
    elif sequence.shape[1] == 13:
        input_13 = resample_sequence(sequence, fixed_len=fixed_len)
    else:
        raise ValueError(f"Expected 13 or 25 joints, got {sequence.shape[1]}")

    root_center, scale = compute_root_and_scale(input_13)
    input_13_norm = (input_13 - root_center[:, None, :]) / scale
    return {
        "input_13": input_13_norm.astype(np.float32),
        "root_center": root_center.astype(np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
    }


def restore_prediction(
    prediction_norm: np.ndarray,
    root_center: np.ndarray,
    scale: float | np.ndarray,
) -> np.ndarray:
    """Restore normalized 25-joint predictions back to original coordinates."""
    scale_value = float(np.asarray(scale).reshape(()))
    return prediction_norm * scale_value + root_center[:, None, :]


class SSRDataset(Dataset):
    """Dataset returning normalized `(13, 25)` skeleton pairs."""

    def __init__(
        self,
        root: Path,
        metadata: list[dict[str, Any]],
        fixed_len: int,
        augment: bool = False,
        rotation_deg: float = 0.0,
        jitter_std: float = 0.0,
        base_seed: int = 42,
    ) -> None:
        self.root = root
        self.metadata = metadata
        self.fixed_len = fixed_len
        self.augment = augment
        self.rotation_deg = rotation_deg
        self.jitter_std = jitter_std
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required to iterate SSRDataset.")
        item = self.metadata[index]
        npy_path = resolve_npy_file(self.root, item["npy_file"])
        sequence_25 = np.load(npy_path, allow_pickle=False).astype(np.float32)
        prepared = prepare_sequence_pair(
            sequence_25=sequence_25,
            fixed_len=self.fixed_len,
            apply_augmentation=self.augment,
            rotation_deg=self.rotation_deg,
            jitter_std=self.jitter_std,
            seed=self.base_seed + index,
        )
        return {
            "input_13": torch.from_numpy(prepared["input_13"]).float(),
            "target_25": torch.from_numpy(prepared["target_25"]).float(),
            "observed_mask": torch.from_numpy(prepared["observed_mask"]).float(),
            "root_center": torch.from_numpy(prepared["root_center"]).float(),
            "scale": torch.as_tensor(prepared["scale"]).float(),
            "action_id": item.get("action_id", ""),
            "npy_file": item.get("npy_file", ""),
        }


def build_etri_elderly_split(cfg: dict[str, Any]) -> tuple[Path, SubjectSplit]:
    root = resolve_npy_merged_path(cfg)
    metadata = load_json(root / "metadata.json")
    elderly_meta = filter_etri_elderly(metadata)
    dataset_cfg = cfg.get("dataset") or {}
    splitter = SubjectSplitter(
        train_ratio=float(dataset_cfg.get("train_ratio", 0.7)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.2)),
        test_ratio=float(dataset_cfg.get("test_ratio", 0.1)),
        seed=int((cfg.get("experiment") or {}).get("seed", 42)),
        stratified=bool(dataset_cfg.get("stratified_subject_split", True)),
    )
    return root, splitter.split(elderly_meta)
