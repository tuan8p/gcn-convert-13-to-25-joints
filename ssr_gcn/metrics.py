"""Losses and metrics for SSR regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ssr_gcn.constants import MISSING_NTU_JOINTS, NTU_EDGES, TOYOTA_TO_NTU_25_MAP


def mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-joint position error."""
    return torch.linalg.norm(prediction - target, dim=-1).mean()


def _select_joints(tensor: torch.Tensor, joint_indices: list[int]) -> torch.Tensor:
    device = tensor.device
    index_tensor = torch.as_tensor(joint_indices, dtype=torch.long, device=device)
    return tensor.index_select(dim=2, index=index_tensor)


def missing_joint_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mpjpe(_select_joints(prediction, MISSING_NTU_JOINTS), _select_joints(target, MISSING_NTU_JOINTS))


def visible_joint_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mpjpe(_select_joints(prediction, TOYOTA_TO_NTU_25_MAP), _select_joints(target, TOYOTA_TO_NTU_25_MAP))


def bone_vectors(sequence: torch.Tensor) -> torch.Tensor:
    src = torch.as_tensor([edge[0] for edge in NTU_EDGES], dtype=torch.long, device=sequence.device)
    dst = torch.as_tensor([edge[1] for edge in NTU_EDGES], dtype=torch.long, device=sequence.device)
    return sequence[:, :, src, :] - sequence[:, :, dst, :]


def bone_length_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bones = torch.linalg.norm(bone_vectors(prediction), dim=-1)
    target_bones = torch.linalg.norm(bone_vectors(target), dim=-1)
    return F.mse_loss(pred_bones, target_bones)


def bone_length_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bones = torch.linalg.norm(bone_vectors(prediction), dim=-1)
    target_bones = torch.linalg.norm(bone_vectors(target), dim=-1)
    return torch.mean(torch.abs(pred_bones - target_bones))


def total_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    joint_weight: float,
    bone_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    joint_loss = F.mse_loss(prediction, target)
    bone_loss = bone_length_loss(prediction, target)
    loss = joint_weight * joint_loss + bone_weight * bone_loss
    stats = {
        "joint_loss": float(joint_loss.detach().item()),
        "bone_loss": float(bone_loss.detach().item()),
    }
    return loss, stats


class MetricTracker:
    """Accumulate weighted averages for epoch-level reporting."""

    def __init__(self) -> None:
        self.total = 0
        self.sums: dict[str, float] = {}

    def update(self, count: int, **metrics: float) -> None:
        self.total += count
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * count

    def compute(self) -> dict[str, float]:
        if self.total <= 0:
            return {}
        return {key: value / self.total for key, value in self.sums.items()}


def per_joint_mpjpe(prediction_batches: list[np.ndarray], target_batches: list[np.ndarray]) -> dict[str, float]:
    prediction = np.concatenate(prediction_batches, axis=0)
    target = np.concatenate(target_batches, axis=0)
    joint_errors = np.linalg.norm(prediction - target, axis=-1).mean(axis=(0, 1))
    return {f"joint_{idx:02d}": float(err) for idx, err in enumerate(joint_errors)}


def summarize_test_metrics(
    tracker: MetricTracker,
    prediction_batches: list[np.ndarray],
    target_batches: list[np.ndarray],
) -> dict[str, Any]:
    metrics = tracker.compute()
    metrics["per_joint_mpjpe"] = per_joint_mpjpe(prediction_batches, target_batches)
    return metrics
