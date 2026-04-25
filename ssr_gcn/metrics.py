"""Losses and metrics for SSR regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ssr_gcn.constants import (
    EXTREMITY_HEAVY_MISSING_NTU,
    MISSING_NTU_JOINTS,
    NTU_EDGES,
    TORSO_MISSING_NTU,
    TOYOTA_TO_NTU_25_MAP,
    VISIBLE_NTU_JOINTS,
)


def mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-joint position error."""
    return torch.linalg.norm(prediction - target, dim=-1).mean()


def _select_joints(tensor: torch.Tensor, joint_indices: list[int]) -> torch.Tensor:
    device = tensor.device
    index_tensor = torch.as_tensor(joint_indices, dtype=torch.long, device=device)
    return tensor.index_select(dim=2, index=index_tensor)


def missing_joint_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mpjpe(_select_joints(prediction, MISSING_NTU_JOINTS), _select_joints(target, MISSING_NTU_JOINTS))


def extremity_missing_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MPJPE chỉ 9 khớp thiếu ở tay/chân/đầu số (không tính SpineBase/Mid/Neck)."""
    idx = list(EXTREMITY_HEAVY_MISSING_NTU)
    return mpjpe(_select_joints(prediction, idx), _select_joints(target, idx))


def torso_missing_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MPJPE 3 khớp thân thiếu (0,1,2) — khó ổn định nhưng khác tầng tay/chân."""
    idx = list(TORSO_MISSING_NTU)
    return mpjpe(_select_joints(prediction, idx), _select_joints(target, idx))


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


def build_joint_mse_weight_vector(
    device: torch.device,
    dtype: torch.dtype,
    loss_cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Một vector (25,) trọng số MSE theo từng joint: visible (Toyota-13) vs thiếu (torso / extremity)."""
    loss_cfg = loss_cfg or {}
    w_vis = float(loss_cfg.get("visible_mse_weight", 1.0))
    w_m_base = float(loss_cfg.get("missing_base_mse_weight", 1.0))
    w_torso = float(loss_cfg.get("missing_torso_mult", 1.0))
    w_ext = float(loss_cfg.get("missing_extremity_mult", 1.0))
    w = torch.empty(25, device=device, dtype=dtype)
    for j in range(25):
        if j in VISIBLE_NTU_JOINTS:
            w[j] = w_vis
        elif j in TORSO_MISSING_NTU:
            w[j] = w_m_base * w_torso
        else:
            w[j] = w_m_base * w_ext
    return w


def joint_mse_loss_weighted(
    prediction: torch.Tensor,
    target: torch.Tensor,
    joint_weight_vector: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean MSE: mean over (B,T) of sum_j w_j * mse_j / sum_j w_j; mse per joint = mean over xyz."""
    diff_sq = (prediction - target) ** 2
    per_joint = diff_sq.mean(dim=(0, 1, 3))
    w = joint_weight_vector.to(device=per_joint.device, dtype=per_joint.dtype)
    return (per_joint * w).sum() / w.sum().clamp_min(1e-8)


def bone_length_loss_weighted(
    prediction: torch.Tensor,
    target: torch.Tensor,
    joint_weight_vector: torch.Tensor,
) -> torch.Tensor:
    """MSE độ dài xương, trọng số cạnh = trung bình trọng số 2 joint."""
    pred_b = torch.linalg.norm(bone_vectors(prediction), dim=-1)
    tgt_b = torch.linalg.norm(bone_vectors(target), dim=-1)
    n_edges = int(pred_b.shape[2])
    src = torch.as_tensor([e[0] for e in NTU_EDGES], device=pred_b.device, dtype=torch.long)
    dst = torch.as_tensor([e[1] for e in NTU_EDGES], device=pred_b.device, dtype=torch.long)
    wj = joint_weight_vector.to(device=pred_b.device, dtype=pred_b.dtype)
    w_edge = 0.5 * (wj[src] + wj[dst])
    err = (pred_b - tgt_b) ** 2
    b, t, _e = err.shape
    return (err * w_edge.view(1, 1, n_edges)).sum() / (b * t * w_edge.sum().clamp_min(1e-8))


def bone_length_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bones = torch.linalg.norm(bone_vectors(prediction), dim=-1)
    target_bones = torch.linalg.norm(bone_vectors(target), dim=-1)
    return torch.mean(torch.abs(pred_bones - target_bones))


def total_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    joint_weight: float,
    bone_weight: float,
    joint_mse_weight_vector: torch.Tensor | None = None,
    use_weighted_bone: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if joint_mse_weight_vector is None:
        joint_loss = F.mse_loss(prediction, target)
        bone_loss = bone_length_loss(prediction, target)
    else:
        w = joint_mse_weight_vector
        joint_loss = joint_mse_loss_weighted(prediction, target, w)
        bone_loss = (
            bone_length_loss_weighted(prediction, target, w)
            if use_weighted_bone
            else bone_length_loss(prediction, target)
        )
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


def val_score_for_checkpoint(
    val_metrics: dict[str, float],
    experiment_cfg: dict[str, Any] | None = None,
) -> tuple[str, str, float]:
    """Chọn score để lưu best checkpoint / early stopping (càng thấp càng tốt).

    Returns:
        (metric_name, val_key, score) với `metric_name` mô tả cách chọn, `val_key` key trong log
        tương ứng (trừ combined dùng ``val_combined_score``).
    """
    exp = experiment_cfg or {}
    name = str(exp.get("best_val_metric", "mpjpe")).strip().lower()
    m = {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))}

    def _f(key: str, default: float = 0.0) -> float:
        return float(m.get(key, default))

    if name in ("", "val_mpjpe", "mpjpe"):
        return "mpjpe", "val_mpjpe", _f("mpjpe")
    if name in ("extremity_missing_mpjpe", "extremity", "ex_m", "ext"):
        return "extremity_missing_mpjpe", "val_extremity_missing_mpjpe", _f("extremity_missing_mpjpe")
    if name in ("torso_missing_mpjpe", "torso", "torso_missing"):
        return "torso_missing_mpjpe", "val_torso_missing_mpjpe", _f("torso_missing_mpjpe")
    if name in ("missing_mpjpe", "missing"):
        return "missing_mpjpe", "val_missing_mpjpe", _f("missing_mpjpe")
    if name in ("combined", "blend", "mpjpe_extremity"):
        w_m = float(exp.get("combined_val_w_mpjpe", 0.5))
        w_e = float(exp.get("combined_val_w_extremity", 0.5))
        w_sum = w_m + w_e
        if w_sum <= 0:
            w_m, w_e, w_sum = 0.5, 0.5, 1.0
        w_m /= w_sum
        w_e /= w_sum
        s = w_m * _f("mpjpe") + w_e * _f("extremity_missing_mpjpe")
        return "combined", "val_combined_score", s
    return "mpjpe", "val_mpjpe", _f("mpjpe")
