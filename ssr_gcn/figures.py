"""Figure export for SSR runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_all_figures(
    training_log: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figures] WARNING: matplotlib not installed; skipping figure export.")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _save_curve(plt, training_log, fig_dir, "loss", "Loss", ("train_loss", "val_loss"))
    _save_curve(plt, training_log, fig_dir, "mpjpe", "MPJPE", ("train_mpjpe", "val_mpjpe"))
    _save_curve(
        plt,
        training_log,
        fig_dir,
        "bone_error",
        "Bone Error",
        ("train_bone_error", "val_bone_error"),
    )
    _save_curve(
        plt,
        training_log,
        fig_dir,
        "throughput",
        "Samples / sec",
        ("train_samples_per_sec", "val_samples_per_sec"),
    )
    _save_summary_bar(plt, test_metrics, fig_dir)
    _save_per_joint_bar(plt, test_metrics, fig_dir)


def _save_curve(
    plt: Any,
    training_log: list[dict[str, Any]],
    fig_dir: Path,
    name: str,
    ylabel: str,
    keys: tuple[str, str],
) -> None:
    if not training_log:
        return
    epochs = [item["epoch"] for item in training_log]
    train_values = [item.get(keys[0], 0.0) for item in training_log]
    val_values = [item.get(keys[1], 0.0) for item in training_log]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_values, marker="o", markersize=3, linewidth=1.5, label="Train")
    ax.plot(epochs, val_values, marker="s", markersize=3, linewidth=1.5, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_curve.png", dpi=120)
    plt.close(fig)


def _save_summary_bar(plt: Any, test_metrics: dict[str, Any], fig_dir: Path) -> None:
    keys = ["loss", "mpjpe", "missing_mpjpe", "visible_mpjpe", "bone_error"]
    values = [float(test_metrics.get(key, 0.0)) for key in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(keys, values, color=["#5B8FF9", "#61DDAA", "#65789B", "#F6BD16", "#E8684A"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    ax.set_ylabel("Metric")
    ax.set_title("Test Metrics Summary")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "test_metrics_summary.png", dpi=120)
    plt.close(fig)


def _save_per_joint_bar(plt: Any, test_metrics: dict[str, Any], fig_dir: Path) -> None:
    per_joint = test_metrics.get("per_joint_mpjpe") or {}
    if not per_joint:
        return

    labels = list(per_joint.keys())
    values = list(per_joint.values())
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(values)), values, color="#5D7092")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("MPJPE")
    ax.set_title("Per-joint MPJPE (test)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_joint_mpjpe.png", dpi=120)
    plt.close(fig)
