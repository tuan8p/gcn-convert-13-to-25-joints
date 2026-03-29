#!/usr/bin/env python3
"""Evaluate quality of SSR inference output without ground-truth 25-joint labels.

Metrics computed (all ground-truth-free):
  bone_len_cv       -- Coefficient of Variation of bone lengths across frames.
                       Low CV = temporally stable bone lengths (anatomically consistent).
  symmetry_error    -- Abs diff between left/right symmetric joint distances to body center.
                       Near 0 = anatomically plausible symmetric skeleton.
  velocity_smoothness -- Mean per-joint velocity std across frames.
                         Low = smooth motion without sudden jumps.
  joint_range       -- Mean range (max-min) of each joint's world position.
                         Cross-check that range is physically plausible.

Usage:
    python tools/evaluate_infer_quality.py \
        --infer-dir outputs/toyota_infer \
        --output    outputs/toyota_infer/quality_report.json \
        --samples   200     # how many files to sample (default: all)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_gcn.constants import NTU_EDGES


# Left-right symmetric joint pairs (NTU 25-joint layout)
_SYMMETRIC_PAIRS: list[tuple[int, int]] = [
    (4, 8),   # ShoulderLeft  ↔ ShoulderRight
    (5, 9),   # ElbowLeft     ↔ ElbowRight
    (6, 10),  # WristLeft     ↔ WristRight
    (7, 11),  # HandLeft      ↔ HandRight
    (12, 16), # HipLeft       ↔ HipRight
    (13, 17), # KneeLeft      ↔ KneeRight
    (14, 18), # AnkleLeft     ↔ AnkleRight
    (15, 19), # FootLeft      ↔ FootRight
    (21, 23), # HandTipLeft   ↔ HandTipRight
    (22, 24), # ThumbLeft     ↔ ThumbRight
]


def _bone_len_cv(seq: np.ndarray) -> float:
    """Mean CV of bone lengths over time. Lower = more stable."""
    cvs = []
    for src, dst in NTU_EDGES:
        lengths = np.linalg.norm(seq[:, src, :] - seq[:, dst, :], axis=-1)
        if lengths.mean() > 1e-6:
            cvs.append(lengths.std() / lengths.mean())
    return float(np.mean(cvs)) if cvs else 0.0


def _symmetry_error(seq: np.ndarray) -> float:
    """Mean abs diff between left/right joint distances to spine center (joint 0)."""
    spine = seq[:, 0, :]  # SpineBase as reference
    errors = []
    for left_idx, right_idx in _SYMMETRIC_PAIRS:
        dist_left = np.linalg.norm(seq[:, left_idx, :] - spine, axis=-1)
        dist_right = np.linalg.norm(seq[:, right_idx, :] - spine, axis=-1)
        errors.append(float(np.mean(np.abs(dist_left - dist_right))))
    return float(np.mean(errors)) if errors else 0.0


def _velocity_smoothness(seq: np.ndarray) -> float:
    """Mean per-joint velocity std — lower means smoother motion."""
    velocity = np.linalg.norm(np.diff(seq, axis=0), axis=-1)  # (T-1, J)
    return float(velocity.std(axis=0).mean())


def _joint_range(seq: np.ndarray) -> float:
    """Mean positional range (max-min) per joint per axis."""
    ranges = seq.max(axis=0) - seq.min(axis=0)  # (J, 3)
    return float(ranges.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SSR inference quality (no ground truth needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--infer-dir", type=str, required=True, dest="infer_dir")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--samples", type=int, default=0,
                        help="Max files to evaluate (0 = all)")
    parser.add_argument("--glob", type=str, default="*.npy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    infer_dir = Path(args.infer_dir).resolve()
    npy_files = sorted(infer_dir.rglob(args.glob))
    # exclude any metadata npy accidentally matched
    npy_files = [f for f in npy_files if "manifest" not in f.name]

    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {infer_dir}")

    if args.samples > 0 and len(npy_files) > args.samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(npy_files), args.samples, replace=False)
        npy_files = [npy_files[i] for i in sorted(idx)]

    print(f"[eval] Evaluating {len(npy_files)} inferred sequences...", flush=True)

    bone_cvs, sym_errors, vel_smooths, joint_ranges = [], [], [], []
    shape_errors = []

    for npy_file in tqdm(npy_files, desc="Quality check", unit="file"):
        seq = np.load(npy_file, allow_pickle=False).astype(np.float32)

        if seq.ndim != 3 or seq.shape[1] != 25 or seq.shape[2] != 3:
            shape_errors.append(str(npy_file))
            continue
        if seq.shape[0] < 2:
            continue

        bone_cvs.append(_bone_len_cv(seq))
        sym_errors.append(_symmetry_error(seq))
        vel_smooths.append(_velocity_smoothness(seq))
        joint_ranges.append(_joint_range(seq))

    def _stats(values: list[float]) -> dict:
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    report = {
        "n_files": len(npy_files),
        "n_evaluated": len(bone_cvs),
        "n_shape_errors": len(shape_errors),
        "metrics": {
            "bone_len_cv": _stats(bone_cvs),
            "symmetry_error": _stats(sym_errors),
            "velocity_smoothness": _stats(vel_smooths),
            "joint_range": _stats(joint_ranges),
        },
        "interpretation": {
            "bone_len_cv": "< 0.05 excellent  |  0.05–0.10 good  |  > 0.15 unstable",
            "symmetry_error": "< 0.03 excellent  |  0.03–0.08 good  |  > 0.15 asymmetric",
            "velocity_smoothness": "lower = smoother  |  > 0.05 may have jitter",
            "joint_range": "sanity check: expect 0.5–3.0 for normalized skeletons",
        },
    }

    print("\n=== Inference Quality Report ===")
    for name, stats in report["metrics"].items():
        interp = report["interpretation"][name]
        print(f"\n{name}:")
        print(f"  mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"p50={stats['p50']:.4f}  p95={stats['p95']:.4f}")
        print(f"  [{interp}]")

    if shape_errors:
        print(f"\n[warn] {len(shape_errors)} files with unexpected shape: {shape_errors[:3]}")

    out_path = Path(args.output) if args.output else infer_dir / "quality_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[done] Report saved to {out_path}", flush=True)

    import shutil
    zip_path = shutil.make_archive(
        str(infer_dir), "zip", root_dir=infer_dir.parent, base_dir=infer_dir.name
    )
    print(f"[done] Zipped infer output to {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
