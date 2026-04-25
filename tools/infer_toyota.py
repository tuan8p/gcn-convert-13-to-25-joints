#!/usr/bin/env python3
"""Infer 25-joint skeletons from Toyota-style 13-joint input.

python tools/infer_toyota.py \
  --checkpoint outputs/ssr_gcn/<run>/best_model.pt \
  --input-dir ../data/processed/npy_merged/toyota_smarthome \
  --output-dir outputs/toyota_infer \
  --preserve-length          # keep original n_frames (recommended for EAR downstream)

# Kaggle: sau khi train, không cần sửa tên thư mục run_* — dùng:
#   --use-latest-checkpoint
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_gcn.config import load_cfg
from ssr_gcn.constants import DEFAULT_CONFIG_PATH
from ssr_gcn.data import (
    compute_root_and_scale,
    extract_toyota_13,
    prepare_inference_input,
    restore_prediction,
)
from ssr_gcn.model import create_model


def _find_latest_best_model(search_root: Path) -> Path:
    """Newest `outputs/ssr_gcn/run_*/best_model.pt` by mtime (Kaggle: avoid hardcoding run id)."""
    if not search_root.is_dir():
        raise FileNotFoundError(
            f"SSR output root not found: {search_root}. Train first (tools/train.py) "
            f"or set --checkpoint to a .pt file."
        )
    candidates = list(search_root.glob("run_*/best_model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No run_*/best_model.pt under {search_root}. Run training in this session or "
            f"point --checkpoint to a .pt (e.g. from a Kaggle output dataset)."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SSR inference on Toyota 13-joint sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to best_model.pt. Omit if using --use-latest-checkpoint.",
    )
    parser.add_argument(
        "--use-latest-checkpoint",
        action="store_true",
        help=(
            "Use the newest best_model.pt under <repo>/outputs/ssr_gcn/run_*/ "
            "(saves hardcoding run_* folder on Kaggle after each train)."
        ),
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=None,
        dest="outputs_root",
        help="With --use-latest-checkpoint, search this dir for run_*/best_model.pt (default: <repo>/outputs/ssr_gcn).",
    )
    parser.add_argument("--input-dir", type=str, required=True, dest="input_dir")
    parser.add_argument("--output-dir", type=str, required=True, dest="output_dir")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--runtime-profile", type=str, default=None, dest="runtime_profile")
    parser.add_argument("--glob", type=str, default="*.npy")
    parser.add_argument(
        "--preserve-length",
        action="store_true",
        dest="preserve_length",
        help=(
            "Keep original n_frames instead of resampling to fixed_len. "
            "Recommended for EAR downstream pipelines (MotionBERT, SkateFormer) "
            "that handle their own temporal sampling."
        ),
    )
    return parser.parse_args()


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"  Train in this Kaggle session (new run_ folder), then use "
            f"--use-latest-checkpoint or pass the correct path to best_model.pt."
        )
    model = create_model(cfg).to(device)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _prepare_preserve_length(sequence: np.ndarray) -> dict:
    """Normalize without resampling — keep original T."""
    if sequence.ndim == 4:
        sequence = sequence[:, 0, :, :]
    if sequence.ndim != 3 or sequence.shape[-1] != 3:
        raise ValueError(f"Expected (T, J, 3), got {sequence.shape}")
    if sequence.shape[1] == 25:
        input_13 = extract_toyota_13(sequence)
    elif sequence.shape[1] == 13:
        input_13 = sequence.astype(np.float32)
    else:
        raise ValueError(f"Expected 13 or 25 joints, got {sequence.shape[1]}")

    root_center, scale = compute_root_and_scale(input_13)
    input_13_norm = (input_13 - root_center[:, None, :]) / scale
    return {
        "input_13": input_13_norm.astype(np.float32),
        "root_center": root_center.astype(np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
    }


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if not args.use_latest_checkpoint and not args.checkpoint:
        raise SystemExit("Provide --checkpoint PATH or --use-latest-checkpoint (see --help).")
    cfg = load_cfg(args.config, runtime_profile=args.runtime_profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_latest_checkpoint:
        out_root = (
            Path(args.outputs_root).resolve()
            if args.outputs_root
            else (PROJECT_ROOT / "outputs" / "ssr_gcn")
        )
        checkpoint_path = _find_latest_best_model(out_root)
        print(f"[infer] Using latest checkpoint: {checkpoint_path}", flush=True)
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(cfg, checkpoint_path, device)
    fixed_len = int((cfg.get("preprocessing") or {}).get("fixed_len", 150))
    npy_files = sorted(input_dir.rglob(args.glob))
    if not npy_files:
        raise FileNotFoundError(f"No `.npy` files found in {input_dir}")

    mode = "preserve_length" if args.preserve_length else f"fixed_len={fixed_len}"
    print(f"[infer] {len(npy_files)} files  mode={mode}  device={device}", flush=True)

    manifest: list[dict] = []
    for npy_file in tqdm(npy_files, desc="Infer Toyota", unit="file"):
        sequence = np.load(npy_file, allow_pickle=False).astype(np.float32)
        orig_frames = int(sequence.shape[0])

        if args.preserve_length:
            prepared = _prepare_preserve_length(sequence)
        else:
            prepared = prepare_inference_input(sequence, fixed_len=fixed_len)

        inputs = torch.from_numpy(prepared["input_13"]).unsqueeze(0).to(device)
        prediction_norm = model(inputs).squeeze(0).cpu().numpy()
        prediction = restore_prediction(
            prediction_norm=prediction_norm,
            root_center=np.asarray(prepared["root_center"]),
            scale=np.asarray(prepared["scale"]),
        )

        relative_path = npy_file.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, prediction.astype(np.float32))
        manifest.append(
            {
                "input_file": str(npy_file),
                "output_file": str(output_path),
                "input_joints": int(sequence.shape[1]),
                "orig_frames": orig_frames,
                "output_shape": list(prediction.shape),
            }
        )

    with (output_dir / "inference_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f"[done] Saved {len(manifest)} inferred sequences to {output_dir}", flush=True)

    zip_path = shutil.make_archive(
        str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    )
    print(f"[done] Zipped to {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
