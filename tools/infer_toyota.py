#!/usr/bin/env python3
"""Infer 25-joint skeletons from Toyota-style 13-joint input.
python tools/infer_toyota.py --checkpoint outputs/ssr_gcn/<run>/best_model.pt --input-dir ../data/processed/npy_merged/toyota_smarthome --output-dir outputs/toyota_infer --config configs/ssr_gcn_kaggle.yaml
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
from ssr_gcn.data import prepare_inference_input, restore_prediction
from ssr_gcn.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SSR inference on Toyota 13-joint sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True, dest="input_dir")
    parser.add_argument("--output-dir", type=str, required=True, dest="output_dir")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--runtime-profile", type=str, default=None, dest="runtime_profile")
    parser.add_argument("--glob", type=str, default="*.npy")
    return parser.parse_args()


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = create_model(cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.config, runtime_profile=args.runtime_profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(cfg, checkpoint_path, device)
    fixed_len = int((cfg.get("preprocessing") or {}).get("fixed_len", 150))
    npy_files = sorted(input_dir.rglob(args.glob))
    if not npy_files:
        raise FileNotFoundError(f"No `.npy` files found in {input_dir}")

    manifest: list[dict[str, str | int | float]] = []
    for npy_file in tqdm(npy_files, desc="Infer Toyota", unit="file"):
        sequence = np.load(npy_file, allow_pickle=False).astype(np.float32)
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
                "output_shape": list(prediction.shape),
            }
        )

    with (output_dir / "inference_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f"[done] Saved {len(manifest)} inferred sequences to {output_dir}")

    zip_path = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
    print(f"[done] Zipped to {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
