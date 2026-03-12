#!/usr/bin/env python3
"""Smoke test for SSR preprocessing without requiring torch.
python tools/smoke_test_preprocessing.py --runtime-profile local_debug --samples 3 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_gcn.config import load_cfg
from ssr_gcn.constants import DEFAULT_CONFIG_PATH
from ssr_gcn.data import build_etri_elderly_split, prepare_sequence_pair, resolve_npy_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a smoke test for SSR preprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--runtime-profile", type=str, default=None, dest="runtime_profile")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.config, runtime_profile=args.runtime_profile)
    root, split = build_etri_elderly_split(cfg)
    fixed_len = int((cfg.get("preprocessing") or {}).get("fixed_len", 150))

    records = split.train[: args.samples]
    if not records:
        raise RuntimeError("No training samples found for smoke test.")

    print(f"[smoke] npy_merged_root={root}")
    print(f"[smoke] split_info={json.dumps(split.info, ensure_ascii=False)}")

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, item in enumerate(records):
        npy_path = resolve_npy_file(root, item["npy_file"])
        sequence_25 = np.load(npy_path, allow_pickle=False).astype(np.float32)
        prepared = prepare_sequence_pair(
            sequence_25=sequence_25,
            fixed_len=fixed_len,
            apply_augmentation=False,
            rotation_deg=0.0,
            jitter_std=0.0,
            seed=index,
        )
        input_13 = np.asarray(prepared["input_13"], dtype=np.float32)
        target_25 = np.asarray(prepared["target_25"], dtype=np.float32)
        inputs.append(input_13)
        targets.append(target_25)
        print(
            f"[sample {index}] file={item['npy_file']} "
            f"input_shape={tuple(input_13.shape)} target_shape={tuple(target_25.shape)} "
            f"scale={float(np.asarray(prepared['scale'])):.4f}"
        )

    input_batch = np.stack(inputs[: args.batch_size], axis=0)
    target_batch = np.stack(targets[: args.batch_size], axis=0)
    assert input_batch.shape[1:] == (fixed_len, 13, 3)
    assert target_batch.shape[1:] == (fixed_len, 25, 3)
    print(f"[batch] input_batch={tuple(input_batch.shape)} target_batch={tuple(target_batch.shape)}")
    print("[smoke] preprocessing smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
