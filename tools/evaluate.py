#!/usr/bin/env python3
"""Evaluate a trained SSR checkpoint.
python tools/evaluate.py --checkpoint outputs/ssr_gcn/<run>/best_model.pt --config configs/ssr_gcn_kaggle.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_gcn.config import load_cfg
from ssr_gcn.constants import DEFAULT_CONFIG_PATH
from ssr_gcn.engine import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SSR checkpoint on val/test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--runtime-profile", type=str, default=None, dest="runtime_profile")
    parser.add_argument("--subset-ratio", type=float, default=1.0, dest="subset_ratio")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.config, runtime_profile=args.runtime_profile)
    metrics = evaluate_checkpoint(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        split_name=args.split,
        subset_ratio=args.subset_ratio,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
