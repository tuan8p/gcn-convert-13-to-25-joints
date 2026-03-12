#!/usr/bin/env python3
"""Entry-point for SSR GCN training.
python tools/train.py --config configs/ssr_gcn_kaggle.yaml --runtime-profile local_debug --subset-ratio 0.02 --max-epochs 2

python tools/train.py --config configs/ssr_gcn_kaggle.yaml --runtime-profile t4

torchrun --standalone --nproc_per_node=2 tools/train.py --config configs/ssr_gcn_kaggle.yaml --runtime-profile t4


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_gcn.config import load_cfg
from ssr_gcn.constants import DEFAULT_CONFIG_PATH
from ssr_gcn.engine import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SSR GCN for 13->25 skeleton super-resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--runtime-profile", type=str, default=None, dest="runtime_profile")
    parser.add_argument("--max-epochs", type=int, default=None, dest="max_epochs")
    parser.add_argument("--subset-ratio", type=float, default=1.0, dest="subset_ratio")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.config, runtime_profile=args.runtime_profile)
    return run(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
