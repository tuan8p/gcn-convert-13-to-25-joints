#!/usr/bin/env python3
"""Entry-point for SSR GCN training.

Rule: every CLI flag is optional. If omitted, values come from the YAML config.
If provided, CLI overrides that key in the loaded config.

    python tools/train.py

    python tools/train.py --runtime-profile local_debug --subset-ratio 0.02 --max-epochs 2

Multi-GPU:

    torchrun --standalone --nproc_per_node=2 tools/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssr_gcn.config import load_cfg
from ssr_gcn.constants import DEFAULT_CONFIG_PATH
from ssr_gcn.engine import run


_EPILOG = """
Override policy: omit a flag → use YAML; pass a flag → override YAML for that key only.
Maps: --runtime-profile → runtime_profile merge; --max-epochs → training.epochs;
--subset-ratio → experiment.subset_ratio; --lr → training.lr;
--per-gpu-batch-size → training.per_gpu_batch_size.
"""


def _apply_training_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Patch cfg.training from optional CLI (only when argument was passed)."""
    t = cfg.setdefault("training", {})
    if getattr(args, "lr", None) is not None:
        t["lr"] = float(args.lr)
    if getattr(args, "per_gpu_batch_size", None) is not None:
        t["per_gpu_batch_size"] = int(args.per_gpu_batch_size)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Keep epilog line breaks and show defaults."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SSR GCN for 13->25 skeleton super-resolution.",
        formatter_class=_HelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config (default: configs/ssr_gcn_kaggle.yaml next to package root).",
    )
    parser.add_argument(
        "--runtime-profile",
        type=str,
        default=None,
        dest="runtime_profile",
        help="Override runtime.profile in config (default: use runtime.profile from YAML).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        dest="max_epochs",
        help="Override training.epochs in config when set.",
    )
    parser.add_argument(
        "--subset-ratio",
        type=float,
        default=None,
        dest="subset_ratio",
        help="Override experiment.subset_ratio when set (1.0 = full data).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        dest="lr",
        help="Override training.lr when set.",
    )
    parser.add_argument(
        "--per-gpu-batch-size",
        type=int,
        default=None,
        dest="per_gpu_batch_size",
        help="Override training.per_gpu_batch_size when set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_cfg(args.config, runtime_profile=args.runtime_profile)
    _apply_training_cli_overrides(cfg, args)
    return run(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
