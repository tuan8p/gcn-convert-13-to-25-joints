"""Minimal DDP helpers for Kaggle and local training."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank0() -> bool:
    return rank() == 0


def setup_distributed(backend: str = "nccl") -> torch.device:
    """Initialize distributed training if launched via torchrun.

    Sets NCCL environment variables before init_process_group to avoid
    hostname-resolution failures on shared cloud environments (Kaggle, etc.).
    Variables are only set when not already configured by the caller.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Disable P2P and InfiniBand; use any non-loopback/docker interface.
        # These defaults prevent the "hostname cannot be retrieved" NCCL warning
        # on Kaggle 2×T4 and similar shared PCIe setups.
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
            return torch.device("cuda", local_rank())
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not is_distributed():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size()
    return float(tensor.item())


def broadcast_object(obj: Any) -> Any:
    if not is_distributed():
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]
