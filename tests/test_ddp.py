"""Tests for ssr_gcn.ddp — runs in single-process (no real distributed setup)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import HAS_TORCH

if not HAS_TORCH:
    pytest.skip("torch unavailable on this machine", allow_module_level=True)

import torch  # noqa: E402

from ssr_gcn.ddp import (
    all_reduce_mean,
    barrier,
    broadcast_object,
    cleanup_distributed,
    is_distributed,
    is_rank0,
    local_rank,
    rank,
    seed_everything,
    setup_distributed,
    world_size,
)


# ---------------------------------------------------------------------------
# Non-distributed helpers (single process)
# ---------------------------------------------------------------------------

class TestNonDistributedHelpers:
    def test_is_distributed_false(self):
        assert not is_distributed()

    def test_rank_is_zero(self):
        assert rank() == 0

    def test_world_size_is_one(self):
        assert world_size() == 1

    def test_is_rank0_true(self):
        assert is_rank0()

    def test_local_rank_default_zero(self):
        os.environ.pop("LOCAL_RANK", None)
        assert local_rank() == 0

    def test_local_rank_from_env(self):
        os.environ["LOCAL_RANK"] = "1"
        assert local_rank() == 1
        os.environ.pop("LOCAL_RANK")

    def test_all_reduce_mean_passthrough(self):
        val = 3.14159
        result = all_reduce_mean(val, torch.device("cpu"))
        assert result == pytest.approx(val)

    def test_broadcast_object_passthrough(self):
        obj = {"key": [1, 2, 3], "nested": {"a": "b"}}
        result = broadcast_object(obj)
        assert result == obj

    def test_barrier_noop(self):
        barrier()  # must not raise

    def test_cleanup_noop(self):
        cleanup_distributed()  # must not raise


# ---------------------------------------------------------------------------
# setup_distributed — single process (no RANK env var)
# ---------------------------------------------------------------------------

class TestSetupDistributed:
    def test_returns_device_when_no_rank_env(self):
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)
        device = setup_distributed()
        assert isinstance(device, torch.device)

    def test_nccl_env_vars_not_set_without_rank(self):
        """NCCL env vars must only be set when RANK is present (torchrun context)."""
        os.environ.pop("RANK", None)
        for key in ("NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME"):
            os.environ.pop(key, None)
        setup_distributed()
        assert "NCCL_P2P_DISABLE" not in os.environ
        assert "NCCL_IB_DISABLE" not in os.environ
        assert "NCCL_SOCKET_IFNAME" not in os.environ

    def test_nccl_env_vars_set_when_rank_present(self):
        """When RANK and WORLD_SIZE are present, NCCL env vars should be set."""
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        for key in ("NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME"):
            os.environ.pop(key, None)

        # Mock dist.init_process_group and dist.is_initialized to avoid real DDP init
        with mock.patch("torch.distributed.is_initialized", return_value=True), \
             mock.patch("torch.distributed.init_process_group"):
            setup_distributed()

        assert os.environ.get("NCCL_P2P_DISABLE") == "1"
        assert os.environ.get("NCCL_IB_DISABLE") == "1"
        assert "NCCL_SOCKET_IFNAME" in os.environ

        for key in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME"):
            os.environ.pop(key, None)

    def test_nccl_env_vars_not_overridden_if_already_set(self):
        """Existing NCCL env vars must not be overridden (uses setdefault)."""
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "2"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        os.environ["NCCL_P2P_DISABLE"] = "0"  # user chose to keep P2P

        with mock.patch("torch.distributed.is_initialized", return_value=True), \
             mock.patch("torch.distributed.init_process_group"):
            setup_distributed()

        assert os.environ["NCCL_P2P_DISABLE"] == "0"  # preserved

        for key in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME"):
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# seed_everything — reproducibility
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_same_seed_same_random_values(self):
        import random
        import numpy as np

        seed_everything(42)
        r1 = random.random()
        n1 = float(np.random.rand())
        t1 = torch.rand(1).item()

        seed_everything(42)
        r2 = random.random()
        n2 = float(np.random.rand())
        t2 = torch.rand(1).item()

        assert r1 == r2
        assert n1 == n2
        assert t1 == t2

    def test_different_seeds_different_values(self):
        seed_everything(1)
        t1 = torch.rand(1).item()
        seed_everything(2)
        t2 = torch.rand(1).item()
        assert t1 != t2
