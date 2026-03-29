"""Tests for ssr_gcn.model — requires torch."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import HAS_TORCH

if not HAS_TORCH:
    pytest.skip("torch unavailable on this machine", allow_module_level=True)

import torch  # noqa: E402

from ssr_gcn.model import SSRGCN, GraphConv, STGCNBlock, build_adjacency, create_model


# ---------------------------------------------------------------------------
# build_adjacency
# ---------------------------------------------------------------------------

class TestBuildAdjacency:
    def test_shape(self):
        adj = build_adjacency(25)
        assert adj.shape == (25, 25)

    def test_row_normalized_not_symmetric(self):
        """
        build_adjacency uses row normalization (D⁻¹·A): each row i is divided by
        degree(i).  Because different joints have different degrees, A[i,j] ≠ A[j,i]
        in general — the matrix is intentionally asymmetric.
        Symmetric normalization (D⁻¹ᐟ²·A·D⁻¹ᐟ²) would be a different design choice.
        """
        adj = build_adjacency(25)
        # Row sums of the raw adjacency before normalization vary per joint,
        # so the normalized result is generally not equal to its transpose.
        assert not torch.allclose(adj, adj.T, atol=1e-6)

    def test_nonnegative(self):
        adj = build_adjacency(25)
        assert (adj >= 0).all()

    def test_diagonal_positive(self):
        # Each joint is connected to itself (identity added before normalization)
        adj = build_adjacency(25)
        assert (adj.diagonal() > 0).all()

    def test_connected_joints_nonzero(self):
        from ssr_gcn.constants import NTU_EDGES
        adj = build_adjacency(25)
        for src, dst in NTU_EDGES:
            assert adj[src, dst] > 0, f"Edge ({src},{dst}) should be nonzero"
            assert adj[dst, src] > 0, f"Edge ({dst},{src}) should be nonzero"


# ---------------------------------------------------------------------------
# SSRGCN forward pass
# ---------------------------------------------------------------------------

class TestSSRGCN:
    @pytest.fixture
    def small_model(self):
        model = SSRGCN(hidden_channels=16, num_blocks=2, temporal_kernel=3, dropout=0.0)
        model.eval()
        return model

    def test_output_shape(self, small_model):
        x = torch.randn(2, 10, 13, 3)
        with torch.no_grad():
            out = small_model(x)
        assert out.shape == (2, 10, 25, 3)

    def test_batch_size_one(self, small_model):
        x = torch.randn(1, 5, 13, 3)
        with torch.no_grad():
            out = small_model(x)
        assert out.shape == (1, 5, 25, 3)

    def test_output_dtype_float32(self, small_model):
        x = torch.randn(1, 10, 13, 3)
        with torch.no_grad():
            out = small_model(x)
        assert out.dtype == torch.float32

    def test_no_nan_or_inf(self, small_model):
        x = torch.randn(2, 15, 13, 3)
        with torch.no_grad():
            out = small_model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_25_joints(self, small_model):
        x = torch.randn(1, 10, 13, 3)
        with torch.no_grad():
            out = small_model(x)
        assert out.shape[2] == 25

    def test_deterministic_in_eval_mode(self, small_model):
        x = torch.randn(1, 10, 13, 3)
        with torch.no_grad():
            out1 = small_model(x)
            out2 = small_model(x)
        assert torch.allclose(out1, out2)

    def test_longer_sequence(self, small_model):
        x = torch.randn(1, 150, 13, 3)
        with torch.no_grad():
            out = small_model(x)
        assert out.shape == (1, 150, 25, 3)

    def test_residual_connection_different_output_than_input(self, small_model):
        # The decoder adds the lifted input, output shouldn't be exactly zero
        x = torch.ones(1, 5, 13, 3)
        with torch.no_grad():
            out = small_model(x)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_different_configs(self):
        for hidden, blocks in [(8, 1), (32, 4), (64, 6)]:
            model = SSRGCN(hidden_channels=hidden, num_blocks=blocks, temporal_kernel=3)
            model.eval()
            x = torch.randn(1, 8, 13, 3)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 8, 25, 3), f"Failed for hidden={hidden}, blocks={blocks}"


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------

class TestCreateModel:
    def test_returns_ssrgcn(self):
        model = create_model({})
        assert isinstance(model, SSRGCN)

    def test_config_applied(self):
        cfg = {"model": {"hidden_channels": 32, "num_blocks": 3, "temporal_kernel": 5, "dropout": 0.0}}
        model = create_model(cfg)
        assert isinstance(model, SSRGCN)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 10, 13, 3))
        assert out.shape == (1, 10, 25, 3)

    def test_empty_model_cfg(self):
        model = create_model({"model": {}})
        assert isinstance(model, SSRGCN)
