"""Lightweight spatio-temporal GCN for 13->25 SSR."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ssr_gcn.constants import NTU_EDGES


def build_adjacency(num_joints: int = 25) -> torch.Tensor:
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    for src, dst in NTU_EDGES:
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


class GraphConv(nn.Module):
    """Simple graph propagation followed by a pointwise convolution."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("nctv,vw->nctw", x, self.adjacency)
        return self.proj(x)


class STGCNBlock(nn.Module):
    """Residual spatial-temporal block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        kernel_size: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.gcn = GraphConv(in_channels, out_channels, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if in_channels == out_channels:
            self.residual: nn.Module = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.act(x + residual)


class SSRGCN(nn.Module):
    """Encoder-decoder style ST-GCN with learned joint lifting."""

    def __init__(
        self,
        hidden_channels: int = 96,
        num_blocks: int = 6,
        temporal_kernel: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        adjacency = build_adjacency()
        self.register_buffer("adjacency", adjacency)

        self.joint_lift = nn.Linear(13, 25)
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        channels = [hidden_channels] * max(num_blocks, 2)
        blocks: list[nn.Module] = []
        for idx, out_channels in enumerate(channels):
            in_channels = hidden_channels if idx == 0 else channels[idx - 1]
            blocks.append(
                STGCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    adjacency=adjacency,
                    kernel_size=temporal_kernel,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

    def _lift_joints(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_joints, dims = x.shape
        x = x.permute(0, 1, 3, 2).reshape(batch_size * time_steps * dims, num_joints)
        x = self.joint_lift(x)
        return x.view(batch_size, time_steps, dims, 25).permute(0, 2, 1, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lifted = self._lift_joints(x)
        features = self.input_proj(lifted)
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            features = block(features)
            if idx < len(self.blocks) // 2:
                skips.append(features)
            elif skips:
                features = features + skips.pop()
        output = self.decoder(features) + lifted
        return output.permute(0, 2, 3, 1).contiguous()


def create_model(cfg: dict[str, Any]) -> SSRGCN:
    model_cfg = cfg.get("model") or {}
    return SSRGCN(
        hidden_channels=int(model_cfg.get("hidden_channels", 96)),
        num_blocks=int(model_cfg.get("num_blocks", 6)),
        temporal_kernel=int(model_cfg.get("temporal_kernel", 9)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
