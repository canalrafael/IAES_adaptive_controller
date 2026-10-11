"""
Vision Transformer with Fine-Grained (Head & Neuron Width) Splitting for SPViT
==============================================================================
Pure PyTorch implementation (zero external einops dependency) supporting:
- Intra-layer Head-Width splitting in Multi-Head Self-Attention (MSA)
- Intra-layer Neuron-Width splitting in Feed-Forward Networks (FFN / FC)
- Full compatibility with Raspberry Pi ARM64 and PC environments
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

class PartitionConfig:
    """Represents a fine-grained head and neuron partition configuration."""

    def __init__(
        self,
        heads_edge: int = 3,
        heads_server: int = 3,
        neurons_edge: int = 768,
        neurons_server: int = 768,
        split_block_idx: int = 0,
    ):
        self.heads_edge = heads_edge
        self.heads_server = heads_server
        self.neurons_edge = neurons_edge
        self.neurons_server = neurons_server
        self.split_block_idx = split_block_idx

    def to_dict(self):
        return {
            "heads_edge": self.heads_edge,
            "heads_server": self.heads_server,
            "neurons_edge": self.neurons_edge,
            "neurons_server": self.neurons_server,
            "split_block_idx": self.split_block_idx,
        }


class FeedForward(nn.Module):
    """
    MLP / Feed-Forward Block with Neuron-Width Partitioning Support.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full forward execution."""
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return res + x

    def forward_edge(self, x: torch.Tensor, neurons_edge: int) -> torch.Tensor:
        """
        Executes edge portion of neuron-width split.
        Projects D -> neurons_edge -> D.
        """
        if neurons_edge <= 0:
            return torch.zeros_like(x)

        x_norm = self.norm(x)
        w1_edge = self.fc1.weight[:neurons_edge, :]
        b1_edge = self.fc1.bias[:neurons_edge] if self.fc1.bias is not None else None

        h = torch.matmul(x_norm, w1_edge.t())
        if b1_edge is not None:
            h = h + b1_edge
        h = self.act(h)

        w2_edge = self.fc2.weight[:, :neurons_edge]
        out = torch.matmul(h, w2_edge.t())
        return out

    def forward_server(self, x: torch.Tensor, neurons_edge: int) -> torch.Tensor:
        """
        Executes server portion of neuron-width split for neurons from neurons_edge to hidden_dim.
        """
        if neurons_edge >= self.hidden_dim:
            return torch.zeros_like(x)

        x_norm = self.norm(x)
        w1_serv = self.fc1.weight[neurons_edge:, :]
        b1_serv = self.fc1.bias[neurons_edge:] if self.fc1.bias is not None else None

        h = torch.matmul(x_norm, w1_serv.t())
        if b1_serv is not None:
            h = h + b1_serv
        h = self.act(h)

        w2_serv = self.fc2.weight[:, neurons_edge:]
        out = torch.matmul(h, w2_serv.t())
        if self.fc2.bias is not None:
            out = out + self.fc2.bias
        return out


class Attention(nn.Module):
    """
    Multi-Head Attention with Head-Width Partitioning Support.
    """

    def __init__(self, dim: int, heads: int = 6, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def _split_qkv(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x_norm.shape
        qkv = self.to_qkv(x_norm)  # (b, n, 3 * inner_dim)
        qkv = qkv.view(b, n, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]  # each is (b, h, n, d)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full forward execution."""
        res = x
        x_norm = self.norm(x)
        q, k, v = self._split_qkv(x_norm)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)  # (b, h, n, d)

        b, h, n, d = out.shape
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        return res + self.to_out(out)

    def compute_heads_edge(self, x: torch.Tensor, heads_edge: int) -> torch.Tensor:
        """
        Edge forward for the first heads_edge attention heads.
        Returns intermediate attended features of shape (B, N, heads_edge * dim_head).
        """
        b, n, _ = x.shape
        if heads_edge <= 0:
            return torch.zeros(b, n, 0, device=x.device, dtype=x.dtype)

        x_norm = self.norm(x)
        q, k, v = self._split_qkv(x_norm)

        q_e = q[:, :heads_edge]
        k_e = k[:, :heads_edge]
        v_e = v[:, :heads_edge]

        dots = torch.matmul(q_e, k_e.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out_e = torch.matmul(attn, v_e)
        return out_e.permute(0, 2, 1, 3).contiguous().view(b, n, heads_edge * self.dim_head)

    def compute_heads_server(self, x: torch.Tensor, heads_edge: int) -> torch.Tensor:
        """
        Server forward for remaining attention heads (heads_edge to total heads).
        Returns intermediate attended features of shape (B, N, (heads - heads_edge) * dim_head).
        """
        b, n, _ = x.shape
        if heads_edge >= self.heads:
            return torch.zeros(b, n, 0, device=x.device, dtype=x.dtype)

        heads_serv = self.heads - heads_edge
        x_norm = self.norm(x)
        q, k, v = self._split_qkv(x_norm)

        q_s = q[:, heads_edge:]
        k_s = k[:, heads_edge:]
        v_s = v[:, heads_edge:]

        dots = torch.matmul(q_s, k_s.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out_s = torch.matmul(attn, v_s)
        return out_s.permute(0, 2, 1, 3).contiguous().view(b, n, heads_serv * self.dim_head)

    def combine_heads_and_project(
        self, x_orig: torch.Tensor, out_heads_edge: torch.Tensor, out_heads_server: torch.Tensor
    ) -> torch.Tensor:
        """Merges head outputs from edge and server and applies final linear projection."""
        merged = torch.cat([out_heads_edge, out_heads_server], dim=-1)
        return x_orig + self.to_out(merged)


class TransformerBlock(nn.Module):
    """Single Transformer Encoder Block with Fine-Grained Split Support."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn.forward_full(x)
        x = self.ff.forward_full(x)
        return x

    def forward_split(self, x: torch.Tensor, partition: PartitionConfig) -> torch.Tensor:
        """Executes one block using collaborative Head & Neuron split."""
        out_edge_attn = self.attn.compute_heads_edge(x, partition.heads_edge)
        out_serv_attn = self.attn.compute_heads_server(x, partition.heads_edge)
        x_attn = self.attn.combine_heads_and_project(x, out_edge_attn, out_serv_attn)

        out_edge_ff = self.ff.forward_edge(x_attn, partition.neurons_edge)
        out_serv_ff = self.ff.forward_server(x_attn, partition.neurons_edge)
        x_out = x_attn + (out_edge_ff + out_serv_ff)
        return x_out


class ViT_split(nn.Module):
    """
    Partitionable Vision Transformer (ViT) implementation for SPViT.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 10,
        dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_dim: int = 1536,
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim

        # Standard Patch Embedding using Conv2d
        self.patch_embed = nn.Conv2d(
            channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_norm = nn.LayerNorm(dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def prepare_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """Converts input image to patched tokens with [CLS] and position embedding."""
        b = img.shape[0]
        # (b, c, h, w) -> (b, dim, gh, gw) -> (b, dim, n) -> (b, n, dim)
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        x = self.patch_norm(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : x.shape[1]]
        return self.dropout(x)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Full standalone execution."""
        x = self.prepare_tokens(img)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)

    def forward_with_partition(
        self, img: torch.Tensor, partition: PartitionConfig
    ) -> torch.Tensor:
        """Executes forward pass using fine-grained SPViT partition parameters."""
        x = self.prepare_tokens(img)
        for block in self.blocks:
            x = block.forward_split(x, partition)
        x = self.norm(x)
        return self.mlp_head(x[:, 0])
