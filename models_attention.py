import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_seq = x.view(b, c, -1).transpose(1, 2)  # [B, HW, C]
        x_ln = self.ln(x_seq)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        x_seq = x_seq + attn
        x_seq = x_seq + self.ff(x_seq)
        return x_seq.transpose(1, 2).view(b, c, h, w)