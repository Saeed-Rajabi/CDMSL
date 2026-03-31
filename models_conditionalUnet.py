import torch
import torch.nn as nn
from .attention import AttentionBlock


class ConditionalUNet(nn.Module):
    def __init__(self, cond_in_channels: int = 1, base: int = 32):
        super().__init__()

        time_emb_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(1, base, 3, padding=1)
        self.cond_conv = nn.Conv2d(cond_in_channels, base, 3, padding=1)

        self.enc1 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(),
            AttentionBlock(base * 4),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(),
            AttentionBlock(base * 8),
            AttentionBlock(base * 8),
            nn.Conv2d(base * 8, base * 8, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, 3, padding=1),
            nn.ReLU(),
            AttentionBlock(base * 4),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        _ = self.time_mlp(t.float().unsqueeze(-1))  # currently unused, kept for future fusion

        x = self.init_conv(x)
        cond = self.cond_conv(conditions)
        x = torch.cat([x, cond], dim=1)

        skip1 = self.enc1(x)
        x = self.down1(skip1)

        skip2 = self.enc2(x)
        x = self.down2(skip2)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2(x)

        return self.final_conv(x)