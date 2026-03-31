from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class Config:
    epochs: int = 700
    batch_size: int = 16
    n_samples: int = 50
    lr: float = 1e-4
    timesteps: int = 1000
    ema_decay: float = 0.85
    cond_channels: int = 1
    num_workers: int = 0

    output_dir: str = "outputs"
    train_vis_every: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def make_dirs(self) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "epoch_visuals").mkdir(exist_ok=True)
        (out / "samples").mkdir(exist_ok=True)
        return out