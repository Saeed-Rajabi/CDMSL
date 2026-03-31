from pathlib import Path
import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import SSHDataset
from models.conditional_unet import ConditionalUNet
from diffusion.diffusion_model import DiffusionModel
from diffusion.ema import EMA
from train import train_diffusion
from infer import run_inference
from utils_io import prepare_train_test_data, summarize_prepared_data


def infer_condition_channels(cond_inputs: torch.Tensor, use_lr_only: bool) -> int:
    """
    low res sea levels, pressure, zonal and meridional wind speeds.
    """
    if use_lr_only:
        return 1
    return cond_inputs.shape[1]


def build_dataset(
    prepared_split,
    use_lr_only: bool = True,
):
    """
    If use_lr_only=True:
        only LR upscaled SSH is used as condition.
    Otherwise:
        extra_conditions are passed separately and concatenated later in training.
    """
    if use_lr_only:
        return SSHDataset(
            hr_ssh=prepared_split.hr_ssh,
            lr_ssh=prepared_split.lr_ssh_upscaled,
            extra_conditions=None,
        )

    return SSHDataset(
        hr_ssh=prepared_split.hr_ssh,
        lr_ssh=prepared_split.lr_ssh_upscaled,
        extra_conditions=prepared_split.extra_conditions,
    )


def main():
    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    cfg = Config()
    save_dir: Path = cfg.make_dirs()

    print(f"Using device: {cfg.device}")
    print(f"Saving outputs to: {save_dir.resolve()}")

    # ------------------------------------------------------------------
    # 2. Data prep
    # ------------------------------------------------------------------
    prepared = prepare_train_test_data(
        train_data_path="train_SL.npy",
        train_conds_path="train_conds.npy",
        test_data_path="test_SL.npy",
        test_conds_path="test_conds.npy",
        scale=4,
        hr_index=-1,
        train_feature_slice=slice(2, -1),   
        test_feature_slice=None,            
        train_condition_channels=None,      
        test_condition_channels=None,
        fill_value=999.0,
    )

    train_split = prepared["train"]
    test_split = prepared["test"]

    summarize_prepared_data("train", train_split)
    summarize_prepared_data("test", test_split)

    # ------------------------------------------------------------------
    # 3. Choose conditioning strategy
    # ------------------------------------------------------------------

    use_lr_only = True

    cond_channels = infer_condition_channels(
        cond_inputs=train_split.cond_inputs,
        use_lr_only=use_lr_only,
    )

    print(f"\nConditioning mode: {'LR only' if use_lr_only else 'LR + extra conditions'}")
    print(f"Condition channels: {cond_channels}")

    # ------------------------------------------------------------------
    # 4. Build datasets and dataloader
    # ------------------------------------------------------------------
    train_dataset = build_dataset(train_split, use_lr_only=use_lr_only)
    test_dataset = build_dataset(test_split, use_lr_only=use_lr_only)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples : {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 5. Initialize model, diffusion, EMA
    # ------------------------------------------------------------------
    model = ConditionalUNet(cond_in_channels=cond_channels).to(cfg.device)
    diffusion = DiffusionModel(timesteps=cfg.timesteps).to(cfg.device)
    ema = EMA(model, decay=cfg.ema_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    train_diffusion(
        model=model,
        diffusion=diffusion,
        ema=ema,
        dataloader=train_loader,
        cfg=cfg,
        save_dir=save_dir,
    )

    # ------------------------------------------------------------------
    # 7. Inference / evaluation
    # ------------------------------------------------------------------
    metrics = run_inference(
        model=model,
        diffusion=diffusion,
        ema=ema,
        test_dataset=test_dataset,
        cfg=cfg,
        save_dir=save_dir,
    )

    # ------------------------------------------------------------------
    # 8. Print metrics
    # ------------------------------------------------------------------
    print("\n=== FINAL TEST METRICS ===")
    for key, values in metrics.items():
        if len(values) > 0:
            mean_val = sum(values) / len(values)
            print(f"{key}: {mean_val:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()