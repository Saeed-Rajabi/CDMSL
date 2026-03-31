from pathlib import Path
import numpy as np
import torch

from utils.metrics import (
    compute_rmse,
    compute_psnr,
    compute_rsquared,
    compute_spatial_corr,
)
from utils.viz import save_inference_panel


def build_conditions_from_sample(sample: dict, device: str) -> torch.Tensor:
    """
    Build conditioning tensor for a single sample.
    """
    lr = sample["lr"].unsqueeze(0).float().to(device)

    if "conditions" in sample and sample["conditions"] is not None:
        extra = sample["conditions"].unsqueeze(0).float().to(device)
        return torch.cat([lr, extra], dim=1)

    return lr


@torch.no_grad()
def run_inference(
    model,
    diffusion,
    ema,
    test_dataset,
    cfg,
    save_dir: Path,
):
    """
    Run stochastic inference on the test set using multiple diffusion samples.

    Returns
    -------
    dict
        Dictionary of per-sample metrics.
    """
    device = cfg.device
    model.eval()

    # Used EMA weights for inference
    ema.apply_shadow()

    samples_dir = save_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    metrics_all = {
        "RMSE": [],
        "PSNR": [],
        "R2": [],
        "SpatialCorr": [],
    }

    n_test = len(test_dataset)
    out_shape = test_dataset[0]["hr"].shape[-2:]
    saved_results = np.zeros((n_test, 2, *out_shape), dtype=np.float32)

    for i in range(n_test):
        sample = test_dataset[i]

        true_hr = sample["hr"].unsqueeze(0).float().to(device)  # [1,1,H,W] after unsqueeze below
        sample_lr = sample["lr"].unsqueeze(0).float().to(device)

        if true_hr.ndim == 3:
            true_hr = true_hr.unsqueeze(0)
        if sample_lr.ndim == 3:
            sample_lr = sample_lr.unsqueeze(0)

        conditions = build_conditions_from_sample(sample, device)

        generated_samples = []

        for _ in range(cfg.n_samples):
            pred_residual = diffusion.sample(
                model=model,
                conditions=conditions,
                device=device,
            )
            generated_samples.append(pred_residual.cpu())

        generated_stack = torch.stack(generated_samples, dim=0)   # [N,1,1,H,W]
        mean_pred = generated_stack.mean(dim=0)                   # [1,1,H,W]
        std_pred = generated_stack.std(dim=0)                     # [1,1,H,W]

        generated_hr = mean_pred.to(device) + sample_lr

        pred = generated_hr
        target = true_hr

        metrics = {
            "RMSE": compute_rmse(pred, target),
            "PSNR": compute_psnr(pred, target),
            "R2": compute_rsquared(pred, target),
            "SpatialCorr": compute_spatial_corr(pred, target),
        }

        for key, value in metrics.items():
            metrics_all[key].append(value)

        print(f"\nSample {i + 1}/{n_test}")
        print(f"  RMSE        : {metrics['RMSE']:.4f}")
        print(f"  PSNR        : {metrics['PSNR']:.4f}")
        print(f"  R2          : {metrics['R2']:.4f}")
        print(f"  SpatialCorr : {metrics['SpatialCorr']:.4f}")

        saved_results[i, 0] = true_hr[0, 0].cpu().numpy()
        saved_results[i, 1] = generated_hr[0, 0].cpu().numpy()

        save_inference_panel(
            sample_idx=i,
            sample_lr=sample_lr[0, 0].cpu(),
            true_hr=true_hr[0, 0].cpu(),
            generated_hr=generated_hr[0, 0].cpu(),
            mean_pred=mean_pred[0, 0].cpu(),
            std_pred=std_pred[0, 0].cpu(),
            save_path=samples_dir / f"sample_{i + 1}.png",
        )

        np.save(
            samples_dir / f"uncertainty_std_sample_{i + 1}.npy",
            std_pred[0, 0].cpu().numpy(),
        )

    np.save(save_dir / "saved_results_test.npy", saved_results)

    print("\n=== TEST METRIC SUMMARY ===")
    for key, values in metrics_all.items():
        print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    ema.restore()
    return metrics_all
