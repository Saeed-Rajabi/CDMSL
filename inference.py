from pathlib import Path
import numpy as np
import torch
from utils.metrics import compute_rmse, compute_psnr, compute_rsquared, compute_spatial_corr
from utils.viz import save_inference_panel


@torch.no_grad()
def run_inference(model, diffusion, ema, test_dataset, cfg, save_dir: Path):
    device = cfg.device
    ema.apply_shadow()
    model.eval()

    metrics_all = {"RMSE": [], "PSNR": [], "R2": [], "SpatialCorr": []}
    samples_dir = save_dir / "samples"

    n = len(test_dataset)
    example_shape = test_dataset[0]["hr"].shape[-2:]
    saved_results = np.zeros((n, 2, *example_shape), dtype=np.float32)

    for i in range(n):
        sample = test_dataset[i]
        true_hr = sample["hr"].unsqueeze(0).float().to(device)
        sample_lr = sample["lr"].unsqueeze(0).float().to(device)

        if "conditions" in sample:
            cond = torch.cat([sample["lr"].unsqueeze(0), sample["conditions"].unsqueeze(0)], dim=1).float().to(device)
        else:
            cond = sample_lr

        generated_samples = []
        for _ in range(cfg.n_samples):
            residual = diffusion.sample(model, cond, device=device)
            generated_samples.append(residual.cpu())

        stack = torch.stack(generated_samples, dim=0)
        mean_pred = stack.mean(dim=0)
        std_pred = stack.std(dim=0)

        generated_hr = mean_pred.to(device) + sample_lr

        pred = generated_hr
        target = true_hr

        metrics = {
            "RMSE": compute_rmse(pred, target),
            "PSNR": compute_psnr(pred, target),
            "R2": compute_rsquared(pred, target),
            "SpatialCorr": compute_spatial_corr(pred, target),
        }

        for k, v in metrics.items():
            metrics_all[k].append(v)

        print(f"Sample {i+1}: {metrics}")

        saved_results[i, 0] = true_hr[0, 0].cpu().numpy()
        saved_results[i, 1] = generated_hr[0, 0].cpu().numpy()

        save_inference_panel(
            i,
            sample_lr[0, 0].cpu(),
            true_hr[0, 0].cpu(),
            generated_hr[0, 0].cpu(),
            mean_pred[0, 0],
            std_pred[0, 0],
            samples_dir / f"sample_{i+1}.png",
        )

        np.save(samples_dir / f"uncertainty_std_sample_{i+1}.npy", std_pred[0, 0].numpy())

    np.save(save_dir / "saved_results_test.npy", saved_results)

    for k, vals in metrics_all.items():
        print(f"{k} mean: {np.mean(vals):.4f}")

    ema.restore()
    return metrics_all