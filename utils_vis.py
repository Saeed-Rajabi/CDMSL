from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_epoch_visuals(epoch, eval_lr, eval_hr, predictions, out_dir: Path, max_samples: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(max_samples, eval_hr.size(0))):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        lr_img = eval_lr[i, 0].cpu()
        hr_img = eval_hr[i, 0].cpu()
        pred_img = predictions[i, 0].cpu()

        true_residual = hr_img - lr_img
        pred_residual = pred_img - lr_img

        vmin = true_residual.min().item()
        vmax = true_residual.max().item()

        axes[0, 0].imshow(lr_img, cmap="viridis")
        axes[0, 0].set_title("LR Input")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(hr_img, cmap="viridis")
        axes[0, 1].set_title("True HR")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(pred_img, cmap="viridis")
        axes[0, 2].set_title("Predicted HR")
        axes[0, 2].axis("off")

        im = axes[1, 0].imshow(true_residual, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("True Residual")
        fig.colorbar(im, ax=axes[1, 0])

        im = axes[1, 1].imshow(pred_residual, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, 1].set_title("Predicted Residual")
        fig.colorbar(im, ax=axes[1, 1])

        axes[1, 2].axis("off")
        plt.suptitle(f"Epoch {epoch} - Sample {i+1}")
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{i+1}_epoch_{epoch}.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_inference_panel(sample_idx, sample_lr, true_hr, generated_hr, mean_pred, std_pred, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    vmin = true_hr.min().item()
    vmax = true_hr.max().item()

    true_residual = true_hr - sample_lr
    vmin_r = true_residual.min().item()
    vmax_r = true_residual.max().item()

    im = axes[0, 0].imshow(sample_lr.cpu(), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("LR Input")
    fig.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(true_hr.cpu(), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("True HR")
    fig.colorbar(im, ax=axes[0, 1])

    im = axes[0, 2].imshow(generated_hr.cpu(), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Generated HR")
    fig.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].imshow(true_residual.cpu(), cmap="viridis", vmin=vmin_r, vmax=vmax_r)
    axes[1, 0].set_title("True Residual")
    fig.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(mean_pred.cpu(), cmap="viridis", vmin=vmin_r, vmax=vmax_r)
    axes[1, 1].set_title("Mean Predicted Residual")
    fig.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].imshow(std_pred.cpu(), cmap="magma", vmin=0, vmax=5)
    axes[1, 2].set_title("Uncertainty (std)")
    fig.colorbar(im, ax=axes[1, 2])

    plt.suptitle(f"Sample {sample_idx + 1}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()