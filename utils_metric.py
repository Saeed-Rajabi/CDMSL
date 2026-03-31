import torch
import torch.nn.functional as F


def compute_rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()


def compute_psnr(pred, target, max_val=120):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device) / torch.sqrt(mse)).item()


def compute_rsquared(pred, target):
    target_mean = torch.mean(target)
    ss_total = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    return (1 - ss_res / ss_total).item()


def compute_spatial_corr(pred, target):
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    corr = F.cosine_similarity(pred_flat, target_flat, dim=1)
    return corr.mean().item()