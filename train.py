from pathlib import Path
import torch
import torch.nn as nn
from utils.viz import save_epoch_visuals


def build_conditions(batch):
    lr = batch["lr"].float()
    if "conditions" in batch:
        extra = batch["conditions"].float()
        return torch.cat([lr, extra], dim=1)
    return lr


def train_diffusion(model, diffusion, ema, dataloader, cfg, save_dir: Path):
    device = cfg.device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    epoch_vis_dir = save_dir / "epoch_visuals"

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        eval_batch = next(iter(dataloader))
        eval_hr = eval_batch["hr"].float().to(device)
        eval_lr = eval_batch["lr"].float().to(device)
        eval_cond = build_conditions(eval_batch).float().to(device)

        for batch in dataloader:
            hr = batch["hr"].float().to(device)
            lr = batch["lr"].float().to(device)
            conditions = build_conditions(batch).float().to(device)

            target_residual = hr - lr
            t = torch.randint(0, diffusion.timesteps, (hr.shape[0],), device=device)

            noisy_residual, noise = diffusion.add_noise(target_residual, t)
            predicted_noise = model(noisy_residual, t, conditions)

            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % cfg.train_vis_every == 0:
            preds = []
            with torch.no_grad():
                for i in range(eval_hr.size(0)):
                    cond = eval_cond[i:i+1]
                    residual = diffusion.sample(model, cond, device=device)
                    pred_hr = residual + eval_lr[i:i+1]
                    preds.append(pred_hr.cpu())
            preds = torch.cat(preds, dim=0)
            save_epoch_visuals(epoch + 1, eval_lr.cpu(), eval_hr.cpu(), preds, epoch_vis_dir)

    torch.save(model.state_dict(), save_dir / "final_model.pth")
    torch.save(ema.shadow, save_dir / "final_model_ema.pth")
    print("Training complete.")