from pathlib import Path
import torch
import torch.nn as nn

from utils.viz import save_epoch_visuals


def build_conditions(batch: dict, device: str) -> torch.Tensor:
    """
    Build conditioning tensor from a batch.

    If extra conditions are available, concatenate them with LR.
    Otherwise, use LR only.
    """
    lr = batch["lr"].float().to(device)

    if "conditions" in batch and batch["conditions"] is not None:
        extra = batch["conditions"].float().to(device)
        return torch.cat([lr, extra], dim=1)

    return lr


def train_diffusion(
    model,
    diffusion,
    ema,
    dataloader,
    cfg,
    save_dir: Path,
):
    """
    Train the conditional diffusion model on HR residuals.

    The model learns to denoise the residual:
        residual = HR - LR_upscaled
    """
    device = cfg.device
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    # epoch_vis_dir = save_dir / "epoch_visuals"
    # epoch_vis_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0


        eval_batch = next(iter(dataloader))
        eval_hr = eval_batch["hr"].float().to(device)
        eval_lr = eval_batch["lr"].float().to(device)
        eval_cond = build_conditions(eval_batch, device)

        for batch_idx, batch in enumerate(dataloader):
            hr = batch["hr"].float().to(device)
            lr = batch["lr"].float().to(device)
            conditions = build_conditions(batch, device)

            # Residual target
            target_residual = hr - lr

            # Random diffusion timestep
            t = torch.randint(
                low=0,
                high=diffusion.timesteps,
                size=(hr.shape[0],),
                device=device,
            )

            noisy_residual, noise = diffusion.add_noise(target_residual, t)
            predicted_noise = model(noisy_residual, t, conditions)

            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            epoch_loss += loss.item()

            if epoch == 0 and batch_idx == 0:
                print("\nTraining batch shapes:")
                print(f"  HR         : {tuple(hr.shape)}")
                print(f"  LR         : {tuple(lr.shape)}")
                print(f"  Conditions : {tuple(conditions.shape)}")
                print(f"  Residual   : {tuple(target_residual.shape)}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{cfg.epochs} | Loss: {avg_loss:.6f}")

        # Save intermediate visual checks
        if (epoch + 1) % cfg.train_vis_every == 0:
            model.eval()
            preds = []

            with torch.no_grad():
                for i in range(eval_hr.size(0)):
                    cond_i = eval_cond[i:i+1]
                    lr_i = eval_lr[i:i+1]

                    pred_residual = diffusion.sample(
                        model=model,
                        conditions=cond_i,
                        device=device,
                    )
                    pred_hr = pred_residual + lr_i
                    preds.append(pred_hr.cpu())

            preds = torch.cat(preds, dim=0)

            save_epoch_visuals(
                epoch=epoch + 1,
                eval_lr=eval_lr.cpu(),
                eval_hr=eval_hr.cpu(),
                predictions=preds,
                # out_dir=epoch_vis_dir,
                max_samples=min(4, eval_hr.size(0)),
            )

            # Save checkpoint
            ckpt_path = save_dir / f"model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)

    # Final saves
    # final_model_path = save_dir / "final_model.pth"
    # final_ema_path = save_dir / "final_model_ema.pth"

    # torch.save(model.state_dict(), final_model_path)
    # torch.save(ema.shadow, final_ema_path)

    # print(f"\nSaved final model weights to: {final_model_path}")
    # print(f"Saved EMA weights to        : {final_ema_path}")
