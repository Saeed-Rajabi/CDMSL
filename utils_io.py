from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PreparedData:
    hr_ssh: torch.Tensor
    lr_ssh: torch.Tensor
    lr_ssh_upscaled: torch.Tensor
    extra_conditions: Optional[torch.Tensor]
    cond_inputs: torch.Tensor


def load_npy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path)


def replace_nans(x: torch.Tensor, fill_value: float = 999.0) -> torch.Tensor:
    """Replace NaNs in a tensor."""
    return torch.nan_to_num(x, nan=fill_value)


def to_tensor(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.tensor(x, dtype=dtype)


def extract_hr_and_features(
    data: np.ndarray,
    conds: np.ndarray,
    hr_index: int = -1,
    feature_slice: Optional[slice] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract high-resolution SSH and external features.

    Parameters
    ----------
    data : np.ndarray
        Input sea-level array, expected shape [N, C, H, W] or similar.
    conds : np.ndarray
        Conditioning array.
    hr_index : int
        Channel index for the HR SSH target in `data`.
    feature_slice : slice or None
        Slice to extract external feature channels from `conds`.
        If None, all channels in `conds` are used.

    Returns
    -------
    ssh_high_res : np.ndarray
    external_features : np.ndarray or None
    """
    ssh_high_res = data[:, hr_index, :, :]
    external_features = conds[:, feature_slice, :, :] if feature_slice is not None else conds
    return ssh_high_res, external_features


def downscale_and_upscale_hr(
    ssh_high_res: np.ndarray | torch.Tensor,
    scale: int = 4,
    target_size: Optional[tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create LR SSH and bilinearly upscaled LR SSH from HR SSH.

    Parameters
    ----------
    ssh_high_res : np.ndarray or torch.Tensor
        HR SSH with shape [N, H, W]
    scale : int
        Downscaling factor
    target_size : tuple[int, int] or None
        Final upscaled target size (H, W). If None, original HR size is used.

    Returns
    -------
    ssh_hr : torch.Tensor
        [N, 1, H, W]
    ssh_lr : torch.Tensor
        [N, 1, H/scale, W/scale]
    ssh_lr_up : torch.Tensor
        [N, 1, H, W]
    """
    if isinstance(ssh_high_res, np.ndarray):
        ssh_hr = torch.tensor(ssh_high_res, dtype=torch.float32).unsqueeze(1)
    else:
        ssh_hr = ssh_high_res.float().unsqueeze(1) if ssh_high_res.ndim == 3 else ssh_high_res.float()

    h, w = ssh_hr.shape[-2], ssh_hr.shape[-1]
    low_res_size = (h // scale, w // scale)

    ssh_lr = F.interpolate(ssh_hr, size=low_res_size, mode="bilinear", align_corners=False)

    if target_size is None:
        target_size = (h, w)

    ssh_lr_up = F.interpolate(ssh_lr, size=target_size, mode="bilinear", align_corners=False)
    return ssh_hr, ssh_lr, ssh_lr_up


def build_condition_inputs(
    ssh_lr_upscaled: torch.Tensor,
    external_features: Optional[np.ndarray | torch.Tensor] = None,
    condition_channels: Optional[Sequence[int]] = None,
    fill_value: float = 999.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Build conditioning inputs for the model.

    Parameters
    ----------
    ssh_lr_upscaled : torch.Tensor
        [N, 1, H, W]
    external_features : np.ndarray or torch.Tensor or None
        Optional extra conditions
    condition_channels : sequence[int] or None
        Optional selected channels from final concatenated tensor
    fill_value : float
        Fill value for NaNs

    Returns
    -------
    extra_conditions : torch.Tensor or None
    cond_inputs : torch.Tensor
    """
    ssh_lr_upscaled = replace_nans(ssh_lr_upscaled, fill_value)

    if external_features is None:
        cond_inputs = ssh_lr_upscaled
        return None, cond_inputs

    if isinstance(external_features, np.ndarray):
        ext_feats = torch.tensor(external_features, dtype=torch.float32)
    else:
        ext_feats = external_features.float()

    ext_feats = replace_nans(ext_feats, fill_value)

    cond_inputs = torch.cat([ssh_lr_upscaled, ext_feats], dim=1)

    if condition_channels is not None:
        cond_inputs = cond_inputs[:, condition_channels, :, :]

    extra_conditions = cond_inputs[:, 1:, :, :] if cond_inputs.shape[1] > 1 else None
    return extra_conditions, cond_inputs


def prepare_single_split(
    data_path: str | Path,
    conds_path: str | Path,
    scale: int = 4,
    hr_index: int = -1,
    feature_slice: Optional[slice] = None,
    condition_channels: Optional[Sequence[int]] = None,
    fill_value: float = 999.0,
) -> PreparedData:
    """
    Prepare one split (train or test) from .npy files.
    """
    data = load_npy(data_path)
    conds = load_npy(conds_path)

    ssh_high_res, external_features = extract_hr_and_features(
        data=data,
        conds=conds,
        hr_index=hr_index,
        feature_slice=feature_slice,
    )

    ssh_hr, ssh_lr, ssh_lr_up = downscale_and_upscale_hr(
        ssh_high_res=ssh_high_res,
        scale=scale,
    )

    extra_conditions, cond_inputs = build_condition_inputs(
        ssh_lr_upscaled=ssh_lr_up,
        external_features=external_features,
        condition_channels=condition_channels,
        fill_value=fill_value,
    )

    ssh_hr = replace_nans(ssh_hr, fill_value)
    ssh_lr = replace_nans(ssh_lr, fill_value)
    ssh_lr_up = replace_nans(ssh_lr_up, fill_value)

    return PreparedData(
        hr_ssh=ssh_hr,
        lr_ssh=ssh_lr,
        lr_ssh_upscaled=ssh_lr_up,
        extra_conditions=extra_conditions,
        cond_inputs=cond_inputs,
    )


def prepare_train_test_data(
    train_data_path: str | Path = "train_SL.npy",
    train_conds_path: str | Path = "train_conds.npy",
    test_data_path: str | Path = "test_SL.npy",
    test_conds_path: str | Path = "test_conds.npy",
    scale: int = 4,
    hr_index: int = -1,
    train_feature_slice: Optional[slice] = slice(2, -1),
    test_feature_slice: Optional[slice] = None,
    train_condition_channels: Optional[Sequence[int]] = None,
    test_condition_channels: Optional[Sequence[int]] = None,
    fill_value: float = 999.0,
) -> Dict[str, PreparedData]:
    """
    Prepare both train and test splits.

    Notes
    -----
    Based on your original script:
    - train external features: conds[:, 2:-1, :, :]
    - test external features: conds2 (all channels)
    - later you sometimes selected channels like [1,2,3]
    """
    train = prepare_single_split(
        data_path=train_data_path,
        conds_path=train_conds_path,
        scale=scale,
        hr_index=hr_index,
        feature_slice=train_feature_slice,
        condition_channels=train_condition_channels,
        fill_value=fill_value,
    )

    test = prepare_single_split(
        data_path=test_data_path,
        conds_path=test_conds_path,
        scale=scale,
        hr_index=hr_index,
        feature_slice=test_feature_slice,
        condition_channels=test_condition_channels,
        fill_value=fill_value,
    )

    return {"train": train, "test": test}


def move_prepared_data_to_device(prepared: PreparedData, device: str) -> PreparedData:
    """
    Move prepared data tensors to device.
    """
    return PreparedData(
        hr_ssh=prepared.hr_ssh.float().to(device),
        lr_ssh=prepared.lr_ssh.float().to(device),
        lr_ssh_upscaled=prepared.lr_ssh_upscaled.float().to(device),
        extra_conditions=None if prepared.extra_conditions is None else prepared.extra_conditions.float().to(device),
        cond_inputs=prepared.cond_inputs.float().to(device),
    )


def summarize_prepared_data(name: str, prepared: PreparedData) -> None:
    """
    Print a brief summary of prepared tensors.
    """
    print(f"\n=== {name.upper()} DATA SUMMARY ===")
    print(f"HR SSH shape           : {tuple(prepared.hr_ssh.shape)}")
    print(f"LR SSH shape           : {tuple(prepared.lr_ssh.shape)}")
    print(f"LR upscaled shape      : {tuple(prepared.lr_ssh_upscaled.shape)}")
    if prepared.extra_conditions is not None:
        print(f"Extra conditions shape : {tuple(prepared.extra_conditions.shape)}")
    else:
        print("Extra conditions shape : None")
    print(f"Condition input shape  : {tuple(prepared.cond_inputs.shape)}")