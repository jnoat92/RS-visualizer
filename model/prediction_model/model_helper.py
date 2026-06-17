"""
Model loading and inference utilities for RS-visualizer.

Last modified: Jun 2026
"""

import argparse
import functools

import numpy as np
from model.prediction_model.DL.unet.unet_model import UNet
import torch
from torch import nn
from tqdm import tqdm
import os

#%% Normalization
def Normalize_min_max(
    img: np.ndarray,
    valid_mask: np.ndarray,
    feature_range=(-1.0, 1.0),
    eps: float = 1e-12,
):
    """
    Min–max normalize an image using only valid pixels.

    Parameters
    ----------
    img : np.ndarray
        Input image, shape (H,W) or (H,W,C).
    valid_mask : np.ndarray
        Boolean mask, shape (H,W), True = valid pixel.
    feature_range : tuple
        Target range (min, max), e.g. (-1, 1).
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    img_norm : np.ndarray
        Normalized image, same shape as img, float32.
    vmin : np.ndarray
        Per-channel min used for normalization, shape (C,) or scalar for (H,W).
    vmax : np.ndarray
        Per-channel max used for normalization, shape (C,) or scalar for (H,W).
    """
    assert img.ndim in (2, 3), "img must be (H,W) or (H,W,C)"
    assert valid_mask.shape == img.shape[:2]

    lo, hi = feature_range
    mid = 0.5 * (lo + hi)   # value for invalid pixels

    img = img.astype(np.float32, copy=False)

    # Normalize to (H,W,C) internally
    if img.ndim == 2:
        img_ = img[..., None]
        squeeze = True
    else:
        img_ = img
        squeeze = False

    H, W, C = img_.shape
    img_norm = np.empty_like(img_, dtype=np.float32)

    vmin = np.empty(C, dtype=np.float32)
    vmax = np.empty(C, dtype=np.float32)

    for c in range(C):
        vals = img_[..., c][valid_mask]

        if vals.size == 0:
            # Degenerate case: no valid pixels
            vmin[c] = 0.0
            vmax[c] = 1.0
            img_norm[..., c] = mid
            continue

        vmin[c] = vals.min()
        vmax[c] = vals.max()

        scale = (hi - lo) / (vmax[c] - vmin[c] + eps)

        img_norm[..., c] = (img_[..., c] - vmin[c]) * scale + lo
        img_norm[..., c][~valid_mask] = mid

    if squeeze:
        return img_norm[..., 0]
    else:
        return img_norm
    
def Normalize_mean_std(
    img: np.ndarray,
    mean=(-16.849971303873264, -27.80859960890284),
    std=(6.601613867039019, 6.390188051517371),
    eps: float = 1e-12,
):
    """
    Standardize image using external mean/std.


    - NaNs are replaced by mean (=> become 0 after normalization)
    - Fully vectorized
    """
    assert img.ndim in (2, 3), "img must be (H,W) or (H,W,C)"
    img = img.astype(np.float32, copy=True)
    squeeze = (img.ndim == 2)
    img_ = img[..., None] if squeeze else img
    H, W, C = img_.shape
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    assert mean.shape == (C,), f"mean must have shape ({C},)"
    assert std.shape == (C,), f"std must have shape ({C},)"
    assert np.all(std > 0), "std must be > 0"
    # Replace NaNs with mean (so normalized value becomes 0)
    img_ = np.where(
        np.isnan(img_),
        mean[None, None, :],
        img_
    )
    # Normalize
    img_norm = (img_ - mean[None, None, :]) / (std[None, None, :] + eps)
    return img_norm[..., 0] if squeeze else img_norm
    

def bn_calibrate_from_image(
    model,
    img_norm,              # (1,C,H,W)
    valid_mask,            # (H,W) bool
    steps=200,
    batch_size=8,
    patch_size=512,
    device="cuda",
    min_valid_frac=0.8,
    bn_momentum=0.01,
):
    orig_device = next(model.parameters()).device
    was_training = model.training

    if img_norm.dim() == 3:
        img_norm = img_norm.unsqueeze(0)

    _, C, H, W = img_norm.shape
    ps = min(int(patch_size), H, W)

    model = model.to(device)
    img_norm = img_norm.to(device, non_blocking=True)
    valid_mask = torch.Tensor(valid_mask).to(device)

    model.train()

    # disable dropout
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.eval()

    # temporarily speed up BN adaptation
    bn_layers, old_moms = [], []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(m)
            old_moms.append(m.momentum)
            m.momentum = bn_momentum
            m.track_running_stats = True

    with torch.no_grad():
        # for _ in range(int(steps)):
        for _ in tqdm(range(int(steps)), desc=f"BN calib ({device})"):
            patches = torch.empty((batch_size, C, ps, ps),
                                  device=device,
                                  dtype=img_norm.dtype)

            max_tries = 2000  # per batch (tune as needed)
            filled = 0
            tries = 0
            best = []  # store (valid_frac, y, x)

            while filled < batch_size and tries < max_tries:
                y = int(torch.randint(0, H - ps + 1, (1,), device=device).item())
                x = int(torch.randint(0, W - ps + 1, (1,), device=device).item())

                vm = valid_mask[y:y+ps, x:x+ps]
                valid_frac = float(vm.float().mean().item())

                # keep a few best candidates as fallback
                if len(best) < batch_size:
                    best.append((valid_frac, y, x))
                    best.sort(reverse=True, key=lambda t: t[0])
                elif valid_frac > best[-1][0]:
                    best[-1] = (valid_frac, y, x)
                    best.sort(reverse=True, key=lambda t: t[0])

                if valid_frac >= min_valid_frac:
                    patches[filled] = img_norm[0, :, y:y+ps, x:x+ps]
                    filled += 1

                tries += 1

            # Fallback: if we didn't fill the batch, use the best patches we saw
            if filled < batch_size:
                # (optional) print once per step if you want to know it happened
                # print(f"Warning: only filled {filled}/{batch_size} with min_valid_frac={min_valid_frac}. Using best-effort patches.")

                for k in range(filled, batch_size):
                    # If best is empty (shouldn't happen), just pick random
                    if best:
                        _, y, x = best[k - filled] if (k - filled) < len(best) else best[-1]
                        patches[k] = img_norm[0, :, y:y+ps, x:x+ps]
                    else:
                        y = int(torch.randint(0, H - ps + 1, (1,), device=device).item())
                        x = int(torch.randint(0, W - ps + 1, (1,), device=device).item())
                        patches[k] = img_norm[0, :, y:y+ps, x:x+ps]

            _ = model(patches)

    # restore BN momentum
    for m, mom in zip(bn_layers, old_moms):
        m.momentum = mom

    model.train(was_training)
    model = model.to(orig_device)

    return model

# def load_model(model_path, img_norm, valid_mask, device='cpu'):
#     """
#     Load a pretrained UNet model from a checkpoint.
#     Parameters
#     ----------
#     model_path : str
#         Path to the model checkpoint file.
#     device : str, optional
#         Device to load the model on ('cpu' or 'cuda'). Default is 'cpu'.
#     Returns
#     -------
#     model : UNet
#         Loaded UNet model with weights from checkpoint.
#     Raises
#     ------
#     AssertionError
#         If the checkpoint file does not exist at model_path.
#     """
#     model = UNet(2, 2)

#     if os.path.exists(model_path):
#         checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
#     else:
#         raise AssertionError("There is not checkpoint for {}".format(model_path))
   
#     state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
#     model.load_state_dict(state_dict)
    
#     # model = bn_calibrate_from_image(
#     #     model,
#     #     img_norm,
#     #     valid_mask,
#     #     device=device,     # calibration on GPU
#     #     steps=100,
#     #     batch_size=8,      # with 8GB + U-Net, start conservative
#     #     patch_size=384,    # if OOM, try 256
#     #     bn_momentum=0.3
#     # )
#     # for m in model.modules():
#     #     if isinstance(m, torch.nn.BatchNorm2d):
#     #         m.train() 
#     #         # print(m.running_mean.mean(), m.running_var.mean())
#     #         # break
#     # i = 0
#     # for m in model.modules():
#     #     if isinstance(m, torch.nn.BatchNorm2d):
#     #         print(i, m.running_var.mean().item())
#     #         i += 1
#     #         if i == 10: break
#     return model

def load_model(config_path, checkpoint_path, device="cpu"):
    """
    Load an mmsegmentation model from a config and MMEngine checkpoint.

    Args:
        config_path (str): Path to the mmseg .py config file.
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (str): Device string, e.g. "cpu" or "cuda:0".

    Returns:
        model: Loaded mmseg model in eval mode.
    """
    from mmseg.apis import init_model

    # PyTorch 2.6 changed weights_only default to True, but MMEngine checkpoints
    # embed HistoryBuffer and numpy objects not in the safe-globals allowlist.
    _original_load = torch.load
    torch.load = functools.partial(_original_load, weights_only=False)
    try:
        model = init_model(config_path, checkpoint_path, device=device)
    finally:
        torch.load = _original_load

    return model


def forward_model_committee(
    model_paths,
    img_norm,
    valid_mask=None,
    class_colors=np.uint8([[0, 255, 255], [255, 130, 0]]),
    device="cpu",
):
    """
    Run an ensemble (committee) of segmentation models by averaging logits.

    Parameters
    ----------
    model_paths : list[str] | tuple[str] | str
        One or more model checkpoint paths (.pt). If str, treated as a single model.
    img_norm : torch.Tensor
        Normalized input image tensor, shape (batch, channels, height, width).
    valid_mask : np.ndarray | None
        Boolean mask indicating valid pixels, shape (height, width).
        True = valid pixel, False = invalid/no-data.
    class_colors : np.ndarray
        RGB palette for each class, shape (num_classes, 3).
    device : str
        "cpu" or "cuda".

    Returns
    -------
    colored_pred_map : np.ndarray
        RGB colored prediction map, shape (height, width, 3).
    """

    if isinstance(model_paths, (str, bytes)):
        model_paths = [model_paths]
    if len(model_paths) == 0:
        raise ValueError("model_paths is empty")
    
    # Make sure input is on the right device
    img_norm = img_norm.to(device)

    logits_sum = None

    for mp in model_paths:
        model = load_model(mp, img_norm, valid_mask, device="cpu")
        
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            logits, _ = model(img_norm)  # (B, C, H, W)
        
        if logits_sum is None:
            logits_sum = logits.detach().clone()
        else:
            logits_sum += logits.detach()
        
        # free model ASAP (useful on CPU too, and especially on GPU)
        del model
    
    logits_avg = logits_sum / float(len(model_paths))
    
    # Convert to prediction
    probs_map = logits_avg.squeeze(0).softmax(0)     # (C, H, W)
    pred_map = torch.argmax(probs_map, 0).cpu().numpy()  # (H, W)
    
    colored_pred_map = class_colors[pred_map]  # (H, W, 3)
    
    if valid_mask is not None:
        colored_pred_map[~valid_mask] = 255
    return colored_pred_map

def get_input_divisor(model):
    """
    Derive the minimum spatial divisor required by the loaded backbone.

    Covers:
      - UNet-style (num_stages + strides + downsamples attributes)
      - ResNet/generic (strides attribute only)
      - Unknown: falls back to 32
    """
    backbone = model.backbone
    # UNet: mirrors _check_input_divisible logic
    if hasattr(backbone, 'num_stages') and hasattr(backbone, 'strides') and hasattr(backbone, 'downsamples'):
        rate = 1
        for i in range(1, backbone.num_stages):
            if backbone.strides[i] == 2 or backbone.downsamples[i - 1]:
                rate *= 2
        return rate
    # ResNet / generic backbones that expose per-stage strides
    if hasattr(backbone, 'strides'):
        return int(np.prod(backbone.strides))
    return 32

def inference(model, img, device):
    """
    Run inference on a normalized (H, W, C) image.

    Args:
        model: Loaded mmseg model (EncoderDecoder subclass).
        img (np.ndarray): Normalized float32 array of shape (H, W, C).
        device: Torch device to run inference on.

    Returns:
        numpy.ndarray: Predicted segmentation map of shape (H, W).
    """
    height, width = img.shape[:2]

    # (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad to meet backbone input divisor requirements (e.g. 32 for ResNet, 64 for UNet)
    divisor = get_input_divisor(model)
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        # img_shape tells predict_by_feat to resize logits back to original size
        seg_logits = model.encode_decode(tensor, batch_img_metas=[{"img_shape": (height, width)}])

    seg_map = seg_logits.argmax(dim=1).squeeze().cpu().numpy()
    return seg_map

# Run model
def run_model():
    parser = argparse.ArgumentParser(description="Load and verify an mmsegmentation checkpoint")
    parser.add_argument("--config", default="D:/Temp/Model_update_code/sea-ice-mmseg/configs/arcticscope/unet_1xb8-amp-coslr-30ki_rcm.py", type=str, help="Path to the mmseg config .py file")
    parser.add_argument("--checkpoint", default="D:/Temp/Model_update_code/sea-ice-mmseg/work_dirs/unet_1xb8-amp-coslr-30ki_rcm/best_mFscore_iter_3500.pth", type=str, help="Path to the .pth checkpoint file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on (default: cpu)")
    args = parser.parse_args()

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {args.device}")

    img = np.random.rand(500, 500, 2).astype(np.float32)
    img = Normalize_mean_std(img)

    model = load_model(args.config, args.checkpoint, device=args.device)
    device = next(model.parameters()).device
    print(f"Model loaded: {type(model).__name__}  |  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"Input divisor:  {get_input_divisor(model)}")
    print("Running inference on 500x500x2 toy image...")
    seg_map = inference(model, img, device)
    print(f"Output shape:   {seg_map.shape}")
    print(f"Unique classes: {np.unique(seg_map)}")
