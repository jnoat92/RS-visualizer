import numpy as np
from model.DL.unet.unet_model import UNet
import torch
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
    

def load_model(model_path, device='cpu'):
    """
    Load a pretrained UNet model from a checkpoint.
    
    Parameters
    ----------
    model_path : str
        Path to the model checkpoint file.
    device : str, optional
        Device to load the model on ('cpu' or 'cuda'). Default is 'cpu'.
    
    Returns
    -------
    model : UNet
        Loaded UNet model with weights from checkpoint.
    
    Raises
    ------
    AssertionError
        If the checkpoint file does not exist at model_path.
    """
    model = UNet(2, 2)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    else:
        raise AssertionError("There is not checkpoint for {}".format(model_path))
    
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    return model

def forward_model(model, img_norm, nan_mask=None, class_colors=np.uint8([[0, 255, 255], [255, 130, 0]])):
    """
    Run model inference on normalized SAR image and generate colored prediction map.
    
    Parameters
    ----------
    model : UNet
        Pretrained UNet model for semantic segmentation.
    img_norm : torch.Tensor
        Normalized input image tensor, shape (batch, channels, height, width).
    nan_mask : np.ndarray
        Boolean mask indicating valid pixels, shape (height, width).
        True = valid pixel, False = no-data/invalid pixel.
    class_colors : np.ndarray, optional
        RGB color palette for each class, shape (num_classes, 3).
        Default is [[0, 255, 255], [255, 130, 0]] (cyan and orange).
    
    Returns
    -------
        - 'colored_pred_map': RGB colored prediction map, shape (height, width, 3)
    """

    logits, _ = model(img_norm)  # ~20 seconds on CPU
    probs_map = logits.squeeze(0).softmax(0)
    pred_map = torch.argmax(probs_map, 0).detach().cpu().numpy()
    
    colored_pred_map = class_colors[pred_map]
    if nan_mask is not None:
        colored_pred_map[~nan_mask] = 255
    
    return colored_pred_map
