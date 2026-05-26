"""
Contrast enhancement utilities for uint8 images using histogram-based quantiles and LUTs.

Last modified: Mar 2026
"""

import numpy as np

def precompute_valid_hist_u8(img_u8: np.ndarray, valid_mask: np.ndarray):
    """
    img_u8: (H,W,C) uint8
    valid_mask: (H,W) bool
    Returns hist: (C,256) int32 and n_valid: (C,) int32
    """
    assert img_u8.dtype == np.uint8
    H, W, C = img_u8.shape
    assert valid_mask.shape == (H, W)

    hist = np.empty((C, 256), dtype=np.int32)
    n_valid = np.empty(C, dtype=np.int32)

    # This copies only the valid pixels once per channel (acceptable since done once per image)
    for c in range(C):
        x = img_u8[..., c][valid_mask]
        hist[c] = np.bincount(x.ravel(), minlength=256).astype(np.int32)
        n_valid[c] = x.size

    return hist, n_valid

def quantile_from_hist_linear(hist256: np.ndarray, n: int, th: float) -> float:
    """
    hist256: (256,) int32, n = hist256.sum()
    th in [0,1]
    Returns float quantile in [0,255] with within-bin linear interpolation.
    """
    if n <= 0:
        return np.nan
    th = float(np.clip(th, 0.0, 1.0))
    target = th * (n - 1)

    cum = 0.0
    prev = 0.0
    for v in range(256):
        prev = cum
        cum += hist256[v]
        if cum > target:
            cnt = hist256[v]
            if cnt <= 0:
                return float(v)
            frac = (target - prev) / cnt  # [0,1)
            return float(v) + frac
    return 255.0


def build_lut_u8(lo: float, hi: float) -> np.ndarray:
    """
    LUT[256] mapping 0..255 -> 0..255 using contrast stretch.
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros(256, dtype=np.uint8)

    vals = np.arange(256, dtype=np.float32)
    y = (vals - np.float32(lo)) * (255.0 / (np.float32(hi) - np.float32(lo)))
    y = np.clip(y, 0.0, 255.0)
    return (y + 0.5).astype(np.uint8)


def enhance_outlier_slider(
    img_u8: np.ndarray,
    hist: np.ndarray,
    n_valid: np.ndarray | None = None,
    s: float = 0.0              # slider in [0, 0.25]
):
    """
    img_u8:
      - (H,W) uint8 or (H,W,C) uint8
    hist:
      - (C,256) int counts computed on valid pixels
    n_valid:
      - (C,) valid pixel counts (or scalar if C==1). If omitted, counts are
        derived from hist. A valid mask is also accepted for windowed images.
    s:
      - 0 -> exact original
      - >0 -> bth = s/2, uth = 1 - bth

    Returns:
      out_u8 (same shape as img_u8), lo (C,), hi (C,)
    """
    assert img_u8.dtype == np.uint8

    # Normalize shape to (H,W,C)
    if img_u8.ndim == 2:
        img3 = img_u8[..., None]
    else:
        img3 = img_u8

    H, W, C = img3.shape

    hist = np.asarray(hist)
    if hist.ndim == 1:
        hist = hist[None, :]

    assert hist.shape[1] == 256

    # Full-scene HH/HV images are stored as tiled grayscale RGB, so their
    # histograms have 3 identical channels. Window reads are plain 2D arrays.
    if C == 1 and hist.shape[0] > 1:
        hist = hist[:1]

    assert hist.shape[0] == C

    # Near-zero -> return original quickly
    if s == 0.0:
        out3 = img3.copy()
        return out3[..., 0] if img_u8.ndim == 2 else out3

    if n_valid is None:
        n_valid_counts = hist.sum(axis=1)
    else:
        n_valid_arr = np.asarray(n_valid)
        if n_valid_arr.dtype == bool:
            if n_valid_arr.ndim == 2 and C == 1:
                n_valid_counts = np.array([int(n_valid_arr.sum())], dtype=np.int64)
            elif n_valid_arr.ndim == 3 and n_valid_arr.shape[2] == C:
                n_valid_counts = n_valid_arr.reshape(-1, C).sum(axis=0)
            else:
                n_valid_counts = hist.sum(axis=1)
        else:
            n_valid_counts = n_valid_arr

    bth = 0.5 * s
    uth = 1.0 - bth

    out3 = np.empty_like(img3)

    # Compute quantiles and LUTs per channel, then apply LUT to the image
    for c in range(C):
        n = int(n_valid_counts[c]) if np.ndim(n_valid_counts) > 0 else int(n_valid_counts)
        l = quantile_from_hist_linear(hist[c], n, bth)
        h = quantile_from_hist_linear(hist[c], n, uth)

        lut = build_lut_u8(l, h)
        out3[..., c] = lut[img3[..., c]]  # fast LUT map

    return out3[..., 0] if img_u8.ndim == 2 else out3
