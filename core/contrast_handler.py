import cv2
import numpy as np

from core.utils import scale_channels_inplace

def prepare_sorted_data(img, valid_mask=None):
    """
    Precompute sorted per-channel values for fast O(1) percentile queries.
    img: (H,W) or (H,W,C)
    valid_mask: (H,W) bool, True where pixels are valid (NOT NaN/masked)
                If None: uses np.isfinite on channel 0.
    Returns: list of 1D sorted arrays, length C (or 1).
    """
    x = np.asarray(img)
    if x.ndim == 2:
        x = x[:, :, None]

    if valid_mask is None:
        valid_mask = np.isfinite(x[:, :, 0])
    else:
        valid_mask = valid_mask.astype(bool)

    sorted_list = []
    for c in range(x.shape[2]):
        v = x[:, :, c][valid_mask] # select valid pixels
        # If dtype isn't float, no NaNs, but still fine
        v = np.asarray(v).ravel()
        v_sorted = np.sort(v, axis=0)
        sorted_list.append(v_sorted)

    return sorted_list


def percentile_from_sorted(x_sorted, th):
    """
    O(1) percentile lookup from a pre-sorted 1D array, with linear interpolation.
    th in [0,1]
    """
    th = float(np.clip(th, 0.0, 1.0))
    n = x_sorted.size
    if n == 0:
        return np.nan

    i = th * (n - 1)
    i0 = int(i)
    w = i - i0
    i1 = min(i0 + 1, n - 1)
    return (1.0 - w) * x_sorted[i0] + w * x_sorted[i1]


def median_repair_outliers_fast(
    img,
    lo=None, hi=None,
    p_lo=1.0, p_hi=99.0,
    ksize=11,
    q_lo=None, q_hi=None,
    return_mask=False,
):
    """
    Efficient outlier repair:
      - repair only values outside [lo, hi] by replacing them with medianBlur values
      - uses uint16 quantization for fast cv2.medianBlur
      - works on 2D input

    Returns:
      out (float32)  or (out, out_mask) if return_mask=True
    """
    x = np.asarray(img)
    if x.ndim != 2:
        raise ValueError("img must be 2D")

    xf = x.astype(np.float32, copy=False)

    finite = np.isfinite(xf)
    if not finite.all():
        xf2 = xf.copy()
        xf2[~finite] = 0.0
        xf = xf2

    # Compute lo/hi if not given (slower; best is passing lo/hi)
    if lo is None or hi is None:
        v = xf[finite] if finite is not None else xf.ravel()
        lo2, hi2 = np.percentile(v, [p_lo, p_hi])
        lo = lo2 if lo is None else lo
        hi = hi2 if hi is None else hi

    out_mask = (xf < lo) | (xf > hi)
    if not np.any(out_mask):
        return (xf, out_mask) if return_mask else xf

    if (ksize & 1) == 0:
        ksize += 1  # OpenCV requires odd

    if q_lo is None:
        q_lo = lo
    if q_hi is None:
        q_hi = hi
    if q_hi <= q_lo:
        q_hi = q_lo + 1e-6

    scale = 65535.0 / (q_hi - q_lo)
    u16 = np.clip((xf - q_lo) * scale, 0.0, 65535.0).astype(np.uint16)

    med_u16 = cv2.medianBlur(u16, ksize)
    med = med_u16.astype(np.float32) / scale + q_lo

    out = xf.copy()
    out[out_mask] = med[out_mask]

    return (out, out_mask) if return_mask else out


def enhance_outlier_slider(
    img,
    sorted_data,
    land_nan_mask=None,
    s=0.0,          # slider in [0,0.25]
    s_max=0.25,
    ksize=11,
    output_dtype=np.uint8,
):
    """
    Slider-controlled outlier:
      s=0 -> exact original
      s>0 -> bth=s/2, uth=1-bth; repair outliers beyond those percentiles.

    img: (H,W) or (H,W,C)
    sorted_data: list of sorted arrays from prepare_sorted_data()
    land_nan_mask: (H,W) bool where you want to exclude pixels (can be placeholder False)
    """
    x = np.asarray(img)
    if x.ndim == 2:
        x = x[:, :, None]

    H, W, C = x.shape
    if land_nan_mask is None:
        land_nan_mask = np.zeros((H, W), dtype=bool)
    # valid = ~land_nan_mask

    s = float(np.clip(s, 0.0, s_max))

    # s=0 -> return original exactly
    if s == 0.0:
        out = x.copy()
        return out[:, :, 0] if img.ndim == 2 else out

    # Map slider -> bth/uth (as suggested)
    bth = s / 2.0
    uth = 1.0 - bth

    out = x.astype(np.float32, copy=True)

    # Per-channel repair using precomputed percentiles
    for c in range(C):
        lo = percentile_from_sorted(sorted_data[c], bth)
        hi = percentile_from_sorted(sorted_data[c], uth)
        scale_channels_inplace(out, lo, hi, c)

    out = np.clip(out, 0, 255)
    out = out.astype(output_dtype)

    return out[:, :, 0] if img.ndim == 2 else out
