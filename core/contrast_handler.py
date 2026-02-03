import cv2
import numpy as np

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
    n_valid: np.ndarray,
    s: float = 0.0              # slider in [0, 0.25]
):
    """
    img_u8:
      - (H,W) uint8 or (H,W,C) uint8
    hist:
      - (C,256) int counts computed on valid pixels
    n_valid:
      - (C,) valid pixel counts (or scalar if C==1)
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
    assert hist.shape[0] == C and hist.shape[1] == 256

    # Near-zero -> return original quickly
    if s == 0.0:
        out3 = img3.copy()
        return out3[..., 0] if img_u8.ndim == 2 else out3

    bth = 0.5 * s
    uth = 1.0 - bth

    out3 = np.empty_like(img3)

    for c in range(C):
        n = int(n_valid[c]) if np.ndim(n_valid) > 0 else int(n_valid)
        l = quantile_from_hist_linear(hist[c], n, bth)
        h = quantile_from_hist_linear(hist[c], n, uth)

        lut = build_lut_u8(l, h)
        out3[..., c] = lut[img3[..., c]]  # fast LUT map

    return out3[..., 0] if img_u8.ndim == 2 else out3