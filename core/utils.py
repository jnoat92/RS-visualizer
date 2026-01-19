from numba import njit, prange, cuda
import numpy as np
from skimage.measure import find_contours


@cuda.jit
def blend_overlay_cuda(pred, img, boundmask, landmask, alpha, out):
    y, x = cuda.grid(2)
    h, w = pred.shape[:2]
    beta = 1 - alpha

    if y < h and x < w:
        if landmask[y, x]:
            for c in range(3):
                out[y, x, c] = 255
        elif boundmask[y, x]:
            for c in range(3):
                out[y, x, c] = pred[y, x, c]
        else:
            for c in range(3):
                out[y, x, c] = alpha * pred[y, x, c] + beta * img[y, x, c]

@njit(parallel=True)
def blend_overlay(pred, img, boundmask, landmask, alpha):
    h, w, c = pred.shape
    out = np.empty((h, w, c), dtype=np.float32)

    beta = 1 - alpha

    for y in prange(h):
        for x in range(w):
            if landmask[y, x]:
                out[y, x, 0] = 255
                out[y, x, 1] = 255
                out[y, x, 2] = 255
            elif boundmask[y, x]:
                out[y, x, :] = pred[y, x, :]
            else:
                for ch in range(c):
                    out[y, x, ch] = alpha * pred[y, x, ch] + beta * img[y, x, ch]

    return out

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def generate_boundaries(lbl):
    boundmask = np.zeros_like(lbl, dtype=bool)   
    for lvl in np.unique(lbl):
        level_ctrs = find_contours(lbl, level=lvl)
        for c in level_ctrs:
            try:
                contours = np.concatenate((contours, c), axis=0)
            except:
                contours = c
    if 'contours' in locals():
        contours = np.uint16(contours)
        boundmask[contours[:,0], contours[:,1]] = True
    return boundmask

@njit(parallel=True)
def apply_brightness(image, brightness=0.0, clip=True):
    # Adjust brightness
    h, w, c = image.shape
    adjusted = np.empty((h, w, c), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            for ch in range(c):
                adjusted[y, x, ch] = image[y, x, ch] + brightness * 128

    if clip:
        for y in prange(h):
            for x in range(w):
                for ch in range(c):
                    if adjusted[y, x, ch] < 0:
                        adjusted[y, x, ch] = 0
                    elif adjusted[y, x, ch] > 255:
                        adjusted[y, x, ch] = 255

    return adjusted.astype(np.uint8)


@njit(inline="always")
def _median_1d(buf, n):
    # n = number of valid elements in buf[0:n]
    if n <= 0:
        return 0.0
    tmp = buf[:n].copy()
    tmp.sort()
    mid = n // 2
    if (n & 1) == 1:  # odd
        return tmp[mid]
    else:            # even
        return 0.5 * (tmp[mid - 1] + tmp[mid])

# ---- 2D image version ----
@njit(parallel=True, fastmath=True)
def _median_filter_2d_numba(img, mask, xs, ys, kernel_size):
    H, W = img.shape
    half = kernel_size // 2
    out = img.copy()

    # max window elements = kernel_size * kernel_size
    max_n = kernel_size * kernel_size

    for i in prange(xs.shape[0]):
        x = xs[i]
        y = ys[i]

        a = x - half
        if a < 0:
            a = 0
        b = x + half
        if b > H:
            b = H

        c = y - half
        if c < 0:
            c = 0
        d = y + half
        if d > W:
            d = W

        # collect masked values into a fixed buffer
        buf = np.empty(max_n, dtype=img.dtype)
        n = 0
        for xx in range(a, b):
            for yy in range(c, d):
                if mask[xx, yy]:
                    buf[n] = img[xx, yy]
                    n += 1

        # if no valid neighbors, keep original
        if n > 0:
            out[x, y] = _median_1d(buf, n)

    return out

# ---- 3D image version (e.g., HxWxC) ----
@njit(parallel=True, fastmath=True)
def _median_filter_3d_numba(img, mask, xs, ys, kernel_size):
    H, W, C = img.shape
    half = kernel_size // 2
    out = img.copy()

    max_n = kernel_size * kernel_size

    for i in prange(xs.shape[0]):
        x = xs[i]
        y = ys[i]

        a = x - half
        if a < 0:
            a = 0
        b = x + half
        if b > H:
            b = H

        c = y - half
        if c < 0:
            c = 0
        d = y + half
        if d > W:
            d = W

        # buffers per channel
        buf0 = np.empty(max_n, dtype=img.dtype)
        buf1 = np.empty(max_n, dtype=img.dtype) if C > 1 else buf0
        buf2 = np.empty(max_n, dtype=img.dtype) if C > 2 else buf0

        n = 0
        for xx in range(a, b):
            for yy in range(c, d):
                if mask[xx, yy]:
                    v = img[xx, yy]
                    buf0[n] = v[0]
                    if C > 1:
                        buf1[n] = v[1]
                    if C > 2:
                        buf2[n] = v[2]
                    n += 1

        if n > 0:
            out[x, y, 0] = _median_1d(buf0, n)
            if C > 1:
                out[x, y, 1] = _median_1d(buf1, n)
            if C > 2:
                out[x, y, 2] = _median_1d(buf2, n)

    return out

def median_filter_numba_parallel(img, clips, mask, kernel_size=10):
    img = np.ascontiguousarray(img)
    mask = np.ascontiguousarray(mask).astype(np.bool_)

    # compute outliers in Python (fast + simple)
    outliers = (img < clips[0]) | (img > clips[1])
    if img.ndim == 3:
        outliers = outliers & mask[:, :, None]
        out2d = np.any(outliers, axis=2)  # filter pixel if any channel is outlier
    else:
        out2d = outliers & mask

    xs, ys = np.where(out2d)
    xs = np.ascontiguousarray(xs, dtype=np.int64)
    ys = np.ascontiguousarray(ys, dtype=np.int64)

    if img.ndim == 2:
        return _median_filter_2d_numba(img, mask, xs, ys, kernel_size)
    elif img.ndim == 3:
        return _median_filter_3d_numba(img, mask, xs, ys, kernel_size)
    else:
        raise ValueError("img must be 2D or 3D (H×W or H×W×C).")

