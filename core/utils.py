from numba import njit, prange, cuda
import numpy as np
from numba.typed import List
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
def blend_overlay(pred, img, boundmask, landmask, local_boundmask, alpha):
    h, w, c = pred.shape
    out = np.empty((h, w, c), dtype=np.float32)

    beta = 1 - alpha

    for y in prange(h):
        for x in range(w):
            if landmask[y, x]:
                out[y, x, 0] = img[y, x, 0]
                out[y, x, 1] = img[y, x, 1]
                out[y, x, 2] = img[y, x, 2]
            elif boundmask[y, x]:
                out[y, x, :] = pred[y, x, :]
            elif local_boundmask is not None and local_boundmask[y, x]:
                out[y, x, :] = 255
            else:
                for ch in range(c):
                    out[y, x, ch] = alpha * pred[y, x, ch] + beta * img[y, x, ch]

    return out

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Optimize this
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
def apply_brightness(image, nan_mask, brightness=0.0, clip=True):
    # Adjust brightness
    h, w, c = image.shape
    adjusted = np.empty((h, w, c), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            for ch in range(c):
                if nan_mask[y, x]:
                    adjusted[y, x, ch] = image[y, x, ch]
                else:
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
