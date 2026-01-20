"""
Contrast enhancement using percentile clipping and CLAHE.
Reference: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
Utilizes OpenCV's CLAHE implementation

Notes about Contrast Limited Adaptive Histogram Equalization (CLAHE):
- Divides the image into small blocks called "tiles" (tileGridSize)
- Create histogram for each tile
- Clips the histogram of each tile to limit contrast amplification (clipLimit)
- Redistributes clipped pixels evenly across all histogram bins
- Applies histogram equalization to each tile
- Interpolates between tiles to avoid artifacts

Last modified: Jan 2026
"""

import numpy as np
import cv2


def enhance_image(img, land_nan_mask, contrast=1.0, bth=0.01, uth=0.99,
    clipLimit=2.0, tileGridSize=(8, 8),
    ):

    assert img.dtype == np.uint8, "Expecting uint8"

    valid = ~land_nan_mask
    enhanced = img.copy()

    for b in range(img.shape[2]):
        channel = img[:, :, b]

        # Percentile clipping
        lo = np.percentile(channel[valid], bth * 100)
        hi = np.percentile(channel[valid], uth * 100)

        clipped = np.clip(channel.astype(np.float32), lo, hi)
        clipped = ((clipped - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)

        # Create CLAHE object and apply to clipped channel
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        eq = clahe.apply(clipped)

        enhanced[:, :, b][valid] = eq[valid]

    enhanced[land_nan_mask] = img[land_nan_mask]  # keep masked pixels unchanged

    return np.clip(enhanced, 0, 255).astype(np.uint8)


"""
Blend original and enhanced images based on contrast factor, called on slider change
contrast: float in [0,1]
    0 -> return original image
    1 -> return fully enhanced
    0-1 -> blend between original and enhanced
"""
def blend(original, enhanced, s):
    s = float(np.clip(s, 0.0, 1.0))
    out = (1.0 - s) * original.astype(np.float32) + s * enhanced.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
