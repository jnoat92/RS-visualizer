import numpy as np
import cv2


def enhance_image(
    img,
    land_nan_mask,
    contrast=1.0,          # slider in [0, 1]
    bth=0.01,
    uth=0.99,
    clipLimit=2.0,
    tileGridSize=(8, 8),
):
    """
    contrast: float in [0,1]
      0 -> return original image
      1 -> return fully enhanced (percentile clip + CLAHE)
    """

    # clamp slider
    contrast = float(np.clip(contrast, 0.0, 1.0))

    # slider=0 => exact original
    if contrast == 0.0:
        return img.copy()

    assert img.dtype == np.uint8, "Expecting uint8"

    valid = ~land_nan_mask
    enhanced = img.copy()

    for b in range(img.shape[2]):
        channel = img[:, :, b]

        # Percentile clipping (bth/uth matter)
        lo = np.percentile(channel[valid], bth * 100)
        hi = np.percentile(channel[valid], uth * 100)

        clipped = np.clip(channel.astype(np.float32), lo, hi)
        clipped = ((clipped - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        eq = clahe.apply(clipped)

        enhanced[:, :, b][valid] = eq[valid]

    enhanced[land_nan_mask] = img[land_nan_mask]  # keep masked pixels unchanged

    # # Blend based on contrast slider, can remove since blend is after this function
    # out = ((1.0 - contrast) * img.astype(np.float32) +
    #        contrast * enhanced.astype(np.float32))

    return np.clip(enhanced, 0, 255).astype(np.uint8)

def blend(original, enhanced, s):
    s = float(np.clip(s, 0.0, 1.0))
    out = (1.0 - s) * original.astype(np.float32) + s * enhanced.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
