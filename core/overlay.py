'''
Overlay related functions

Last modified: Jan 2026
'''

import numpy as np
from core.utils import blend_overlay_cuda, blend_overlay
from numba import cuda

    # Display handle

def overlay_GPU(pred, img, boundmask, landmask, local_boundmask, alpha):
    # Using Numba on the GPU to parallelize 
    h, w, c = pred.shape
    pred = pred.astype(np.float32)
    img = img.astype(np.float32)

    d_pred = cuda.to_device(pred)
    d_img = cuda.to_device(img)
    d_boundmask = cuda.to_device(boundmask)
    d_landmask = cuda.to_device(landmask)
    d_out = cuda.device_array((h, w, c), dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = (w + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_y = (h + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid = (blockspergrid_y, blockspergrid_x)

    blend_overlay_cuda[blockspergrid, threadsperblock](d_pred, d_img, d_boundmask, d_landmask, alpha, d_out)

    blended = d_out.copy_to_host()
    # self.overlay = blended.astype(np.uint8)
    return blended.astype(np.uint8)

def overlay(pred, img, boundmask, landmask, local_boundmask,alpha):

    # alpha = self.alpha
    # beta = 1 - alpha
    # overlay = alpha * self.pred_resized + beta * self.img_resized
    # overlay = np.where(self.boundmask_resized[..., None], self.pred_resized, overlay)
    # overlay = np.where(self.landmask_resized[..., None], 255, overlay)        # Use float32 for fast computation

    # Using Numba on the CPU to parallelize 
    pred = pred.astype(np.float32)
    img = img.astype(np.float32)
    overlay = blend_overlay(pred, img, boundmask, landmask, local_boundmask, alpha)

    return overlay.astype(np.uint8)

def compose_overlay(pred, img, boundmask, landmask, local_boundmask, alpha, use_gpu=False):
    if use_gpu:
        return overlay_GPU(pred, img, boundmask, landmask, local_boundmask, alpha)
    else:
        return overlay(pred, img, boundmask, landmask, local_boundmask, alpha)