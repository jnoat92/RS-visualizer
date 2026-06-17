'''
Segmentation related functions

Last modified: Jun 2026
'''

import numpy as np
from shapely import polygons
from skimage.measure import find_contours, label
from magic_py.magic_rag import magic_rag

from model.utils import erase_edge_touching_polygons_numba

def get_segment_contours(pred, x, y):
    """Get contours for a specific segment in the prediction image."""
    target_rgb = pred[x, y]
    mask = np.all(pred == target_rgb, axis=-1)
    labeled = label(mask, connectivity=2)
    region_label = labeled[x, y]

    if region_label == 0:
        return []

    segment_mask = labeled == region_label

    # Get list of contours — each a Nx2 array of [row, col]
    contours = find_contours(segment_mask.astype(np.uint8), level=0.5)

    return contours, segment_mask

# Segmentation on a local zoomed area, applied to cropped image only
def Map_labels(labels):
    # map labels to [0, 1, 2,..., n_labels-1]
    id = np.unique(labels)
    aux_id = -1 * np.ones_like(labels)     
    c = 0
    for i in id:                                
        if i != -1:
            aux_id[labels == i] = c
            c += 1
    return aux_id

def IRGS(img, n_classes, n_iter, mask=None):
    # --- RUN IRGS --- #
    rag = None
    if mask is None:
        rag = magic_rag(img, msk=None, N_class=n_classes, verbose=True)
    else:
        rag = magic_rag(img, msk=mask, N_class=n_classes, verbose=True)
    #print("Initializing k-means with", n_classes, "classes")
    rag.initialize_kmeans()
    #print("Performing", str(n_iter), "IRGS iterations...")
    # for j in tqdm(range(n_iter), ncols=50):
    for j in range(n_iter):
        # rag.irgs_step(beta1=beta1, current_iter=i+1)
        rag.irgs_step(current_iter=j+1)
    irgs_output = rag.result_image
    # irgs_output = rag.result_image_with_boundaries
    # boundaries = np.int16(rag.bmp == -2) # Not consistent with rag.result_image_with_boundaries
    boundaries = np.int16(rag.result_image_with_boundaries != -2)
    boundaries[boundaries == 0] = -1
    boundaries[irgs_output < 0] = -1
    irgs_output[irgs_output < 0] = -1           # background and boundaries.
                                                # IRGS returns an aditional class with 
                                                # label -2 for landmask and boundaries\
    irgs_output = Map_labels(irgs_output)
    return irgs_output, boundaries

def one_pixel_boundaries(polygons, background=None):
    """
    Returns a boolean boundary mask that is 1-pixel thick.
    Marks a pixel (r,c) as boundary if it differs from its right or down neighbor.
    Tie-break: boundary pixel belongs to the current pixel (top/left side).
    """
    polygons = np.asarray(polygons)

    boundaries = np.zeros(polygons.shape, dtype=bool)

    # compare with right neighbor
    # get all pixels except last column and compare with all pixels except first column
    diff_right = polygons[:, :-1] != polygons[:, 1:]
    boundaries[:, :-1] |= diff_right

    # compare with down neighbor
    # get all pixels except last row and compare with all pixels except first row
    diff_down = polygons[:-1, :] != polygons[1:, :]
    boundaries[:-1, :] |= diff_down

    if background is not None:
        boundaries &= (polygons != background)  # optional: don't draw boundaries *on* background pixels

    return boundaries

def remove_edge_touching_polygons(irgs_output):
    """   
    Get only the enclosed polygons that don't touch the edges of the selected area
    Set the polygons touching the edges to -1
    """
    # Using numba-optimized version instead of pure Python for performance
    polygons = np.array(irgs_output, dtype=np.int32)  # ensure numpy
    erase_edge_touching_polygons_numba(polygons, background=-1)

    boundaries = one_pixel_boundaries(polygons).astype(np.int8)
    boundaries[boundaries == 1] = -1
    boundaries[boundaries == 0] = 1

    return polygons, boundaries
