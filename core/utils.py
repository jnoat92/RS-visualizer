"""
Utility functions for image processing and visualization.

Last modified: Apr 2026
"""

from numba import njit, prange, cuda
import numpy as np
from numba.typed import List
from skimage.measure import find_contours


@cuda.jit
def blend_overlay_cuda(pred, img, boundmask, landmask, alpha, out):
    """Blend the prediction and image using GPU acceleration with Numba's CUDA."""
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
    """Blend the prediction and image using CPU with Numba for parallelization."""
    h, w, c = pred.shape
    out = np.empty((h, w, c), dtype=np.float32)

    beta = 1 - alpha

    for y in prange(h):
        for x in range(w):
            if landmask[y, x]:
                out[y, x, 0] = img[y, x, 0]
                out[y, x, 1] = img[y, x, 1]
                out[y, x, 2] = img[y, x, 2]
            elif boundmask is not None and boundmask[y, x]:
                out[y, x, :] = pred[y, x, :]
            elif local_boundmask is not None and local_boundmask[y, x]:
                out[y, x, :] = 255
            else:
                for ch in range(c):
                    out[y, x, ch] = alpha * pred[y, x, ch] + beta * img[y, x, ch]

    return out

def rgb2gray(rgb):
    """Convert an RGB image to grayscale using standard luminosity method."""
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Optimize this
def generate_boundaries(lbl):
    """Generate a boolean mask of boundary pixels from a labeled segmentation image."""
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
    """Apply brightness adjustment to the image while preserving NaN areas defined by nan_mask."""
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

def ds_to_src_pixel(row_ds, col_ds, src_h, src_w, dst_h, dst_w):
    """Convert pixel coordinates from the downsampled image to the original image using bilinear scaling."""
    x_scale = src_w / dst_w
    y_scale = src_h / dst_h
    col_src = (col_ds + 0.5) * x_scale
    row_src = (row_ds + 0.5) * y_scale
    return row_src, col_src

def tiepoints_1d_to_grid(tie_lines, tie_pixels, tie_lats, tie_lons):
    """Convert 1D arrays of tie points into 2D grids for latitude and longitude."""
    tie_lines  = np.asarray(tie_lines,  dtype=float)
    tie_pixels = np.asarray(tie_pixels, dtype=float)
    tie_lats   = np.asarray(tie_lats,   dtype=float)
    tie_lons   = np.asarray(tie_lons,   dtype=float)

    if not (len(tie_lines) == len(tie_pixels) == len(tie_lats) == len(tie_lons)):
        raise ValueError("All tie arrays must have the same length.")

    rows = np.array(sorted(np.unique(tie_lines)), dtype=float)
    cols = np.array(sorted(np.unique(tie_pixels)), dtype=float)

    r_index = {r: i for i, r in enumerate(rows)}
    c_index = {c: j for j, c in enumerate(cols)}

    lat_grid = np.full((len(rows), len(cols)), np.nan, dtype=float)
    lon_grid = np.full((len(rows), len(cols)), np.nan, dtype=float)

    for r, c, la, lo in zip(tie_lines, tie_pixels, tie_lats, tie_lons):
        lat_grid[r_index[r], c_index[c]] = la
        lon_grid[r_index[r], c_index[c]] = lo

    if np.isnan(lat_grid).any() or np.isnan(lon_grid).any():
        raise ValueError("Tie points do not cover a full regular grid (missing row/col combos).")

    return rows, cols, lat_grid, lon_grid

def make_pix2ll(rows, cols, lat_grid, lon_grid):
    """
    Create a function that maps pixel coordinates (row, col) to latitude and longitude 
    using bilinear interpolation on the provided tie point grid.
    """
    rows = np.asarray(rows, float)
    cols = np.asarray(cols, float)
    lat_grid = np.asarray(lat_grid, float)
    lon_grid = np.asarray(lon_grid, float)

    def bilinear(col, row, xs, ys, v):
        col = float(np.clip(col, xs[0], xs[-1]))
        row = float(np.clip(row, ys[0], ys[-1]))

        j = np.searchsorted(xs, col) - 1
        i = np.searchsorted(ys, row) - 1
        j = int(np.clip(j, 0, len(xs) - 2))
        i = int(np.clip(i, 0, len(ys) - 2))

        x0, x1 = xs[j], xs[j+1]
        y0, y1 = ys[i], ys[i+1]

        q00 = v[i,   j]
        q01 = v[i,   j+1]
        q10 = v[i+1, j]
        q11 = v[i+1, j+1]

        tx = 0.0 if x1 == x0 else (col - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (row - y0) / (y1 - y0)

        return (q00*(1-tx)*(1-ty) + q01*tx*(1-ty) + q10*(1-tx)*ty + q11*tx*ty)

    def pix2ll(row, col):
        lat = bilinear(col, row, cols, rows, lat_grid)
        lon = bilinear(col, row, cols, rows, lon_grid)
        return float(lat), float(lon)

    return pix2ll


@njit(cache=True)
def erase_edge_touching_polygons_numba(polygons, background=-1):
    """
    Perform a flood_fill from the edges to find all polygons that touch the edge
    Set all edge-touching polygons to background value
    """

    rows, cols = polygons.shape
    if rows == 0 or cols == 0:
        return

    visited = np.zeros((rows, cols), dtype=np.uint8)

    # Border stack (linear indices)
    border_stack = np.empty(rows * cols, dtype=np.int32)
    bsp = 0

    # push border cells (duplicates ok; visited will skip)
    # store as linear indices, can decompose to (r, c) by r = idx // cols, c = idx % cols
    for c in range(cols):
        border_stack[bsp] = 0 * cols + c; bsp += 1
        border_stack[bsp] = (rows - 1) * cols + c; bsp += 1
    for r in range(rows):
        border_stack[bsp] = r * cols + 0; bsp += 1
        border_stack[bsp] = r * cols + (cols - 1); bsp += 1

    # Component DFS stack + component storage (both linear indices)
    comp_stack = np.empty(rows * cols, dtype=np.int32)
    comp_cells = np.empty(rows * cols, dtype=np.int32)

    # flood-fill from each border cell
    while bsp > 0:
        bsp -= 1
        idx = border_stack[bsp]
        r = idx // cols
        c = idx - r * cols

        if visited[r, c] == 1:
            continue

        val = polygons[r, c]
        visited[r, c] = 1

        if val == background:
            continue

        # DFS the component of same `val`
        csp = 0
        comp_stack[csp] = idx
        csp += 1

        keep = 0  # number of cells in this component

        while csp > 0:
            csp -= 1
            cur = comp_stack[csp]
            x = cur // cols
            y = cur - x * cols

            if visited[x, y] == 1 and (x != r or y != c):
                # already processed from another seed
                continue

            if polygons[x, y] != val:
                continue

            visited[x, y] = 1
            comp_cells[keep] = cur
            keep += 1

            # push neighbors
            if x > 0 and visited[x - 1, y] == 0:
                comp_stack[csp] = (x - 1) * cols + y; csp += 1
            if x < rows - 1 and visited[x + 1, y] == 0:
                comp_stack[csp] = (x + 1) * cols + y; csp += 1
            if y > 0 and visited[x, y - 1] == 0:
                comp_stack[csp] = x * cols + (y - 1); csp += 1
            if y < cols - 1 and visited[x, y + 1] == 0:
                comp_stack[csp] = x * cols + (y + 1); csp += 1

        # Component is border-reachable => erase it
        for i in range(keep):
            pos = comp_cells[i]
            x = pos // cols
            y = pos - x * cols
            polygons[x, y] = background

def decimal_to_dms(decimal_degree, is_latitude=True):
    """Convert decimal degrees to degrees, minutes, seconds (DMS) format."""
    try:
        if not isinstance(decimal_degree, (int, float)):
            raise ValueError("Coordinate must be a number.")

        # Determine hemisphere
        if is_latitude:
            hemisphere = 'N' if decimal_degree >= 0 else 'S'
        else:
            hemisphere = 'E' if decimal_degree >= 0 else 'W'

        # Absolute value for calculation
        abs_val = abs(decimal_degree)

        # Degrees
        degrees = int(abs_val)
        # Minutes
        minutes_full = (abs_val - degrees) * 60
        minutes = int(minutes_full)
        # Seconds
        seconds = (minutes_full - minutes) * 60

        return f"{degrees}°{minutes}'{seconds:.2f}\" {hemisphere}"
    except Exception as e:
        return f"Error: {e}"
