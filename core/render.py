'''
Rendering functions to handle image display after zooming and panning

Last modified: Mar 2026
'''
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window
from skimage.morphology import binary_dilation
import cv2
from core.utils import apply_brightness

# TO DO: Optimize as it is the bottleneck for performance when changing contrast/brightness and during panning/zooming
def crop_resize(pred, img, boundmask, landmask, 
                local_boundmask, nan_mask, zoom_factor, 
                offset_x, offset_y, brightness, canvas_width, 
                canvas_height, show_local_segmentation, sar_img, band_stack):
    """
    Crop the image to the current viewport, resize it according to the zoom factor, 
    and apply brightness adjustment.
    Handles special cases for landmask, boundmask, local_boundmask, 
    and nan_mask to ensure correct rendering of different areas 
    (e.g., land, boundaries, local segmentation boundaries, and NaN areas).
    """

    print(zoom_factor)
    zoom_factor = max(0.01, zoom_factor)  # Prevent division by zero or negative zoom
    
    crop = get_zoomed_region(pred, zoom_factor, offset_x, offset_y, canvas_width, canvas_height)
    if crop is None:
        return
    
    view_top, view_bottom, view_left, view_right = crop
    h, w = pred.shape[:2]

    # Clamp values to image bounds
    view_top = max(0, min(h-1, view_top))
    view_bottom = max(0, min(h, view_bottom))
    view_left = max(0, min(w-1, view_left))
    view_right = max(0, min(w, view_right))

    if view_bottom <= view_top or view_right <= view_left:
        return  # invalid crop

    pred_crop = pred[view_top:view_bottom, view_left:view_right].astype(np.float32)
    #img_crop = img[view_top:view_bottom, view_left:view_right].astype(np.float32)
    boundmask_crop = boundmask[view_top:view_bottom, view_left:view_right]
    landmask_crop = landmask[view_top:view_bottom, view_left:view_right]
    nan_mask_crop = nan_mask[view_top:view_bottom, view_left:view_right]

    # Determine canvas display size
    zoomed_width = max(1, int((view_right - view_left) * zoom_factor))
    zoomed_height = max(1, int((view_bottom - view_top) * zoom_factor))

    # Probably best to group these as well when returning and in visualizer
    pred_resized = cv2.resize(pred_crop, (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST)
    #img_resized = cv2.resize(img_crop, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
    boundmask_resized = cv2.resize(boundmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    landmask_resized = cv2.resize(landmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    nan_mask_resized = cv2.resize(nan_mask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)

    # For boundmask only, use dilation to make it more visible at lower zoom levels
    boundmask_resized = np.uint8(binary_dilation(boundmask_resized.astype('uint8'), np.ones((2,2)).astype('uint8')))

    # Adjust where the image is drawn (canvas position)
    draw_x = int(offset_x + view_left * zoom_factor)
    draw_y = int(offset_y + view_top * zoom_factor)

    #img_resized = apply_brightness(img_resized, nan_mask_resized, brightness, clip=True)

    if show_local_segmentation and local_boundmask is not None:
        local_boundmask_crop = local_boundmask[view_top:view_bottom, view_left:view_right]
        local_boundmask_resized = cv2.resize(local_boundmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        local_boundmask_resized = None

    display_img = read_band_window(sar_img, band_stack, view_left, view_top, view_right, view_bottom, 10.0, canvas_width, canvas_height)
    display_img = cv2.resize(display_img, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
    display_img = apply_brightness(display_img, nan_mask_resized, brightness, clip=True)

    #print(f"Windowed image shape: {display_img.shape}, Display image shape: {img_resized.shape}")
    #print(f"Pred resized shape: {pred_resized.shape}, Boundmask resized shape: {boundmask_resized.shape}, Landmask resized shape: {landmask_resized.shape}, Local boundmask resized shape: {local_boundmask_resized.shape if local_boundmask_resized is not None else None}")

    return pred_resized, display_img, boundmask_resized, landmask_resized, local_boundmask_resized, draw_x, draw_y

# Next step is to combine zoom factor, offset, etc into a state variable
# for canvas_width get from self.canvas.winfo_width()
# for canvas_height get from self.canvas.winfo_height() in visualizer when calling this function
def get_zoomed_region(image, zoom_factor, offset_x, offset_y, canvas_width, canvas_height):
    """Calculate the coordinates of the zoomed region of the image based on the current zoom factor and offsets."""
    h, w = image.shape[:2]
    
    # Image coordinates of the viewport
    img_left = max(0, int(-offset_x / zoom_factor))
    img_top = max(0, int(-offset_y / zoom_factor))
    img_right = min(w, int((canvas_width - offset_x) / zoom_factor))
    img_bottom = min(h, int((canvas_height - offset_y) / zoom_factor))

    if img_right <= img_left or img_bottom <= img_top:
        return None
    
    print(f"Zoomed region in image coordinates: top={img_top}, bottom={img_bottom}, left={img_left}, right={img_right}")

    return img_top, img_bottom, img_left, img_right

# Should we put set_overlay here as well as it counts as rendering, keep display_image in visualizer

def layer_imagery(HH_img, HV_img, stack="(HH, HH, HV)"):
    """
    Layer the HH and HV images into a single RGB image for display.
    The stack parameter determines how the channels are combined:
    """
    HH_img = HH_img[:, :, 0]
    HV_img = HV_img[:, :, 0]
    if stack == "(HH, HH, HV)":
        layered_img = np.stack([HH_img, HH_img, HV_img], axis=-1)
    else: # "(HH, HV, HV)"
        layered_img = np.stack([HH_img, HV_img, HV_img], axis=-1)

    # print(layered_img.shape)
    return layered_img

def read_band_window(
    sar_img,
    band_stack: list[str],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    scale_factor: float,
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    """
    Read a window from one band and resample it to the requested output size.

    Inputs:
        band_name:
            Logical band name, e.g. "HH" or "HV".

        x_min, y_min, x_max, y_max:
            Raster pixel coordinates describing the source window.

        scale_factor:
            Factor by which to scale the output array.

    Output:
        2D float32 NumPy array with shape out_height x out_width.
    """
    ds = sar_img
    band_index_hh = 1
    band_index_hv = 2

    arr_hh = None
    arr_hv = None

    # Size of the output array after scaling
    out_width = max(1, int((x_max - x_min)))
    out_height = max(1, int((y_max - y_min)))

    # Check if out ratio is same as canvas ratio
    # If yes, we can directly scale to canvas size for better visualization
    if abs((out_width / out_height) - (canvas_width / canvas_height)) < 0.01:
        out_width = canvas_width
        out_height = canvas_height


    # Size of the window to read from the source image, scaled by the factor
    x_min = max(0, min(ds.width, x_min * scale_factor))
    x_max = max(0, min(ds.width, x_max * scale_factor))
    y_min = max(0, min(ds.height, y_min * scale_factor))
    y_max = max(0, min(ds.height, y_max * scale_factor))

    print(x_min, y_min, x_max, y_max)

    if x_max <= x_min or y_max <= y_min:
        return np.zeros((max(1, out_height), max(1, out_width)), dtype=np.float32)

    window = Window.from_slices(
        rows=(int(y_min), int(np.ceil(y_max))),
        cols=(int(x_min), int(np.ceil(x_max))),
    )

    if "HH" in band_stack:
        arr_hh = get_band_array(ds, band_index_hh, window, out_height, out_width)

    if "HV" in band_stack:
        arr_hv = get_band_array(ds, band_index_hv, window, out_height, out_width)

    # Constuct the RGB array based on the requested band stack
    if band_stack == ["HH"]:
        rgb = np.dstack([arr_hh, arr_hh, arr_hh])
    elif band_stack == ["HV"]:
        rgb = np.dstack([arr_hv, arr_hv, arr_hv])
    elif band_stack == ["HH", "HH", "HV"]:
        rgb = np.dstack([arr_hh, arr_hh, arr_hv])
    elif band_stack == ["HH", "HV", "HV"]:
        rgb = np.dstack([arr_hh, arr_hv, arr_hv])

    # plt.imshow(rgb)
    # plt.show(block=False)
    return rgb

def get_band_array(ds, band_index, window, out_height, out_width):
    arr = ds.read(
        band_index,
        window=window,
        out_shape=(max(1, out_height), max(1, out_width)),
        masked=True,
    )

    arr = np.asarray(arr.filled(np.nan), dtype=np.float32)

    sample_width = min(1024, ds.width)
    sample_height = min(1024, ds.height)

    full_arr = ds.read(
        band_index,
        out_shape=(sample_height, sample_width),
        masked=True,
    )

    full_arr = np.asarray(full_arr.filled(np.nan), dtype=np.float32)

    finite = full_arr[np.isfinite(full_arr)]

    if finite.size == 0:
        display_range = (0.0, 1.0)
    else:
        low = float(np.nanpercentile(finite, 0))
        high = float(np.nanpercentile(finite, 100))

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(finite))
            high = float(np.nanmax(finite))

        if high <= low:
            high = low + 1.0

        display_range = (low, high)

    # Change this later
    display_min = display_range[0]
    display_max = display_range[1]

    normalized = (arr - display_min) / (display_max - display_min)
    gray = normalized * 255.0

    gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray