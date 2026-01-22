'''
Rendering functions to handle image display after zooming and panning

Last modified: Jan 2026
'''
import numpy as np
from skimage.morphology import binary_dilation
import cv2
from core.utils import apply_brightness

# TO DO: Optimize as it is the bottleneck for performance when changing contrast/brightness and during panning/zooming
def crop_resize(pred, img, boundmask, landmask, local_boundmask, zoom_factor, offset_x, offset_y, brightness,canvas_width, canvas_height, show_local_segmentation):
    #crop = self.get_zoomed_region(self.pred)
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
    img_crop = img[view_top:view_bottom, view_left:view_right].astype(np.float32)
    boundmask_crop = boundmask[view_top:view_bottom, view_left:view_right]
    landmask_crop = landmask[view_top:view_bottom, view_left:view_right]

    # Determine canvas display size
    zoomed_width = max(1, int((view_right - view_left) * zoom_factor))
    zoomed_height = max(1, int((view_bottom - view_top) * zoom_factor))

    # Probably best to group these as well when returning and in visualizer
    pred_resized = cv2.resize(pred_crop, (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST)
    img_resized = cv2.resize(img_crop, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
    boundmask_resized = cv2.resize(boundmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    landmask_resized = cv2.resize(landmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)

    boundmask_resized = np.uint8(binary_dilation(boundmask_resized.astype('uint8'), np.ones((2,2)).astype('uint8')))

    # Adjust where the image is drawn (canvas position)
    draw_x = int(offset_x + view_left * zoom_factor)
    draw_y = int(offset_y + view_top * zoom_factor)

    img_resized = apply_brightness(img_resized, brightness, clip=True)

    if show_local_segmentation and local_boundmask is not None:
        local_boundmask_crop = local_boundmask[view_top:view_bottom, view_left:view_right]
        local_boundmask_resized = cv2.resize(local_boundmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        local_boundmask_resized = None

    return pred_resized, img_resized, boundmask_resized, landmask_resized, local_boundmask_resized, draw_x, draw_y

# Next step is to combine zoom factor, offset, etc into a state variable
# for canvas_width get from self.canvas.winfo_width()
# for canvas_height get from self.canvas.winfo_height() in visualizer when calling this function
def get_zoomed_region(image, zoom_factor, offset_x, offset_y, canvas_width, canvas_height):
    h, w = image.shape[:2]
    
    # Image coordinates of the viewport
    img_left = max(0, int(-offset_x / zoom_factor))
    img_top = max(0, int(-offset_y / zoom_factor))
    img_right = min(w, int((canvas_width - offset_x) / zoom_factor))
    img_bottom = min(h, int((canvas_height - offset_y) / zoom_factor))

    if img_right <= img_left or img_bottom <= img_top:
        return None

    return img_top, img_bottom, img_left, img_right

# Should we put set_overlay here as well as it counts as rendering, keep display_image in visualizer

def layer_imagery(HH_img, HV_img, stack="(HH, HH, HV)"):
    HH_img = HH_img[:, :, 0]
    HV_img = HV_img[:, :, 0]
    if stack == "(HH, HH, HV)":
        layered_img = np.stack([HH_img, HH_img, HV_img], axis=-1)
    else: # "(HH, HV, HV)"
        layered_img = np.stack([HH_img, HV_img, HV_img], axis=-1)

    # print(layered_img.shape)
    return layered_img