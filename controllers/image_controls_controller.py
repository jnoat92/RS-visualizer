'''
ImageControlsController manages the interactions between the image 
controls (like sliders, buttons) and the display, scene, and annotation 
controllers. It handles user inputs related to image manipulation, such 
as adjusting contrast, adjusting brightness, toggling overlay, and 
selecting channel. 

They share the "UI control changed -> update state -> rerender" flow.

Last modified: Mar 2026
'''

from core.render import layer_imagery
from core.contrast_handler import enhance_outlier_slider

# Move color_composite, HH_HV, contrast_slider_handle, right_click_contrast_rest
# brightness_slider_handle, right_click_brightness_reset, opacity_slider_handle, segmentation_toggle

class ImageControlsController:
    def __init__(self, deps, display_controller, 
                 scene_controller, annotation_controller):
        self.deps = deps
        self.display = display_controller
        self.scene = scene_controller
        self.annotation = annotation_controller