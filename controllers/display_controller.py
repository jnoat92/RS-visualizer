'''
DisplayController manages the display of the image including rendering 
the main image, segmentation overlay, boundaries, and landmask.
It houses the refresh_view function which composites the final image 
based on the current state and settings.

Last modified: Mar 2026
'''

from dataclasses import dataclass
from PIL import Image, ImageTk

from core.overlay import compose_overlay
from core.render import crop_resize

@dataclass
class RenderResult:
    """Data class to hold the results of the rendering process."""
    pred_resized: object
    img_resized: object
    boundmask_resized: object
    landmask_resized: object
    local_boundmask_resized: object
    draw_x: int
    draw_y: int

# Move set_overlay, choose_image, display_image, and refresh_view

class DisplayController:
    def __init__(self, visualizer):
        self.visualizer = visualizer