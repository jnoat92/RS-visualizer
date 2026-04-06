'''
DisplayController manages the display of the image including rendering 
the main image, segmentation overlay, boundaries, and landmask.
It houses the refresh_view function which composites the final image 
based on the current state and settings.

Last modified: Mar 2026
'''

from dataclasses import dataclass
from PIL import Image, ImageTk
import tkinter as tk

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


class DisplayController:
    def __init__(self, deps):
        self.deps = deps
        self.tk_image = None
        self.render_result: RenderResult | None = None

    def set_overlay(self):
        """
        Set the overlay for the current scene based on the current predictions, boundaries, landmask, local seg, and opacity.
        """
        self.overlay = compose_overlay(self.pred_resized, self.img_resized, self.boundmask_resized, self.landmask_resized, 
                                    self.local_boundmask_resized, self.deps.app_state.overlay.alpha)

    def choose_image(self):
        """
        Update the displayed image and minimap based on the current active label.
        """
        scene = self.deps.app_state.scene
        display = self.deps.app_state.display
        scene.img = scene.color_composites[display.channel_mode]
        custom_anno = "Custom_Annotation"

        # Check if custom annotation exists and if user wants to show it on minimap
        if custom_anno in scene.lbl_sources and self.deps.widgets['show_prev_anno_switch'].get():
            changed_area_mask = scene.predictions[custom_anno][:,:,0] != scene.predictions[scene.lbl_sources[0]][:,:,0]
            self.deps.minimap.show_changed_area(scene.img, changed_area_mask)
        else:
            self.deps.minimap.set_image(scene.img)

    def display_image(self):
        """
        Display the current image with overlay on the canvas.
        """
        image = self.overlay if self.deps.app_state.overlay.show_overlay else self.img_resized.astype('uint8')

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image))

        self.deps.canvas.delete("main_image")  # Remove previous image
        self.deps.canvas.create_image(self.draw_x, self.draw_y, anchor=tk.NW, image=self.tk_image, tags=("main_image"))

    
    def refresh_view(self):
        """
        Refresh the displayed image and minimap viewport based on the current view settings (zoom, pan) and display settings (contrast, brightness).
        """

        view = self.deps.app_state.view
        scene = self.deps.app_state.scene
        display = self.deps.app_state.display
        overlay = self.deps.app_state.overlay
        # NEXT STEP: Group the returns, and optimize crop_resize, right now it's the bottleneck for performance on contrast change
        self.pred_resized, self.img_resized, self.boundmask_resized, self.landmask_resized, self.local_boundmask_resized, self.draw_x, self.draw_y = crop_resize(
                    scene.predictions[scene.active_source], scene.img, scene.boundmasks[scene.active_source], scene.land_nan_masks[scene.active_source], 
                    overlay.local_segmentation_bounds, scene.nan_mask["HH"], view.zoom_factor, view.offset_x, view.offset_y, display.brightness,
                    self.deps.canvas.winfo_width(), self.deps.canvas.winfo_height(), overlay.show_local_segmentation)
        self.set_overlay()
        self.display_image()

        # Update minimap viewport
        self.deps.minimap.set_viewport_rect(scene.img, view.zoom_factor, view.offset_x, view.offset_y, self.deps.canvas.winfo_width(), self.deps.canvas.winfo_height())

    def refresh_all(self):
        self.refresh_view()

        if self.deps.app_state.anno.polygon_points_img_coor: 
            self.deps.app.draw_polygon_on_canvas()

        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():
                self.deps.annotation_panel.update_zoomed_display()