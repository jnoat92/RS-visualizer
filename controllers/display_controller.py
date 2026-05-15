from PIL import Image, ImageTk
import tkinter as tk


class DisplayController:
    def __init__(self, deps, display_viewmodel):
        self.deps = deps
        self.display_viewmodel = display_viewmodel
        self.tk_image = None

    def set_overlay(self):
        """
        Set the overlay for the current scene based on the current predictions, boundaries, landmask, local seg, and opacity.
        """
        self.display_viewmodel.update_overlay_only()

    def choose_image(self):
        """
        Update the displayed image and minimap based on the current active label.
        """
        img = self.display_viewmodel.choose_image()
        _, changed_area_mask = self.display_viewmodel.get_minimap_data(
            self.deps.widgets['show_prev_anno_switch'].get()
        )
        if changed_area_mask is not None:
            self.deps.minimap.show_changed_area(img, changed_area_mask)
        else:
            self.deps.minimap.set_image(img)

    def display_image(self):
        """
        Display the current image with overlay on the canvas.
        """
        image = self.display_viewmodel.current_display_image()
        if image is None:
            return

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image))

        self.deps.canvas.delete("main_image")  # Remove previous image
        render_result = self.display_viewmodel.render_result
        self.deps.canvas.create_image(render_result.draw_x, render_result.draw_y, anchor=tk.NW, image=self.tk_image, tags=("main_image"))

    
    def refresh_view(self):
        """
        Refresh the displayed image and minimap viewport based on the current view settings (zoom, pan) and display settings (contrast, brightness).
        """

        scene = self.deps.app_state.scene
        view = self.deps.app_state.view
        render_result = self.display_viewmodel.render(
            self.deps.canvas.winfo_width(),
            self.deps.canvas.winfo_height(),
        )
        if render_result is None:
            return
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
