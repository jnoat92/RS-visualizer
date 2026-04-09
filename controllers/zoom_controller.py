'''
ZoomController manages zooming and panning interactions on the image 
canvas. It handles mouse events to allow users to zoom in/out and pan 
around the image, ensuring that the display updates accordingly. 
This controller interacts closely with the DisplayController to adjust 
the view based on user input.

Last modified: Apr 2026
'''

# Move enable_zoom_selection, zoom_to_rectangle, reset_zoom
# Most of _on_mousewheel
# pan parts of _on_left_drag

class ZoomController:
    def __init__(self, deps, display_controller, annotation_controller):
        self.deps = deps
        self.display_controller = display_controller
        self.annotation_controller = annotation_controller

    def enable_zoom_selection(self):
        """
        Enable zoom selection mode, change cursor to crosshair, and update button appearance.
        """
        view = self.deps.app_state.view
        overlay = self.deps.app_state.overlay
        if not overlay.select_local_segmentation: # If not in local segmentation mode perform zoom selection
            view.zoom_select_mode = True
            self.deps.widgets["zoom_select_btn"].configure(**self.deps.widgets["zoom_btn_active_style"])
        self.deps.canvas.config(cursor="crosshair")

    def zoom_to_rectangle(self, x_min, y_min, x_max, y_max):
        """
        Zoom the view to fit the rectangle drawn by the user, update the view's zoom factor
        and offsets accordingly, and refresh the display.
        """

        view = self.deps.app_state.view
        rect_width = x_max - x_min
        rect_height = y_max - y_min

        canvas_width = self.deps.canvas.winfo_width()
        canvas_height = self.deps.canvas.winfo_height()

        zoom_x = canvas_width / rect_width
        zoom_y = canvas_height / rect_height
        view.zoom_factor = min(zoom_x, zoom_y, view.max_zoom)

        center_x = x_min + rect_width / 2
        center_y = y_min + rect_height / 2

        view.offset_x = int(canvas_width / 2 - center_x * view.zoom_factor)
        view.offset_y = int(canvas_height / 2 - center_y * view.zoom_factor)

        self.display_controller.refresh_view()
        if self.deps.app_state.anno.polygon_points_img_coor: 
            self.annotation_controller.draw_polygon_on_canvas()

    def reset_zoom(self):
        """
        Reset the zoom to fit the entire image in the canvas,
        center the image, and refresh the display.
        """

        view = self.deps.app_state.view
        scene = self.deps.app_state.scene
        # Get canvas dimensions
        canvas_width = self.deps.canvas.winfo_width()
        canvas_height = self.deps.canvas.winfo_height()

        # Use overlay or base image to get image size
        img_height, img_width = scene.img.shape[:2]

        # Compute scale to fit the whole image
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        view.zoom_factor = min(scale_x, scale_y)

        # Center image in canvas
        new_width = int(img_width * view.zoom_factor)
        new_height = int(img_height * view.zoom_factor)
        view.offset_x = (canvas_width - new_width) // 2
        view.offset_y = (canvas_height - new_height) // 2

        self.display_controller.refresh_view()
        if self.deps.app_state.anno.polygon_points_img_coor:
            self.annotation_controller.draw_polygon_on_canvas()