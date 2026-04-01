'''
CanvasEventsController: Handles mouse events on the canvas, including 
clicks, double-clicks, and movements. It interacts with the display, 
zoom, annotation, and panel controllers to perform actions based on 
user input.

Last modified: Apr 2026
'''

import numpy as np
from rasterio.transform import xy
from tkinter import messagebox
from core.segmentation import get_segment_contours
from core.utils import ds_to_src_pixel, decimal_to_dms

class CanvasEventsController:
    '''Controller for handling canvas events such as mouse clicks and movements.'''
    def __init__(self, deps, display_controller, 
                 annotation_controller):
        self.deps = deps
        self.display_controller = display_controller
        self.annotation_controller = annotation_controller
        self.double_click_flag = False
        self.selection_start_coord = None
        self.selection_rect_id = None

    def _on_mousewheel(self, event):
        """Handle mouse wheel events for zooming in and out."""
        view = self.deps.app_state.view
        scale = 1.1 if event.delta > 0 or event.num == 4 else 1 / 1.1
        old_zoom = view.zoom_factor
        new_zoom = max(view.min_zoom, min(view.max_zoom, old_zoom * scale))

        if new_zoom == old_zoom:
            return  # no change

        # Mouse position in canvas
        canvas_x = self.deps.canvas.canvasx(event.x)
        canvas_y = self.deps.canvas.canvasy(event.y)

        # Convert to image coordinates before zoom
        img_x = (canvas_x - view.offset_x) / old_zoom
        img_y = (canvas_y - view.offset_y) / old_zoom

        # Update zoom
        view.zoom_factor = new_zoom

        # Adjust offsets so the image pixel under the cursor stays at the same canvas position
        view.offset_x = canvas_x - img_x * view.zoom_factor
        view.offset_y = canvas_y - img_y * view.zoom_factor

        self.display_controller.refresh_view()
        if self.deps.app_state.anno.polygon_points_img_coor:
            self.annotation_controller.draw_polygon_on_canvas()

    def _on_left_click_await(self, event):
        """Handle left mouse click with differentiation between single and double clicks."""
        anno = self.deps.app_state.anno
        view = self.deps.app_state.view
        if view.zoom_select_mode or anno.annotation_mode in ['rectangle', 'polygon'] \
            or self.deps.app_state.overlay.select_local_segmentation:
            self._on_left_click(event)
        else:
            self.deps.app.after(180, lambda: self.choose_click_event(event))
            self.double_click_flag = False

    def choose_click_event(self, event):
        """Determine whether the click was a single or double click and call the appropriate handler."""
        if self.double_click_flag:
            self._on_double_click(event)
            self.double_click_flag = False
        else:
            self._on_left_click(event)

    def _on_left_click(self, event):
        """Handle left mouse click for zoom selection, panning, rectangle, polygon drawing. or bucket fill."""
        view = self.deps.app_state.view
        anno = self.deps.app_state.anno
        overlay = self.deps.app_state.overlay
        if view.zoom_select_mode:
            # Start selection
            self.selection_start_coord = (event.x, event.y)
            self.selection_rect_id = self.deps.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=2)
        elif overlay.select_local_segmentation:
            # Start selection for local segmentation
            self.selection_start_coord = (event.x, event.y)
            self.selection_rect_id = self.deps.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='blue', width=2)
        elif anno.annotation_mode == 'rectangle':
                self.selection_start_coord = (event.x, event.y)
                anno.selected_polygon = self.deps.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='yellow', width=2)
        elif anno.annotation_mode == 'polygon':
                self._add_polygon_point(event)
        elif anno.annotation_mode == 'bucket_fill':
            x = round(np.float64((event.x - view.offset_x) / view.zoom_factor))
            y = round(np.float64((event.y - view.offset_y) / view.zoom_factor))
            if anno.selected_polygon_area_idx is None or \
                (y,x) not in zip(anno.selected_polygon_area_idx[0], anno.selected_polygon_area_idx[1]):
                self._on_double_click(event) # Use double-click handler for bucket fill
            else:
                self.annotation_controller.bucket_fill_polygon_area(event)
        else:
            # Start pan
            view.pan_start_screen = (event.x, event.y)

    def _on_left_drag(self, event):
        """Handle mouse drag for zoom selection, panning, or rectangle drawing for polygon or local segmentation."""

        view = self.deps.app_state.view
        scene = self.deps.app_state.scene
        anno = self.deps.app_state.anno
        overlay = self.deps.app_state.overlay

        if scene.img is None:
            return # No image loaded, ignore drag

        if view.zoom_select_mode and self.selection_start_coord:
            # Update selection rectangle
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y
            self.deps.canvas.coords(self.selection_rect_id, x0, y0, x1, y1)
        elif overlay.select_local_segmentation and self.selection_start_coord:
            # Update selection rectangle for local segmentation
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y
            self.deps.canvas.coords(self.selection_rect_id, x0, y0, x1, y1)
        elif anno.annotation_mode == 'rectangle' and self.selection_start_coord:
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y
            self.deps.canvas.coords(anno.selected_polygon, x0, y0, x1, y1)
        elif view.pan_start_screen:
            # Pan mode
            dx = event.x - view.pan_start_screen[0]
            dy = event.y - view.pan_start_screen[1]
            view.offset_x += dx
            view.offset_y += dy
            view.pan_start_screen = (event.x, event.y)
            
            self.display_controller.refresh_view()
            if anno.polygon_points_img_coor: 
                self.annotation_controller.draw_polygon_on_canvas()

    def _on_left_release(self, event):
        """Handle left mouse button release to finalize zoom selection, rectangle, or polygon drawing."""	
        
        view = self.deps.app_state.view
        scene = self.deps.app_state.scene
        overlay = self.deps.app_state.overlay
        anno = self.deps.app_state.anno
        if anno.annotation_mode == 'bucket_fill':
            self.deps.canvas.config(cursor="spraycan")
        elif anno.annotation_mode == 'polygon':
            self.deps.canvas.config(cursor="crosshair")
        else:
            self.deps.canvas.config(cursor="")

        if view.zoom_select_mode and self.selection_start_coord:
            # Complete selection and zoom
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y

            # Reset variables
            self.deps.canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None
            self.selection_start_coord = None
            view.zoom_select_mode = False
            self.deps.widgets["zoom_select_btn"].configure(**self.deps.widgets["zoom_btn_default_style"])

            # Convert canvas to image coords
            x_min = min(x0, x1)
            y_min = min(y0, y1)
            x_max = max(x0, x1)
            y_max = max(y0, y1)

            if x_max - x_min < 10 or y_max - y_min < 10:
                return  # too small

            img_x_min = int((x_min - view.offset_x) / view.zoom_factor)
            img_y_min = int((y_min - view.offset_y) / view.zoom_factor)
            img_x_max = int((x_max - view.offset_x) / view.zoom_factor)
            img_y_max = int((y_max - view.offset_y) / view.zoom_factor)

            img_x_min = max(0, img_x_min)
            img_y_min = max(0, img_y_min)
            img_x_max = min(scene.img.shape[1], img_x_max)
            img_y_max = min(scene.img.shape[0], img_y_max)

            if img_x_max < 0 or img_y_max < 0 or img_x_min < 0 or img_y_min < 0:
                return  # invalid selection

            self.deps.app.zoom_to_rectangle(img_x_min, img_y_min, img_x_max, img_y_max)

        elif overlay.select_local_segmentation and self.selection_start_coord:
            # Complete selection and zoom
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y

            # Convert canvas to image coords
            x_min = min(x0, x1)
            y_min = min(y0, y1)
            x_max = max(x0, x1)
            y_max = max(y0, y1)

            if x_max - x_min < 10 or y_max - y_min < 10:
                return  # too small

            img_x_min = int((x_min - view.offset_x) / view.zoom_factor)
            img_y_min = int((y_min - view.offset_y) / view.zoom_factor)
            img_x_max = int((x_max - view.offset_x) / view.zoom_factor)
            img_y_max = int((y_max - view.offset_y) / view.zoom_factor)

            img_x_min = max(0, img_x_min)
            img_y_min = max(0, img_y_min)
            img_x_max = min(scene.img.shape[1], img_x_max)
            img_y_max = min(scene.img.shape[0], img_y_max)

            if img_x_max < 0 or img_y_max < 0 or img_x_min < 0 or img_y_min < 0:
                return  # invalid selection
            
            overlay.local_segmentation_limits = (img_x_min, img_y_min, img_x_max, img_y_max)

            self.deps.app.run_local_segmentation(img_x_min, img_y_min, img_x_max, img_y_max)
        
        elif anno.annotation_mode == 'rectangle' and self.selection_start_coord:
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y

            polygon_points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # Rectangle points
            for x, y in polygon_points:
                anno.polygon_points_img_coor.append((int((x - view.offset_x) / view.zoom_factor),
                                                     int((y - view.offset_y) / view.zoom_factor)))
            self.annotation_controller.finish_polygon()
            
            # Reset variables
            self.selection_start_coord = None

        elif view.pan_start_screen:
            view.pan_start_screen = None  # end pan

    def _on_right_click(self, event):
        """Handle right-click to finish polygon drawing."""
        if self.deps.app_state.anno.annotation_mode == 'polygon':
            self.annotation_controller.finish_polygon()

    def _on_double_click_set_flag(self, event):
        """Set flag on double click to differentiate between single and double clicks."""
        self.double_click_flag = True

    def _on_double_click(self, event):
        """
        Handle double-click to select polygon area, check if selection is within bounds 
        and local segmentation area, draw polygon on canvas, or bucket fill if in bucket fill mode.
        """
        view = self.deps.app_state.view
        scene = self.deps.app_state.scene
        anno = self.deps.app_state.anno
        overlay = self.deps.app_state.overlay
        if self.deps.annotation_window.winfo_viewable():

            if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
                self.deps.annotation_panel.zoom_window is not None and 
                self.deps.annotation_panel.zoom_window.winfo_exists()):
                if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                    self.deps.annotation_panel.zoom_window.destroy()

            self.annotation_controller.reset_annotation()

            x = int((event.x - view.offset_x) / view.zoom_factor)
            y = int((event.y - view.offset_y) / view.zoom_factor)

            # Check if selection is outside local segmentation area
            if overlay.show_local_segmentation:
                if x < overlay.local_segmentation_limits[0] or x >= overlay.local_segmentation_limits[2] or \
                   y < overlay.local_segmentation_limits[1] or y >= overlay.local_segmentation_limits[3]:
                    result = messagebox.askyesno("Selection Out of Local Segmentation Area", "Selecting outside of local segmentation view will close the local segmentation view.\n\nClose local segmentation view?", parent=self.deps.app.master)
                    if result: # Close local segmentation view
                        overlay.show_local_segmentation = False
                        self.display_controller.refresh_view()
                    else:
                        return

            h, w = scene.predictions[scene.active_source].shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                return
            
            if overlay.show_local_segmentation:
                # Change scene.predictions to local irgs for unsupervised segmentation
                contours, mask = get_segment_contours(overlay.local_segmentation_mask, y, x)
                selected_outside = False
                # Check if selected segment includes border region, may not be needed with edge-touching polygons removed
                for i in range(len(contours)):
                    if selected_outside:
                        break
                    for j in range(len(contours[i])):
                        if contours[i][j][1] <= overlay.local_segmentation_limits[0]-0.5 or contours[i][j][1] >= overlay.local_segmentation_limits[2]+0.5 or \
                            contours[i][j][0] <= overlay.local_segmentation_limits[1]-0.5 or contours[i][j][0] >= overlay.local_segmentation_limits[3]+0.5:
                            result = messagebox.askyesno("Selection Not Part of Local Segmentation", "Selecting a polygon not part of the local segmentation view will close the local segmentation view.\n\nClose local segmentation view?", parent=self.deps.app.master)
                            
                            if result: # Close local segmentation view and get contours from prediction
                                overlay.show_local_segmentation = False
                                selected_outside = True
                                self.display_controller.refresh_view()
                                # Change to scene.predictions for selection
                                contours, mask = get_segment_contours(scene.predictions[scene.active_source], y, x)
                                break
                            else:
                                return
            else:
                contours, mask = get_segment_contours(scene.predictions[scene.active_source], y, x)

            # select polygon area on image
            anno.selected_polygon_area_idx = [(y, x) for y, x in zip(*np.where(mask))]
            img_y_min = np.asarray(anno.selected_polygon_area_idx)[:,0].min()
            img_y_max = np.asarray(anno.selected_polygon_area_idx)[:,0].max()
            img_x_min = np.asarray(anno.selected_polygon_area_idx)[:,1].min()
            img_x_max = np.asarray(anno.selected_polygon_area_idx)[:,1].max()
            anno.selected_polygon_window = (img_y_min, img_y_max, img_x_min, img_x_max)
            anno.selected_polygon_area_idx = tuple(zip(*anno.selected_polygon_area_idx))

            # Check if selected area is all land/nan
            if scene.land_nan_masks[scene.active_source][anno.selected_polygon_area_idx].all():
                self.annotation_controller.reset_annotation()
                return
            
            # draw polygon(s) on canvas
            anno.polygon_points_img_coor = [[(x, y) for y, x in c] for c in contours]
            anno.multiple_polygons = True
            self.annotation_controller.draw_polygon_on_canvas()

            # If in bucket fill mode and double clicked
            if anno.annotation_mode == 'bucket_fill' and self.double_click_flag:
                self.annotation_controller.bucket_fill_polygon_area(event)
                self.double_click_flag = False

    def _on_mouse_move(self, event):
        """
        Handle mouse move events to display lat/lon coordinates in the status bar 
        based on the current mouse position, check if the coordinates are valid, 
        and convert to DMS format for display.
        """
        view = self.deps.app_state.view
        scene = self.deps.app_state.scene
        x = int((event.x - view.offset_x) / view.zoom_factor)
        y = int((event.y - view.offset_y) / view.zoom_factor)

        # Check if coordinates are nan or out of bounds
        if scene.img is not None:
            h, w = scene.predictions[scene.active_source].shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                self.deps.status_bar.configure(text=f"Lat: N/A, Lon: N/A")
            elif scene.nan_mask["HH"][y, x]:
                self.deps.status_bar.configure(text=f"Lat: N/A, Lon: N/A")
            else:
                # To handle cases where transformer is not available, use tie points to interpolate lat/lon
                if scene.geo_coord_helpers["transformer"] is None:
                    row_src, col_src = ds_to_src_pixel(y, x, scene.rcm_scaled_data["src_height"], scene.rcm_scaled_data["src_width"],
                                                    scene.rcm_scaled_data["dst_height"], scene.rcm_scaled_data["dst_width"])
                    lat, lon = scene.pix2ll(row_src, col_src)
                else:
                    # Convert downscaled image coordinates to original image coordinates
                    # x and y (row and col) are flipped so flip back before geocoding
                    x, y = xy(scene.geo_coord_helpers["dst_transform"], y, x, offset="center")
                    # Convert image coordinates to geographic coordinates (lat/lon)
                    lon, lat = scene.geo_coord_helpers["transformer"].transform(x, y)
                
                # Convert lat and lon to DMS format
                lat_dms = decimal_to_dms(lat, is_latitude=True)
                lon_dms = decimal_to_dms(lon, is_latitude=False)
                self.deps.status_bar.configure(text=f"Lat: {lat:.4f}, Lon: {lon:.4f}\n{lat_dms} {lon_dms}")

    def _on_escape_key(self, event):
        """Handle Escape key press to exit bucket fill mode or deselect polygons when in annotation mode."""
        anno = self.deps.app_state.anno
        if self.deps.annotation_window.winfo_viewable():
            if anno.annotation_mode == 'bucket_fill':
                self.annotation_controller.exit_bucket_fill(event)
            else:
                self.annotation_controller.reset_annotation()
                anno.active_label = None
                anno.annotation_mode = None
                self.deps.canvas.config(cursor="")

    def _on_ctrl_z(self, event=None):
        """
        Handle Ctrl+Z key press to undo the last annotation action, 
        pop from the undo stack, push to the redo stack, and update the display accordingly.
        """
        anno = self.deps.app_state.anno
        scene = self.deps.app_state.scene
        if self.deps.annotation_window.winfo_viewable() and anno.undo_stack:
            # Pop from undo stack and push to redo stack with current state
            last_polygon, last_colours, last_window = anno.undo_stack.pop()
            anno.redo_stack.append((last_polygon, scene.predictions[scene.active_source][last_polygon].copy(), last_window))
            self.annotation_controller.undo_redo_annotation(last_polygon, last_colours, last_window)


    def _on_ctrl_y(self, event=None):
        """
        Handle Ctrl+Y (also Ctrl+Shift+Z) key press to redo the last undone annotation action, 
        pop from the redo stack, push to the undo stack, and update the display accordingly.
        """
        anno = self.deps.app_state.anno
        scene = self.deps.app_state.scene
        if self.deps.annotation_window.winfo_viewable() and anno.redo_stack:
            # Pop from redo stack and append to undo stack with current state
            last_polygon, last_colours, last_window = anno.redo_stack.pop()
            anno.undo_stack.append((last_polygon, scene.predictions[scene.active_source][last_polygon].copy(), last_window))
            self.annotation_controller.undo_redo_annotation(last_polygon, last_colours, last_window)

    