'''
Visualizer module for Remote Sensing Visualizer application

Contains the Visualizer class which manages the GUI and image processing.

Last modified: Mar 2026
'''
import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
from rasterio.transform import xy

from ui.evaluation import EvaluationPanel
from ui.annotation import AnnotationPanel
from ui.minimap import Minimap
from core.utils import rgb2gray, generate_boundaries, ds_to_src_pixel, tiepoints_1d_to_grid, make_pix2ll
from core.io import load_existing_annotation, load_rcm_product, run_pred_model, scale_hh_hv, build_land_masks, normalize_and_prepare_images, resource_path
from core.segmentation import get_segment_contours, IRGS, remove_edge_touching_polygons
from core.overlay import compose_overlay
from core.render import crop_resize, layer_imagery
from core.contrast_handler import enhance_outlier_slider
from app.state import AppState
from app.dependencies import AppDeps
from ui.visualizer_layout import build_visualizer_layout
from controllers.scene_controller import SceneController
from controllers.display_controller import DisplayController


class Visualizer(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.app_state = AppState()
        display = self.app_state.display
        # ==================== GUI DESIGN

        # ------- Visualizer settings
        self.title("Visualizer")

        ctk.set_appearance_mode("System")  # or "Dark", "Light"
        ctk.set_default_color_theme("blue")  # or another theme

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # Use 80% of screen size
        window_width = int(screen_width * 0.80)
        window_height = int(screen_height * 0.80)
        self.geometry(f"{window_width}x{window_height}")

        #%% Initial state
        self.selection_start_coord = None

        # Annotation state
        self.annotation_mode = None  # 'rectangle' or None
        self.selected_polygon = None   # Current canvas item being drawn
        self.double_click_flag = False
        self.reset_annotation()

        layout = build_visualizer_layout(self, self.app_state)

        self.deps = AppDeps(
            app=self,
            app_state=self.app_state,
            canvas=layout.canvas,
            sidebar=layout.sidebar,
            minimap=layout.minimap,
            minimap_window_id=layout.minimap_window_id,
            status_bar=layout.status_bar,
            annotation_panel=layout.annotation_panel,
            evaluation_panel=layout.evaluation_panel,
            annotation_window=layout.annotation_window,
            evaluation_window=layout.evaluation_window,
            loading_bar=layout.loading_bar,
            loading_bar_label=layout.loading_bar_label,
            widgets=layout.widgets,
        )

        self.scene_controller = SceneController(self.deps)
        self.display_controller = DisplayController(self.deps)

        #%% INITIAL VISUALIZATION / STATE

        display.channel_mode = self.deps.widgets['mode_var_color_composite'].get()
        if display.channel_mode == "(HH/HV)":
            display.channel_mode = "HV" if self.deps.widgets['hh_hv_switch'].get() else "HH"

        # Disable everything until SAR scene is chosen
        self._set_all_children_enabled(
            self.deps.sidebar,
            False,
            exclude=[self.deps.widgets['choose_SAR_scene_toggle_btn']]
        )

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(500, self.choose_SAR_scene)

    # Minimap control
    def _update_minimap_position(self, event=None):
        """
        Keep the minimap in the bottom-right corner of the canvas when resizing.
        """
        pad = 12
        w = self.deps.canvas.winfo_width()
        h = self.deps.canvas.winfo_height()
        self.deps.canvas.coords(self.deps.minimap_window_id, w - pad, h - pad)


    # Load images
        
    def update_label_source_widgets(self, lbl_source, i):
        """
        Update the label source selection widgets (button) when new label sources are loaded.
        """
        # Radio buttons for explicit selection
        if self.deps.widgets['mode_var_lbl_source'] is None:
            self.deps.widgets['mode_var_lbl_source'] = ctk.StringVar(value=lbl_source)  # Default selection
            self.deps.widgets['mode_var_lbl_source_prev'] = self.deps.widgets['mode_var_lbl_source'].get()
        self.deps.widgets['lbl_source_btn'][lbl_source] = ctk.CTkRadioButton(self.deps.widgets['lbl_source_frame'], 
                                                                                text=lbl_source, 
                                                                                variable=self.deps.widgets['mode_var_lbl_source'],
                                                                value=lbl_source, 
                                                                command=self.choose_lbl_source)
        self.deps.widgets['lbl_source_btn'][lbl_source].grid(row=i+1, column=0, sticky="w", pady=(10, 10))

    def load_pred(self):
        self.scene_controller.load_pred()


    # Display handle

    def set_overlay(self):
        self.display_controller.set_overlay()

    def choose_image(self):
        self.display_controller.choose_image()

    def display_image(self):
        self.display_controller.display_image()
    
    def refresh_view(self):
        self.display_controller.refresh_view()

    def refresh_all(self):
        self.display_controller.refresh_all()


    # Image selection handle

    def choose_SAR_scene(self):
        self.scene_controller.choose_SAR_scene()

    def color_composite(self):
        """
        Handle color composite selection changes, enable/disable HH/HV switch accordingly, and update the displayed image.
        """
        display = self.app_state.display
        display.channel_mode = self.deps.widgets["mode_var_color_composite"].get()

        if display.channel_mode == "(HH/HV)":
            self.deps.widgets["hh_hv_switch"].configure(state=ctk.NORMAL)
            self.HH_HV(get_channel=True)
        else:
            self.deps.widgets["hh_hv_switch"].configure(state=ctk.DISABLED)
            self.HH_HV(get_channel=False)

        
    def HH_HV(self, get_channel=True):
        """
        Handle color composite changes, update the displayed image based on the selected channel, and reset contrast slider.
        """
        display = self.app_state.display
        scene = self.app_state.scene

        if get_channel:
            display.channel_mode = "HV" if self.deps.widgets["hh_hv_switch"].get() else "HH"

        self.deps.widgets["contrast_slider"].set(0)  # Reset contrast slider
        self.contrast_slider_handle(0)

        self.title(f"Scene {scene.scene_name}-{display.channel_mode}")
        self.choose_image()

        self.refresh_all()

    # Image handle
    def contrast_slider_handle(self, val):
        """
        Handle contrast slider changes, apply contrast enhancement to the current image based on the selected channel(s), and refresh the display.
        """
        scene = self.app_state.scene
        display = self.app_state.display

        display.contrast = (val/200) * 0.15

        if display.channel_mode in ["(HH, HH, HV)", "(HH, HV, HV)"]:
            HH_contrasted = enhance_outlier_slider(
                img_u8=scene.orig_img["HH"], # Pass raw image for faster processing
                hist=scene.hist["HH"],
                n_valid=scene.n_valid["HH"],
                s=display.contrast
            )

            HV_contrasted = enhance_outlier_slider(
                img_u8=scene.orig_img["HV"], # Pass raw image for faster processing
                hist=scene.hist["HV"],
                n_valid=scene.n_valid["HV"],
                s=display.contrast
            )

            # Re-layer the imagery with new contrast
            scene.img = layer_imagery(
                HH_contrasted,
                HV_contrasted,
                display.channel_mode
            )
        else:
            scene.img = enhance_outlier_slider(
                img_u8=scene.orig_img[display.channel_mode], # Pass raw image for faster processing
                hist=scene.hist[display.channel_mode],
                n_valid=scene.n_valid[display.channel_mode],
                s=display.contrast
            )

        self.refresh_all()

    def right_click_contrast_reset(self, event):
        """
        Handle right-click on contrast slider to reset contrast to default, refresh the display.
        """
        self.deps.widgets["contrast_slider"].set(0) # reset to default
        self.app_state.display.contrast = 0.0
        self.contrast_slider_handle(0)
        self.refresh_all()

    def brightness_slider_handle(self,val):
        """
        Handle brightness slider changes, update the displayed image based on the selected channel, and refresh the display.
        """
        self.app_state.display.brightness = float(val)/100
        self.refresh_all()

    def right_click_brightness_reset(self, event):
        """
        Handle right-click on brightness slider to reset brightness to default, refresh the display.
        """
        self.deps.widgets["brightness_slider"].set(0) # reset to default
        self.app_state.display.brightness = 0.0
        self.refresh_all()

    # Segmentation handle

    def opacity_slider_handle(self, val):
        """
        Handle opacity slider changes, update the overlay opacity, and refresh the display.
        """
        self.app_state.overlay.alpha = float(val)/100
        self.set_overlay()
        self.display_image()

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                self.deps.annotation_panel.update_zoomed_display()

    def segmentation_toggle(self):
        """
        Handle segmentation overlay toggle, update the button appearance based on the state, and refresh the display.
        When 'OFF' just show base image, when 'ON' show overlay
        """
        overlay_state = self.app_state.overlay
        overlay_state.show_overlay = not overlay_state.show_overlay
        state = "ON" if overlay_state.show_overlay else "OFF"
        self.deps.widgets["segmentation_toggle_btn"].configure(text=state)

        self.display_image()

        if overlay_state.show_overlay:
            # Restore default appearance
            self.deps.widgets["segmentation_toggle_btn"].configure(
                fg_color=self.deps.widgets["default_fg_color"],  # Default customtkinter blue
                hover_color=self.deps.widgets["default_hover_color"],
                text_color=self.deps.widgets["default_text_color"]
            )
        else:
            # Set to gray when OFF
            self.deps.widgets["segmentation_toggle_btn"].configure(
                fg_color="#888888",     # Gray background
                hover_color="#777777",  # Slightly darker on hover
                text_color="white"
            )

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                self.deps.annotation_panel.update_zoomed_display()


    # Zoom handle

    def enable_zoom_selection(self):
        """
        Enable zoom selection mode, change cursor to crosshair, and update button appearance.
        """
        view = self.app_state.view
        overlay = self.app_state.overlay
        if not overlay.select_local_segmentation: # If not in local segmentation mode perform zoom selection
            view.zoom_select_mode = True
            self.deps.widgets["zoom_select_btn"].configure(**self.deps.widgets["zoom_btn_active_style"])
        self.deps.canvas.config(cursor="crosshair")

    def zoom_to_rectangle(self, x_min, y_min, x_max, y_max):
        """
        Zoom the view to fit the rectangle drawn by the user, update the view's zoom factor
        and offsets accordingly, and refresh the display.
        """

        view = self.app_state.view
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

        self.refresh_view()
        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

    def run_local_segmentation(self, x_min, y_min, x_max, y_max):
        """
        Run local segmentation (IRGS) on the area selected by the user, 
        update the overlay with the local segmentation results, and refresh the display.
        """
        overlay = self.app_state.overlay
        scene = self.app_state.scene

        overlay.local_segmentation_area = np.stack([scene.raw_img[overlay.local_segmentation_source], 
                                                    scene.raw_img[overlay.local_segmentation_source]], axis=-1)[y_min:y_max, x_min:x_max]

        overlay.local_segmentation_limits = (x_min, y_min, x_max, y_max)
        land_nan_mask_crop = scene.land_nan_masks[scene.active_source][y_min:y_max, x_min:x_max]
        # Disable select local segmentation mode after selection
        overlay.select_local_segmentation = False

        self.deps.canvas.delete(self.selection_rect_id)
        self.selection_rect_id = None
        self.selection_start_coord = None

        # Show loading bar
        self.deps.loading_bar_label.grid(row=0, column=0)
        self.deps.loading_bar.grid(row=1, column=0)
        self.update_idletasks()

        self.deps.loading_bar.set(0)
        self.deps.loading_bar_label.configure(text="Running local segmentation...")
        self.update_idletasks()

        # Run IRGS on the selected area
        irgs_output, boundaries = IRGS(overlay.local_segmentation_area, n_classes=overlay.local_seg_n_classes, n_iter=120, mask=~land_nan_mask_crop)

        self.deps.loading_bar.set(0.4)
        self.deps.loading_bar_label.configure(text="Clearing border polygons...")
        self.update_idletasks()

        irgs_output, boundaries = remove_edge_touching_polygons(irgs_output)

        self.deps.loading_bar.set(0.7)
        self.deps.loading_bar_label.configure(text="Applying segmentation on overlay...")
        self.update_idletasks()

        overlay.local_segmentation_mask = np.zeros_like(scene.boundmasks[scene.active_source], dtype=np.uint8)
        overlay.local_segmentation_mask[y_min:y_max, x_min:x_max] = irgs_output
        overlay.local_segmentation_mask = np.tile(overlay.local_segmentation_mask[:, :, np.newaxis], (1, 1, 3))

        overlay.local_segmentation_bounds = np.zeros_like(scene.boundmasks[scene.active_source], dtype=bool)
        boundaries_bool = boundaries != 1
        overlay.local_segmentation_bounds[y_min:y_max, x_min:x_max] = boundaries_bool
        overlay.show_local_segmentation = True

        self.reset_annotation() # Reset annotation to prevent annotation on old local segmentation
        self.refresh_view()

        self.deps.loading_bar.set(1)
        self.deps.loading_bar_label.configure(text="Local segmentation applied")
        self.update_idletasks()

        self.after(3000, self.deps.loading_bar_label.grid_remove)
        self.after(3000, self.deps.loading_bar.grid_remove)
        self.update_idletasks()



    def reset_zoom(self):
        """
        Reset the zoom to fit the entire image in the canvas,
        center the image, and refresh the display.
        """

        view = self.app_state.view
        scene = self.app_state.scene
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

        self.refresh_view()
        if self.app_state.anno.polygon_points_img_coor:
            self.draw_polygon_on_canvas()


    # Label source handle

    def choose_lbl_source(self, plot=True):
        """
        Handle label source selection changes, update the active label source in the app state, 
        check if predictions for that source exist, refresh the display, and reset annotations.
        """

        scene = self.app_state.scene

        if scene.img is None:
            return 0 # No image loaded, ignore label source change

        scene.active_source = self.deps.widgets["mode_var_lbl_source"].get()
        key = scene.active_source

        if scene.predictions[key] is None:
            messagebox.showinfo("Error", f"The selected directory does not contain prediction files for {key}.", parent=self.master)
            self.deps.widgets["mode_var_lbl_source"].set(self.mode_var_lbl_source_prev)
            return 0
        self.mode_var_lbl_source_prev = key

        if plot:
            self.refresh_view()

        self.reset_annotation()

        if self.deps.evaluation_window.winfo_viewable():
            self.deps.evaluation_panel.load_existing_evaluation()

        return 1


    # Canvas Events

    def _on_mousewheel(self, event):
        """Handle mouse wheel events for zooming in and out."""
        view = self.app_state.view
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

        self.refresh_view()
        if self.app_state.anno.polygon_points_img_coor:
            self.draw_polygon_on_canvas()

    def _on_left_click_await(self, event):
        """Handle left mouse click with differentiation between single and double clicks."""
        anno = self.app_state.anno
        view = self.app_state.view
        if view.zoom_select_mode or anno.annotation_mode in ['rectangle', 'polygon'] \
            or self.app_state.overlay.select_local_segmentation:
            self._on_left_click(event)
        else:
            self.after(180, lambda: self.choose_click_event(event))
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
        view = self.app_state.view
        anno = self.app_state.anno
        overlay = self.app_state.overlay
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
                self.selected_polygon = self.deps.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='yellow', width=2)
        elif anno.annotation_mode == 'polygon':
                self._add_polygon_point(event)
        elif anno.annotation_mode == 'bucket_fill':
            x = round(np.float64((event.x - view.offset_x) / view.zoom_factor))
            y = round(np.float64((event.y - view.offset_y) / view.zoom_factor))
            if anno.selected_polygon_area_idx is None or \
                (y,x) not in zip(anno.selected_polygon_area_idx[0], anno.selected_polygon_area_idx[1]):
                self._on_double_click(event) # Use double-click handler for bucket fill
            else:
                self.bucket_fill_polygon_area(event)
        else:
            # Start pan
            view.pan_start_screen = (event.x, event.y)

    def _on_left_drag(self, event):
        """Handle mouse drag for zoom selection, panning, or rectangle drawing for polygon or local segmentation."""

        view = self.app_state.view
        scene = self.app_state.scene
        anno = self.app_state.anno
        overlay = self.app_state.overlay

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
            self.deps.canvas.coords(self.selected_polygon, x0, y0, x1, y1)
        elif view.pan_start_screen:
            # Pan mode
            dx = event.x - view.pan_start_screen[0]
            dy = event.y - view.pan_start_screen[1]
            view.offset_x += dx
            view.offset_y += dy
            view.pan_start_screen = (event.x, event.y)
            
            self.refresh_view()
            if anno.polygon_points_img_coor: 
                self.draw_polygon_on_canvas()

    def _on_left_release(self, event):
        """Handle left mouse button release to finalize zoom selection, rectangle, or polygon drawing."""	
        
        view = self.app_state.view
        scene = self.app_state.scene
        overlay = self.app_state.overlay
        anno = self.app_state.anno
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

            self.zoom_to_rectangle(img_x_min, img_y_min, img_x_max, img_y_max)

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

            self.run_local_segmentation(img_x_min, img_y_min, img_x_max, img_y_max)
        
        elif anno.annotation_mode == 'rectangle' and self.selection_start_coord:
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y

            polygon_points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # Rectangle points
            for x, y in polygon_points:
                anno.polygon_points_img_coor.append((int((x - view.offset_x) / view.zoom_factor),
                                                     int((y - view.offset_y) / view.zoom_factor)))
            self._finish_polygon()
            
            # Reset variables
            self.selection_start_coord = None

        elif view.pan_start_screen:
            view.pan_start_screen = None  # end pan

    def _on_right_click(self, event):
        """Handle right-click to finish polygon drawing."""
        if self.app_state.anno.annotation_mode == 'polygon':
            self._finish_polygon()

    def _on_double_click_set_flag(self, event):
        """Set flag on double click to differentiate between single and double clicks."""
        self.double_click_flag = True

    def _on_double_click(self, event):
        """
        Handle double-click to select polygon area, check if selection is within bounds 
        and local segmentation area, draw polygon on canvas, or bucket fill if in bucket fill mode.
        """
        view = self.app_state.view
        scene = self.app_state.scene
        anno = self.app_state.anno
        overlay = self.app_state.overlay
        if self.deps.annotation_window.winfo_viewable():

            if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
                self.deps.annotation_panel.zoom_window is not None and 
                self.deps.annotation_panel.zoom_window.winfo_exists()):
                if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                    self.deps.annotation_panel.zoom_window.destroy()

            self.reset_annotation()

            x = int((event.x - view.offset_x) / view.zoom_factor)
            y = int((event.y - view.offset_y) / view.zoom_factor)

            # Check if selection is outside local segmentation area
            if overlay.show_local_segmentation:
                if x < overlay.local_segmentation_limits[0] or x >= overlay.local_segmentation_limits[2] or \
                   y < overlay.local_segmentation_limits[1] or y >= overlay.local_segmentation_limits[3]:
                    result = messagebox.askyesno("Selection Out of Local Segmentation Area", "Selecting outside of local segmentation view will close the local segmentation view.\n\nClose local segmentation view?", parent=self.master)
                    if result: # Close local segmentation view
                        overlay.show_local_segmentation = False
                        self.refresh_view()
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
                            result = messagebox.askyesno("Selection Not Part of Local Segmentation", "Selecting a polygon not part of the local segmentation view will close the local segmentation view.\n\nClose local segmentation view?", parent=self.master)
                            
                            if result: # Close local segmentation view and get contours from prediction
                                overlay.show_local_segmentation = False
                                selected_outside = True
                                self.refresh_view()
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
                self.reset_annotation()
                return
            
            # draw polygon(s) on canvas
            anno.polygon_points_img_coor = [[(x, y) for y, x in c] for c in contours]
            anno.multiple_polygons = True
            self.draw_polygon_on_canvas()

            # If in bucket fill mode and double clicked
            if anno.annotation_mode == 'bucket_fill' and self.double_click_flag:
                self.bucket_fill_polygon_area(event)
                self.double_click_flag = False

    def _on_mouse_move(self, event):
        """
        Handle mouse move events to display lat/lon coordinates in the status bar 
        based on the current mouse position, check if the coordinates are valid, 
        and convert to DMS format for display.
        """
        view = self.app_state.view
        scene = self.app_state.scene
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
                lat_dms = self.decimal_to_dms(lat, is_latitude=True)
                lon_dms = self.decimal_to_dms(lon, is_latitude=False)
                self.deps.status_bar.configure(text=f"Lat: {lat:.4f}, Lon: {lon:.4f}\n{lat_dms} {lon_dms}")

    def _on_escape_key(self, event):
        """Handle Escape key press to exit bucket fill mode or deselect polygons when in annotation mode."""
        anno = self.app_state.anno
        if self.deps.annotation_window.winfo_viewable():
            if anno.annotation_mode == 'bucket_fill':
                self.exit_bucket_fill(event)
            else:
                self.reset_annotation()
                anno.active_label = None
                anno.annotation_mode = None
                self.deps.canvas.config(cursor="")

    def _on_ctrl_z(self, event=None):
        """
        Handle Ctrl+Z key press to undo the last annotation action, 
        pop from the undo stack, push to the redo stack, and update the display accordingly.
        """
        anno = self.app_state.anno
        scene = self.app_state.scene
        if self.deps.annotation_window.winfo_viewable() and anno.undo_stack:
            # Pop from undo stack and push to redo stack with current state
            last_polygon, last_colours, last_window = anno.undo_stack.pop()
            anno.redo_stack.append((last_polygon, scene.predictions[scene.active_source][last_polygon].copy(), last_window))
            self.undo_redo_annotation(last_polygon, last_colours, last_window)


    def _on_ctrl_y(self, event=None):
        """
        Handle Ctrl+Y (also Ctrl+Shift+Z) key press to redo the last undone annotation action, 
        pop from the redo stack, push to the undo stack, and update the display accordingly.
        """
        anno = self.app_state.anno
        scene = self.app_state.scene
        if self.deps.annotation_window.winfo_viewable() and anno.redo_stack:
            # Pop from redo stack and append to undo stack with current state
            last_polygon, last_colours, last_window = anno.redo_stack.pop()
            anno.undo_stack.append((last_polygon, scene.predictions[scene.active_source][last_polygon].copy(), last_window))
            self.undo_redo_annotation(last_polygon, last_colours, last_window)


    # Operations
    
    def show_evaluation_panel(self):
        """Show evaluation panel, close annotation panel if open"""
        ann_flag = True
        if self.deps.annotation_window.winfo_viewable():
            ann_flag = self.close_annotation_panel()
        if not ann_flag:
            return
        
        if self.deps.evaluation_panel.scene_name != self.app_state.scene.scene_name:
            self.deps.evaluation_panel.set_scene_name(self.app_state.scene.scene_name)
        
        self.deps.evaluation_window.deiconify()
        self.deps.evaluation_window.focus_force()
    
    def show_annotation_panel(self):
        """Show annotation panel, close evaluation panel if open. Load existing annotation if exists."""
        eva_flag = True
        if self.deps.evaluation_window.winfo_viewable():
            eva_flag = self.close_evaluation_panel()
        if not eva_flag:
            return
        
        annotation_loaded = self.check_existing_annotation()

        if annotation_loaded:
            self.deps.annotation_panel.insert_existing_notes(self.app_state.anno.annotation_notes)
            self.deps.annotation_window.deiconify()
            self.deps.annotation_window.focus_force()

        for btn in self.deps.widgets["lbl_source_btn"].values():
            btn.configure(state=ctk.DISABLED) # Disable label source selection when annotation panel is open

    def close_evaluation_panel(self):
        """Close evaluation panel, check for unsaved changes, reset scene name and fields, and hide the window."""
        if self.deps.evaluation_panel.unsaved_changes:
            result = messagebox.askyesnocancel("Unsaved Changes", "You have unsaved evaluation data. Do you want to save before exiting?")
            if result is None:
                return  0   # Cancel
            elif result:
                if not self.deps.evaluation_panel.save_evaluation():
                    return  0   # Failed to save → don't close
        
        self.deps.evaluation_panel.scene_name = ""
        self.deps.evaluation_panel.reset_fields()
        self.deps.evaluation_window.withdraw()
        return 1

    def close_annotation_panel(self):
        """Close annotation panel, check for unsaved changes, reset annotation fields, and hide the window."""
        anno = self.app_state.anno
        if self.deps.annotation_panel.unsaved_changes:
            result = messagebox.askyesnocancel("Unsaved Changes", "Your 'Custom Annotation is unsaved'. Do you want to save before exiting?")
            if result is None:
                return  0   # Cancel
            elif result:
                if not self.deps.annotation_panel.save_annotation():
                    return  0   # Failed to save → don't close

        self.reset_annotation()
        if self.app_state.overlay.show_local_segmentation:
            self.clear_local_seg()
        self.deps.annotation_panel.unsaved_changes = False
        self.deps.annotation_window.withdraw()
        anno.annotation_mode = None
        self.exit_bucket_fill(None)
        self.deps.canvas.config(cursor="")
        for btn in self.deps.widgets["lbl_source_btn"].values():
            btn.configure(state=ctk.NORMAL) # Re-enable label source buttons when annotation panel is closed

        return 1


    # Annotation options

    def draw_rectangle(self):
        """Enable rectangle drawing mode."""
        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                self.deps.annotation_panel.zoom_window.destroy()

        self.app_state.anno.annotation_mode = 'rectangle'
        self.deps.canvas.config(cursor="crosshair")
        self.reset_annotation()

    def draw_polygon(self):
        """Enable Polygon drawing mode."""
        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                self.deps.annotation_panel.zoom_window.destroy()

        self.app_state.anno.annotation_mode = 'polygon'
        self.deps.canvas.config(cursor="crosshair")
        self.reset_annotation()



    def _add_polygon_point(self, event):
        """Add a point to the polygon."""
        view = self.app_state.view
        anno = self.app_state.anno
        if anno.annotation_mode == 'polygon':
            anno.polygon_points_img_coor.append((int((event.x - view.offset_x) / view.zoom_factor), 
                                                 int((event.y - view.offset_y) / view.zoom_factor)))
            
            self.draw_polygon_on_canvas()
    
    def draw_polygon_on_canvas(self):
        """Draw the polygon defined by the image coordinates on the canvas, converting to canvas coordinates."""
        view = self.app_state.view
        anno = self.app_state.anno
        # Remove existing polygon if exists before drawing new one
        if self.selected_polygon:
            if isinstance(self.selected_polygon, list):
                for poly in self.selected_polygon:
                    self.deps.canvas.delete(poly)
            else:
                self.deps.canvas.delete(self.selected_polygon)
            self.selected_polygon = None

        if not anno.multiple_polygons:
            polygon_points_img_coor = [anno.polygon_points_img_coor]
        else:
            polygon_points_img_coor = anno.polygon_points_img_coor
        
        self.selected_polygon = []
        for p_img_coor in polygon_points_img_coor:
            polygon_points = [
                (x * view.zoom_factor + view.offset_x, 
                 y * view.zoom_factor + view.offset_y) for x, y in p_img_coor
            ]

            self.selected_polygon.append(self.draw_single_polygon_on_canvas(polygon_points))

        if self.deps.canvas.find_withtag("polygon") and not self.app_state.overlay.show_overlay:
                self.deps.canvas.itemconfig("polygon", state="hidden")

    def draw_single_polygon_on_canvas(self, polygon_points):
        """
        Draw a single polygon on the canvas based on the number of points (1 for point, 2 for line, 3+ for polygon) and 
        return the canvas item ID.
        """
        if len(polygon_points) == 1:
            x, y = polygon_points[0]
            r = 3  # radius for the point
            selected_polygon = self.deps.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill='yellow', outline='yellow', tags=("polygon",)
            )
        elif len(polygon_points) == 2:
            selected_polygon = self.deps.canvas.create_line(
                *polygon_points, fill='yellow', width=2, tags=("polygon",)
            )
        elif len(polygon_points) >= 3:
            selected_polygon = self.deps.canvas.create_polygon(
                polygon_points, outline='yellow', width=2, fill='', tags=("polygon",)
            )

        return selected_polygon

    def _finish_polygon(self):
        """Finish drawing a polygon and store it."""
        scene = self.app_state.scene
        anno = self.app_state.anno
        img_points = anno.polygon_points_img_coor
        if len(img_points) >= 3:

            img_x_min = max(0, min(x for x, y in img_points))
            img_y_min = max(0, min(y for x, y in img_points))
            img_x_max = min(scene.img.shape[1], max(x for x, y in img_points))
            img_y_max = min(scene.img.shape[0], max(y for x, y in img_points))
            if img_x_max > img_x_min and img_y_max > img_y_min:
                anno.selected_polygon_window = (img_y_min, img_y_max, img_x_min, img_x_max)

                mask = np.zeros((img_y_max - img_y_min, img_x_max - img_x_min), dtype=np.uint8)
                shifted_points = [(x - img_x_min, y - img_y_min) for x, y in img_points]
                cv2.fillPoly(mask, [np.array(shifted_points, dtype=np.int32)], 255)
                anno.selected_polygon_area_idx = [(y + img_y_min, x + img_x_min) for y, x in zip(*np.where(mask==255))]
                anno.selected_polygon_area_idx = tuple(zip(*anno.selected_polygon_area_idx))

            # Reset variables
            anno.annotation_mode = None
            self.deps.canvas.config(cursor="")

    def reset_annotation(self):
        """Reset the annotation state."""
        anno = self.app_state.anno
        if self.selected_polygon:
            if isinstance(self.selected_polygon, list):
                for poly in self.selected_polygon:
                    self.deps.canvas.delete(poly)
            else:
                self.deps.canvas.delete(self.selected_polygon)
            self.selected_polygon = None

        anno.polygon_points_img_coor = []
        anno.selected_polygon_window = None
        anno.selected_polygon_area_idx = None
        anno.multiple_polygons = False


    def annotate_class(self, class_color=[0, 0, 0]):
        """
        Annotate the selected polygon area with the specified class color, update the Custom Annotation layer, 
        handle undo/redo stacks, and refresh the display.
        """
        scene = self.app_state.scene
        anno = self.app_state.anno

        if anno.selected_polygon_area_idx is None:
            if anno.annotation_mode == 'polygon':
                if len(anno.polygon_points_img_coor) < 3:
                    messagebox.showinfo("Error", "Polygon incomplete.", parent=self.master)
                    return
                else:
                    self._finish_polygon()
            else:
                messagebox.showinfo("Error", "Please select a polygon area first.", parent=self.master)
                return
        elif scene.land_nan_masks[scene.active_source][anno.selected_polygon_area_idx].all():
            messagebox.showinfo("Error", "Selected area is land or invalid data.", parent=self.master)
            self.reset_annotation()
            return
        
        # Check if this area is already annotated with the selected class.
        if (scene.predictions[scene.active_source][anno.selected_polygon_area_idx] == class_color).all():
            return
        
        key = "Custom_Annotation"

        scene.predictions[key] = scene.predictions[scene.active_source].copy()
        scene.land_nan_masks[key] = scene.land_nan_masks[scene.active_source].copy()
        scene.boundmasks[key] = scene.boundmasks[scene.active_source].copy()
        scene.active_source = key

        if key not in self.deps.widgets["lbl_source_btn"].keys():
            # Add custom annotation as and additional label source
            scene.lbl_sources.append(key)
            scene.filenames.append("{}/{}/{}".format(scene.lbl_sources[-1], scene.scene_name, "custom_annotation.png"))
            self.deps.widgets["lbl_source_btn"][key] = ctk.CTkRadioButton(self.deps.widgets["lbl_source_frame"], 
                                                                text=f"* {key}", 
                                                                variable=self.deps.widgets["mode_var_lbl_source"], 
                                                                value=key, command=self.choose_lbl_source)
            self.deps.widgets["lbl_source_btn"][key].grid(row=len(scene.lbl_sources), column=0, sticky="w", pady=(10, 10))
            
        else:
            self.deps.widgets["lbl_source_btn"][key].configure(text=f"* {key}")
            
        self.deps.annotation_panel.unsaved_changes = True
        self.deps.annotation_panel.save_button.configure(state=ctk.NORMAL)

        # Store in undo stack and clear redo stack
        if anno.undo_stack and len(anno.undo_stack) > anno.stack_limit:
            anno.undo_stack.pop(0)  # Remove oldest entry if stack limit exceeded
        anno.undo_stack.append((anno.selected_polygon_area_idx, scene.predictions[scene.active_source][anno.selected_polygon_area_idx].copy(), anno.selected_polygon_window))
        anno.redo_stack.clear() # Clear redo stack after new annotation

        self.deps.widgets["mode_var_lbl_source"].set(key)   # set custom annotation as current label source
        scene.predictions[scene.active_source][anno.selected_polygon_area_idx] = class_color
        scene.predictions[scene.active_source][scene.land_nan_masks[scene.active_source]] = [255, 255, 255]

        # # Show annotated area on minimap (excluding land and invalid areas)
        # valid_polygon_idx = tuple(zip(*[(y, x) for y, x in zip(*anno.selected_polygon_area_idx) if not scene.land_nan_masks[scene.active_source][y, x]]))

        # Do a vectorized compare of the existing prediction (only 1 so [0]) with the new prediction to get a mask of the changed area
        if self.deps.widgets["show_prev_anno_switch"].get():
            changed_area_mask = scene.predictions[scene.active_source][:,:,0] != scene.predictions[scene.lbl_sources[0]][:,:,0]
            self.deps.minimap.show_changed_area(scene.img, changed_area_mask)

        img_y_min, img_y_max, img_x_min, img_x_max = anno.selected_polygon_window
        img_y_min = max(0, img_y_min-20)
        img_y_max = min(scene.predictions[scene.active_source].shape[0], img_y_max+20)
        img_x_min = max(0, img_x_min-20)
        img_x_max = min(scene.predictions[scene.active_source].shape[1], img_x_max+20)
        scene.boundmasks[scene.active_source][img_y_min: img_y_max, 
                    img_x_min: img_x_max] = generate_boundaries(rgb2gray(scene.predictions[scene.active_source][img_y_min: img_y_max, 
                                                                                    img_x_min: img_x_max]))

        self.refresh_view()
        if anno.polygon_points_img_coor: 
                self.draw_polygon_on_canvas()

    def undo_redo_annotation(self, last_polygon_area_idx, last_colours, last_window):
        """Undo or redo an annotation by restoring the previous state."""
        scene = self.app_state.scene

        # Change colours in the polygon area back to the last colours
        scene.predictions[scene.active_source][last_polygon_area_idx] = last_colours

        # Find new boundaries in the affected area
        img_y_min, img_y_max, img_x_min, img_x_max = last_window
        img_y_min = max(0, img_y_min-20)
        img_y_max = min(scene.predictions[scene.active_source].shape[0], img_y_max+20)
        img_x_min = max(0, img_x_min-20)
        img_x_max = min(scene.predictions[scene.active_source].shape[1], img_x_max+20)
        scene.boundmasks[scene.active_source][img_y_min: img_y_max, 
                    img_x_min: img_x_max] = generate_boundaries(rgb2gray(scene.predictions[scene.active_source][img_y_min: img_y_max, 
                                                                                    img_x_min: img_x_max]))
        # Show annotated area on minimap
        if self.deps.widgets["show_prev_anno_switch"].get():
            changed_area_mask = scene.predictions[scene.active_source][:,:,0] != scene.predictions[scene.lbl_sources[0]][:,:,0]
            self.deps.minimap.show_changed_area(scene.img, changed_area_mask)

        # Reset annotation and refresh view
        self.reset_annotation()
        self.refresh_view()


    def check_existing_annotation(self):
        """
        Check for existing custom annotation, prompt user to use it or create new annotation from the prediction, 
        and set active source to custom annotation if not canceled.
        """
        scene = self.app_state.scene
        key = "Custom_Annotation"

        # Duplicate scene for new/updated custom annotation scene
        if key != scene.active_source and key in self.deps.widgets["lbl_source_btn"].keys():
            result = messagebox.askyesnocancel("Existing annotation", "You have an existing custom annotation. Do you want to use it?")
            if result is None:
                self.reset_annotation()
                return 0 # Cancel
            elif not result:  # No, create new annotation from choice of overlay
                self.deps.annotation_panel.reset_label_from()

            scene.active_source = key
            self.deps.widgets["mode_var_lbl_source"].set(key)
            self.refresh_view()
        return 1
    
    def toggle_show_anno_on_minimap(self):
        """
        Toggle the display of the annotated area on the minimap by comparing the 
        current annotation with the original prediction and showing the changed area 
        if the switch is on, or resetting to the original image if the switch is off.
        """
        scene = self.app_state.scene
        custom_anno = "Custom_Annotation"

        if custom_anno in scene.lbl_sources and self.deps.widgets["show_prev_anno_switch"].get():
            changed_area_mask = scene.predictions[custom_anno][:,:,0] != scene.predictions[scene.lbl_sources[0]][:,:,0]
            self.deps.minimap.show_changed_area(scene.img, changed_area_mask)
        else:
            self.deps.minimap.set_image(scene.img)


    def label_water(self, bucket_fill=False):
        """Label selected polygon as water with specified color, check if called from bucket fill."""
        # Check if called by left click or bucket fill
        if not bucket_fill and self.app_state.anno.annotation_mode == 'bucket_fill':
            self.exit_bucket_fill(None)
        self.annotate_class([0, 255, 255])

    def label_ice(self, bucket_fill=False):
        """Label selected polygon as ice with specified color, check if called from bucket fill."""
        if not bucket_fill and self.app_state.anno.annotation_mode == 'bucket_fill':
            self.exit_bucket_fill(None)
        self.annotate_class([255, 130, 0])

    def label_shoal(self):
        self.annotate_class([0, 255, 0])

    def label_ship(self):
        self.annotate_class([255, 255, 0])

    def label_iceberg(self):
        self.annotate_class([255, 0, 255])

    def label_unknown(self, bucket_fill=False):
        if not bucket_fill and self.app_state.anno.annotation_mode == 'bucket_fill':
            self.exit_bucket_fill(None)
        self.annotate_class([150, 150, 150])

    def bucket_fill(self, event, label):
        """Enable bucket fill mode for the specified label, set cursor, and update annotation panel button styles."""
        anno = self.app_state.anno
        anno.annotation_mode = 'bucket_fill'
        anno.active_label = label
        self.deps.canvas.config(cursor="spraycan")
        
        # Should clean this up later
        if label == "water":
            self.deps.annotation_panel.water_btn.configure(**self.deps.annotation_panel.label_btn_active_style)
            self.deps.annotation_panel.ice_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
            self.deps.annotation_panel.unknown_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
        elif label == "ice":
            self.deps.annotation_panel.ice_btn.configure(**self.deps.annotation_panel.label_btn_active_style)
            self.deps.annotation_panel.water_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
            self.deps.annotation_panel.unknown_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
        elif label == "unknown":
            self.deps.annotation_panel.unknown_btn.configure(**self.deps.annotation_panel.label_btn_active_style)
            self.deps.annotation_panel.water_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
            self.deps.annotation_panel.ice_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
        
        self.focus_set()

    def bucket_fill_polygon_area(self, event):
        """Perform bucket fill annotation on the selected polygon area based on the active label."""
        anno = self.app_state.anno
        if anno.active_label is None:
            return
        elif anno.active_label == "water":
            self.label_water(bucket_fill=True)
        elif anno.active_label == "ice":
            self.label_ice(bucket_fill=True)
        elif anno.active_label == "unknown":
            self.label_unknown(bucket_fill=True)

    def exit_bucket_fill(self, event):
        """Exit bucket fill mode, reset annotation mode and active label, reset cursor, and update annotation panel button styles."""
        anno = self.app_state.anno
        anno.annotation_mode = None
        anno.active_label = None
        self.deps.canvas.config(cursor="")
        self.deps.annotation_panel.water_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
        self.deps.annotation_panel.ice_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
        self.deps.annotation_panel.unknown_btn.configure(**self.deps.annotation_panel.label_btn_default_style)

    # Change this function name later
    def select_area_local_seg(self):
        overlay = self.app_state.overlay
        overlay.select_local_segmentation = True
        # Using zoom selection for local segmentation area selection
        self.enable_zoom_selection()

    def toggle_local_seg_source(self):
        """Toggle the source for local segmentation between HV and HH, and rerun local segmentation if area is already selected."""
        overlay = self.app_state.overlay
        if self.deps.annotation_panel.local_seg_switch.get():
            overlay.local_segmentation_source = "HV"
        else:
            overlay.local_segmentation_source = "HH"

        if overlay.local_segmentation_area is not None:
            x_min, y_min, x_max, y_max = overlay.local_segmentation_limits
            self.run_local_segmentation(x_min, y_min, x_max, y_max)

    def clear_local_seg(self):
        """Clear local segmentation results, reset related variables, exit local segmentation view, and refresh display."""
        overlay = self.app_state.overlay
        if not overlay.show_local_segmentation:
            return
        overlay.local_segmentation_area = None
        overlay.local_segmentation_mask = None
        overlay.local_segmentation_bounds = None
        overlay.show_local_segmentation = False
        self.reset_annotation()
        self.refresh_view()

    def update_local_seg_n_classes(self, value):
        """Update the number of classes for local segmentation, rerun local segmentation if area is already selected."""
        overlay = self.app_state.overlay
        overlay.local_seg_n_classes = int(value)
        if overlay.local_segmentation_area is not None:
            x_min, y_min, x_max, y_max = overlay.local_segmentation_limits
            self.run_local_segmentation(x_min, y_min, x_max, y_max)

    # Misc

    def decimal_to_dms(self, decimal_degree, is_latitude=True):
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

    def _set_all_children_enabled(self, parent, enabled=True, exclude=[]):
        state = ctk.NORMAL if enabled else ctk.DISABLED

        for child in parent.winfo_children():
            if child in exclude:
                continue
            if type(child) in (ctk.CTkFrame, tk.Frame):
                self._set_all_children_enabled(child, enabled, exclude)
            else:
                try:
                    child.configure(state=state)
                except (tk.TclError, ValueError):
                    pass

    def on_close(self):

        ann_flag = True
        if self.deps.annotation_window.winfo_viewable():
            ann_flag = self.close_annotation_panel()

        eva_flag = True
        if self.deps.evaluation_window.winfo_viewable():
            eva_flag = self.close_evaluation_panel()
        
        if ann_flag and eva_flag:
            self.destroy()


# if __name__ == '__main__':

#     multiprocessing.freeze_support()
    
#     app = Visualizer()
#     app.mainloop()

