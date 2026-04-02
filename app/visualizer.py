'''
Visualizer module for Remote Sensing Visualizer application

Contains the Visualizer class which manages the GUI and image processing.

Last modified: Apr 2026
'''
import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
import numpy as np

from app.state import AppState
from app.dependencies import AppDeps
from ui.visualizer_layout import build_visualizer_layout
from controllers.scene_controller import SceneController
from controllers.display_controller import DisplayController
from controllers.image_controls_controller import ImageControlsController
from controllers.annotation_controller import AnnotationController
from controllers.canvas_events_controller import CanvasEventsController
from controllers.zoom_controller import ZoomController
from controllers.local_seg_controller import LocalSegController

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

        # Annotation state
        self.annotation_mode = None  # 'rectangle' or None
        self.double_click_flag = False

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
        self.image_controls_controller = ImageControlsController(
                                            self.deps, 
                                            self.display_controller
                                            )
        self.annotation_controller = AnnotationController(self.deps, self.display_controller)
        self.canvas_events_controller = CanvasEventsController(self.deps, self.display_controller, self.annotation_controller)
        self.zoom_controller = ZoomController(self.deps, self.display_controller, self.annotation_controller)
        self.local_seg_controller = LocalSegController(self.deps, self.display_controller, self.annotation_controller, self.canvas_events_controller, self.zoom_controller)

        #%% INITIAL VISUALIZATION / STATE
        self.reset_annotation()

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
        self.image_controls_controller.color_composite()

        
    def HH_HV(self, get_channel=True):
        self.image_controls_controller.HH_HV(get_channel)

    # Image handle
    def contrast_slider_handle(self, val):
        self.image_controls_controller.contrast_slider_handle(val)

    def right_click_contrast_reset(self, event):
        self.image_controls_controller.right_click_contrast_reset(event)

    def brightness_slider_handle(self,val):
        self.image_controls_controller.brightness_slider_handle(val)

    def right_click_brightness_reset(self, event):
        self.image_controls_controller.right_click_brightness_reset(event)

    # Segmentation handle

    def opacity_slider_handle(self, val):
        self.image_controls_controller.opacity_slider_handle(val)

    def segmentation_toggle(self):
        self.image_controls_controller.segmentation_toggle()


    # Zoom handle

    def enable_zoom_selection(self):
        self.zoom_controller.enable_zoom_selection()

    def zoom_to_rectangle(self, x_min, y_min, x_max, y_max):
        self.zoom_controller.zoom_to_rectangle(x_min, y_min, x_max, y_max)

    def run_local_segmentation(self, x_min, y_min, x_max, y_max):
        self.local_seg_controller.run_local_segmentation(x_min, y_min, x_max, y_max)



    def reset_zoom(self):
        self.zoom_controller.reset_zoom()


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
        self.canvas_events_controller._on_mousewheel(event)

    def _on_left_click_await(self, event):
        self.canvas_events_controller._on_left_click_await(event)

    def choose_click_event(self, event):
        self.canvas_events_controller.choose_click_event(event)

    def _on_left_click(self, event):
        self.canvas_events_controller._on_left_click(event)

    def _on_left_drag(self, event):
        self.canvas_events_controller._on_left_drag(event)

    def _on_left_release(self, event):
        self.canvas_events_controller._on_left_release(event)

    def _on_right_click(self, event):
        self.canvas_events_controller._on_right_click(event)
    
    def _on_double_click_set_flag(self, event):
        self.canvas_events_controller._on_double_click_set_flag(event)

    def _on_double_click(self, event):
        self.canvas_events_controller._on_double_click(event)

    def _on_mouse_move(self, event):
        self.canvas_events_controller._on_mouse_move(event)

    def _on_escape_key(self, event):
        self.canvas_events_controller._on_escape_key(event)

    def _on_ctrl_z(self, event=None):
        self.canvas_events_controller._on_ctrl_z(event)


    def _on_ctrl_y(self, event=None):
        self.canvas_events_controller._on_ctrl_y(event)


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
        self.annotation_controller.draw_rectangle()

    def draw_polygon(self):
        self.annotation_controller.draw_polygon()



    def _add_polygon_point(self, event):
        self.annotation_controller.add_polygon_point(event)
    
    def draw_polygon_on_canvas(self):
        self.annotation_controller.draw_polygon_on_canvas()

    def draw_single_polygon_on_canvas(self, polygon_points):
        self.annotation_controller.draw_single_polygon_on_canvas(polygon_points)

    def _finish_polygon(self):
        self.annotation_controller.finish_polygon()

    def reset_annotation(self):
        self.annotation_controller.reset_annotation()


    def annotate_class(self, class_color=[0, 0, 0]):
        self.annotation_controller.annotate_class(class_color)

    def undo_redo_annotation(self, last_polygon_area_idx, last_colours, last_window):
        self.annotation_controller.undo_redo_annotation(last_polygon_area_idx, last_colours, last_window)


    def check_existing_annotation(self):
        return self.annotation_controller.check_existing_annotation()
    
    def toggle_show_anno_on_minimap(self):
        self.annotation_controller.toggle_show_anno_on_minimap()


    def label_water(self, bucket_fill=False):
        self.annotation_controller.label_water(bucket_fill)

    def label_ice(self, bucket_fill=False):
        self.annotation_controller.label_ice(bucket_fill)

    def label_shoal(self):
        self.annotation_controller.label_shoal()

    def label_ship(self):
        self.annotation_controller.label_ship()

    def label_iceberg(self):
        self.annotation_controller.label_iceberg()

    def label_unknown(self, bucket_fill=False):
        self.annotation_controller.label_unknown(bucket_fill)

    def bucket_fill(self, event, label):
        self.annotation_controller.bucket_fill(event, label)

    def bucket_fill_polygon_area(self, event):
        self.annotation_controller.bucket_fill_polygon_area(event)

    def exit_bucket_fill(self, event):
        self.annotation_controller.exit_bucket_fill(event)

    # Change this function name later
    def select_area_local_seg(self):
        self.local_seg_controller.select_area_local_seg()

    def toggle_local_seg_source(self):
        self.local_seg_controller.toggle_local_seg_source()

    def clear_local_seg(self):
        self.local_seg_controller.clear_local_seg()

    def update_local_seg_n_classes(self, value):
        self.local_seg_controller.update_local_seg_n_classes(value)

    # Misc

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

