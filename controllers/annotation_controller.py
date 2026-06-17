'''
AnnotationController manages the annotation process, including handling 
user interactions for drawing and labeling. It contains methods for
polygon drawing, rectangle drawing, bucket fill, and label assignment.
It also manages the iteractions with local segmentation and 
undo/redo functionality.

Last modified: Jun 2026
'''

import numpy as np
import cv2
import customtkinter as ctk
from tkinter import messagebox
from model.render import layer_imagery


class AnnotationController:
    def __init__(self, deps, display_controller, annotation_viewmodel):
        self.deps = deps
        self.display_controller = display_controller
        self.annotation_viewmodel = annotation_viewmodel

    def draw_rectangle(self):
        """Enable rectangle drawing mode."""
        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                self.deps.annotation_panel.zoom_window.destroy()

        self.deps.app_state.anno.annotation_mode = 'rectangle'
        self.deps.canvas.config(cursor="crosshair")
        self.reset_annotation()

    def draw_polygon(self):
        """Enable Polygon drawing mode."""
        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                self.deps.annotation_panel.zoom_window.destroy()

        self.deps.app_state.anno.annotation_mode = 'polygon'
        self.deps.canvas.config(cursor="crosshair")
        self.reset_annotation()



    def add_polygon_point(self, event):
        """Add a point to the polygon."""
        view = self.deps.app_state.view
        anno = self.deps.app_state.anno
        if anno.annotation_mode == 'polygon':
            anno.polygon_points_img_coor.append((int((event.x - view.offset_x) / view.zoom_factor), 
                                                 int((event.y - view.offset_y) / view.zoom_factor)))
            
            self.draw_polygon_on_canvas()
    
    def draw_polygon_on_canvas(self):
        """Draw the polygon defined by the image coordinates on the canvas, converting to canvas coordinates."""
        view = self.deps.app_state.view
        anno = self.deps.app_state.anno
        # Remove existing polygon if exists before drawing new one
        if anno.selected_polygon:
            if isinstance(anno.selected_polygon, list):
                for poly in anno.selected_polygon:
                    self.deps.canvas.delete(poly)
            else:
                self.deps.canvas.delete(anno.selected_polygon)
            anno.selected_polygon = None

        if not anno.multiple_polygons:
            polygon_points_img_coor = [anno.polygon_points_img_coor]
        else:
            polygon_points_img_coor = anno.polygon_points_img_coor
        
        anno.selected_polygon = []
        for p_img_coor in polygon_points_img_coor:
            polygon_points = [
                (x * view.zoom_factor + view.offset_x, 
                 y * view.zoom_factor + view.offset_y) for x, y in p_img_coor
            ]

            anno.selected_polygon.append(self.draw_single_polygon_on_canvas(polygon_points))

        if self.deps.canvas.find_withtag("polygon") and not self.deps.app_state.overlay.show_overlay:
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

    def finish_polygon(self):
        """Finish drawing a polygon and store it."""
        scene = self.deps.app_state.scene
        anno = self.deps.app_state.anno
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
        anno = self.deps.app_state.anno
        if anno.selected_polygon:
            if isinstance(anno.selected_polygon, list):
                for poly in anno.selected_polygon:
                    self.deps.canvas.delete(poly)
            else:
                self.deps.canvas.delete(anno.selected_polygon)
            anno.selected_polygon = None

        anno.polygon_points_img_coor = []
        anno.selected_polygon_window = None
        anno.selected_polygon_area_idx = None
        anno.multiple_polygons = False


    def annotate_class(self, class_color=[0, 0, 0]):
        """
        Annotate the selected polygon area with the specified class color, update the Custom Annotation layer, 
        handle undo/redo stacks, and refresh the display.
        """
        scene = self.deps.app_state.scene
        anno = self.deps.app_state.anno
        display = self.deps.app_state.display

        if anno.selected_polygon_area_idx is None:
            if anno.annotation_mode == 'polygon':
                if len(anno.polygon_points_img_coor) < 3:
                    messagebox.showinfo("Error", "Polygon incomplete.", parent=self.deps.app.master)
                    return
                else:
                    self.finish_polygon()
            else:
                messagebox.showinfo("Error", "Please select a polygon area first.", parent=self.deps.app.master)
                return
        elif scene.land_nan_masks[scene.active_source][anno.selected_polygon_area_idx].all():
            messagebox.showinfo("Error", "Selected area is land or invalid data.", parent=self.deps.app.master)
            self.reset_annotation()
            return
        
        # Check if this area is already annotated with the selected class.
        if self.annotation_viewmodel.selected_area_matches_color(class_color):
            return
        
        key = "Custom_Annotation"

        if key not in self.deps.widgets["lbl_source_btn"].keys():
            self.annotation_viewmodel.ensure_custom_annotation_source()
            # Add custom annotation as and additional label source
            self.deps.widgets["lbl_source_btn"][key] = ctk.CTkRadioButton(self.deps.widgets["lbl_source_frame"], 
                                                                text=f"* {key}", 
                                                                variable=self.deps.widgets["mode_var_lbl_source"], 
                                                                value=key, command=self.deps.app.choose_lbl_source)
            self.deps.widgets["lbl_source_btn"][key].grid(row=len(scene.lbl_sources), column=0, sticky="w", pady=(10, 10))
            
        else:
            self.deps.widgets["lbl_source_btn"][key].configure(text=f"* {key}")
            
        self.deps.annotation_panel.unsaved_changes = True
        self.deps.annotation_panel.save_button.configure(state=ctk.NORMAL)

        self.deps.widgets["mode_var_lbl_source"].set(key)   # set custom annotation as current label source
        self.annotation_viewmodel.apply_class_to_selection(class_color)

        # Do a vectorized compare of the existing prediction (only 1 so [0]) with the new prediction to get a mask of the changed area
        if self.deps.widgets["show_prev_anno_switch"].get():
            changed_area_mask = self.annotation_viewmodel.changed_area_mask()
            self.deps.minimap.show_changed_area(scene.color_composites[display.channel_mode], changed_area_mask)

        self.display_controller.refresh_view()
        if anno.polygon_points_img_coor: 
                self.draw_polygon_on_canvas()

    def undo_redo_annotation(self, last_polygon_area_idx, last_colours, last_window):
        """Undo or redo an annotation by restoring the previous state."""
        scene = self.deps.app_state.scene
        display = self.deps.app_state.display
        self.annotation_viewmodel.undo_redo_annotation(last_polygon_area_idx, last_colours, last_window)
        # Show annotated area on minimap
        if self.deps.widgets["show_prev_anno_switch"].get():
            changed_area_mask = self.annotation_viewmodel.changed_area_mask()
            self.deps.minimap.show_changed_area(scene.color_composites[display.channel_mode], changed_area_mask)

        # Reset annotation and refresh view
        self.reset_annotation()
        self.display_controller.refresh_view()


    def check_existing_annotation(self):
        """
        Check for existing custom annotation, prompt user to use it or create new annotation from the prediction, 
        and set active source to custom annotation if not canceled.
        """
        scene = self.deps.app_state.scene
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
            self.display_controller.refresh_view()
        return 1
    
    def toggle_show_anno_on_minimap(self):
        """
        Toggle the display of the annotated area on the minimap by comparing the 
        current annotation with the original prediction and showing the changed area 
        if the switch is on, or resetting to the original image if the switch is off.
        """
        scene = self.deps.app_state.scene
        display = self.deps.app_state.display
        custom_anno = "Custom_Annotation"

        if custom_anno in scene.lbl_sources and self.deps.widgets["show_prev_anno_switch"].get():
            changed_area_mask = scene.predictions[custom_anno][:,:,0] != scene.predictions[scene.lbl_sources[0]][:,:,0]
            self.deps.minimap.show_changed_area(scene.color_composites[display.channel_mode], changed_area_mask)
        else:
            self.deps.minimap.set_image(scene.color_composites[display.channel_mode])


    def label_water(self, bucket_fill=False):
        """Label selected polygon as water with specified color, check if called from bucket fill."""
        # Check if called by left click or bucket fill
        if not bucket_fill and self.deps.app_state.anno.annotation_mode == 'bucket_fill':
            self.exit_bucket_fill(None)
        self.annotate_class([0, 255, 255])

    def label_ice(self, bucket_fill=False):
        """Label selected polygon as ice with specified color, check if called from bucket fill."""
        if not bucket_fill and self.deps.app_state.anno.annotation_mode == 'bucket_fill':
            self.exit_bucket_fill(None)
        self.annotate_class([255, 130, 0])

    def label_shoal(self):
        self.annotate_class([0, 255, 0])

    def label_ship(self):
        self.annotate_class([255, 255, 0])

    def label_iceberg(self):
        self.annotate_class([255, 0, 255])

    def label_unknown(self, bucket_fill=False):
        if not bucket_fill and self.deps.app_state.anno.annotation_mode == 'bucket_fill':
            self.exit_bucket_fill(None)
        self.annotate_class([150, 150, 150])

    def bucket_fill(self, event, label):
        """Enable bucket fill mode for the specified label, set cursor, and update annotation panel button styles."""
        anno = self.deps.app_state.anno
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
        
        self.deps.app.focus_set()

    def bucket_fill_polygon_area(self, event):
        """Perform bucket fill annotation on the selected polygon area based on the active label."""
        anno = self.deps.app_state.anno
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
        anno = self.deps.app_state.anno
        anno.annotation_mode = None
        anno.active_label = None
        self.deps.canvas.config(cursor="")
        self.deps.annotation_panel.water_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
        self.deps.annotation_panel.ice_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
        self.deps.annotation_panel.unknown_btn.configure(**self.deps.annotation_panel.label_btn_default_style)
