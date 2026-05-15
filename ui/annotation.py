'''
Annotation panel setup and functions

Last modified: Mar 2026
'''

from email.mime import text
import customtkinter as ctk
from tkinter import messagebox
import json
import os
import datetime
import csv
from PIL import Image, ImageTk
from tkinter import Canvas
import numpy as np
import cv2
from core.utils import blend_overlay, generate_boundaries, rgb2gray
from core.io import resource_path
from app.state import AppState

class AnnotationPanel(ctk.CTkFrame):
    def __init__(self, parent, command_parent=None):
        super().__init__(parent)
        # self.configure(padx=10, pady=10)

        if command_parent is None:
            command_parent = self
        self.command_parent = command_parent
        self.app_state = command_parent.app_state
        self.annotation_viewmodel = command_parent.annotation_viewmodel

        # Drawing tools frame
        self.drawing_frame = ctk.CTkFrame(self)
        self.drawing_frame.grid(row=0, column=0, padx=5, pady=5, sticky="n")
        ctk.CTkLabel(self.drawing_frame, text="Draw Polygon").grid(row=0, column=0, columnspan=3, sticky="nsew", pady=5)
        ctk.CTkButton(self.drawing_frame, text="", image=ctk.CTkImage(Image.open(resource_path("icons/rectangle.png")), size=(20, 20)), 
                      width=20, command=command_parent.draw_rectangle).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkButton(self.drawing_frame, text="", image=ctk.CTkImage(Image.open(resource_path("icons/polygon.png")), size=(20, 20)), 
                      width=20, command=command_parent.draw_polygon).grid(row=1, column=1, padx=5, pady=5)
        
        # Labels for annotation
        self.labels_frame = ctk.CTkFrame(self)
        self.labels_frame.grid(row=0, column=1, padx=5, pady=5, sticky="n")
        ctk.CTkLabel(self.labels_frame, text="Labels").grid(row=0, column=0, columnspan=5, sticky="nsew", pady=5)
        self.ice_btn = ctk.CTkButton(self.labels_frame, text="ice", width=20, fg_color="#bf803f", text_color="#000000",
                      command=command_parent.label_ice)
        self.ice_btn.grid(row=1, column=0, padx=5, pady=5)
        self.water_btn = ctk.CTkButton(self.labels_frame, text="water", width=20, fg_color="#3fbfbf", text_color="#000000",
                      command=command_parent.label_water)
        self.water_btn.grid(row=1, column=1, padx=5, pady=5)
        self.unknown_btn = ctk.CTkButton(self.labels_frame, text="unknown", width=20, fg_color="#969696", text_color="#000000",
                      command=command_parent.label_unknown)
        self.unknown_btn.grid(row=1, column=2, padx=5, pady=5)

        self.ice_btn.bind("<Button-3>", lambda e: command_parent.bucket_fill(e, "ice"))
        self.water_btn.bind("<Button-3>", lambda e: command_parent.bucket_fill(e, "water"))
        self.unknown_btn.bind("<Button-3>", lambda e: command_parent.bucket_fill(e, "unknown"))

        self.label_btn_default_style = {     # store default style
            "text_color": self.ice_btn.cget("text_color"),
            "font": self.ice_btn.cget("font")
        }
        self.label_btn_active_style = {      # define active style
            "text_color": "white",
            "font": ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        }

        # Other options dropdown
        self.other_options_list = ["shoal", "ship", "iceberg"]
        self.other_options_color = {
            "shoal": "#00ff00",
            "ship": "#ffff00",
            "iceberg": "#ff00ff"
        }
        self.other_options_var = ctk.StringVar(value="Other")
        self.other_options_menu = ctk.CTkOptionMenu(self.labels_frame, values=self.other_options_list, 
                                                    variable=self.other_options_var, command=self.select_other_label, 
                                                    fg_color="#A5A5A5", button_color="#A5A5A5",
                                                    dropdown_fg_color="#A5A5A5", text_color="#000000",
                                                    width=40, corner_radius=10)
        self.other_options_menu.grid(row=1, column=3, padx=5, pady=5)
        ctk.CTkButton(self.labels_frame, text="reset from", 
                      width=20, command=self.reset_label_from).grid(row=1, column=4, padx=5, pady=5)
        ctk.CTkButton(self.labels_frame, text="", width=15, image=ctk.CTkImage(Image.open(resource_path("icons/undo.png")), size=(15, 15)),
                      command=command_parent._on_ctrl_z).grid(row=2, column=0, padx=5, pady=5, columnspan=3, sticky='e')
        ctk.CTkButton(self.labels_frame, text="", width=15, image=ctk.CTkImage(Image.open(resource_path("icons/redo.png")), size=(15, 15)),
                      command=command_parent._on_ctrl_y).grid(row=2, column=3, padx=5, pady=5, columnspan=2, sticky='w')


        # Local segmentation frame
        self.local_seg_frame = ctk.CTkFrame(self)
        self.local_seg_frame.grid(row=0, column=2, padx=5, pady=5, sticky="n")
        ctk.CTkLabel(self.local_seg_frame, text="Local Segmentation").grid(row=0, column=0, columnspan=3, sticky="nsew", pady=5)
        self.local_segmentation_btn = ctk.CTkButton(self.local_seg_frame, text="Select Area", 
                      width=20, command=self.command_parent.select_area_local_seg).grid(row=1, column=0, padx=5, pady=5)
        self.local_seg_switch = ctk.CTkSwitch(
            self.local_seg_frame,
            text="HH-HV",
            command=self.command_parent.toggle_local_seg_source
        )
        self.local_seg_switch.select()  # Default to HV
        self.local_seg_switch.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.local_seg_clear_btn = ctk.CTkButton(self.local_seg_frame, text="Clear Local Seg", 
                      width=20, command=self.command_parent.clear_local_seg).grid(row=1, column=2, padx=5, pady=5)
        
        ctk.CTkLabel(self.local_seg_frame, text="Granularity").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.local_seg_slider = ctk.CTkSlider(self.local_seg_frame, from_=3, to=15, number_of_steps=6, command=self.command_parent.update_local_seg_n_classes)
        self.local_seg_slider.set(15)
        self.local_seg_slider.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

        # Notes frame
        self.notes_frame = ctk.CTkFrame(self)
        self.notes_frame.grid(row=0, column=3, padx=5, pady=5, rowspan=3, sticky="n")
        ctk.CTkLabel(self.notes_frame, text="Notes:").grid(row=0, column=0, sticky="w", padx=10)
        self.notes_text = ctk.CTkTextbox(self.notes_frame, width=300, height=100)
        self.notes_text.grid(row=1, column=0, pady=(0, 5), padx=10)

        # Saving annotations
        self.saving_frame = ctk.CTkFrame(self)
        self.saving_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        self.save_button = ctk.CTkButton(self.saving_frame, text="Save Annotation", 
                                         width=20, command=self.save_annotation)
        self.save_button.grid(row=0, column=0, padx=5, pady=5)

        self.unsaved_changes = False
        self.save_button.configure(state=ctk.NORMAL)

        self.bind("<Escape>", command_parent.exit_bucket_fill)

    def select_other_label(self, choice):
        # Using if, elif for ensuring version compatibility
        if choice == "shoal":
            self.command_parent.label_shoal()
        elif choice == "ship":
            self.command_parent.label_ship()
        elif choice == "iceberg":
            self.command_parent.label_iceberg()
        elif choice == "unknown":
            self.command_parent.label_unknown()
        self.other_options_var.set("Other")

    def reset_label_from(self):
        """Reset the label from available label sources."""
        anno = self.app_state.anno
        scene = self.app_state.scene
        if 'Custom_Annotation' not in scene.lbl_sources:
            messagebox.showinfo("Error", "There is no custom annotation registered.", parent=self.master)
            return

        if anno.selected_polygon_window is None:
            if anno.annotation_mode == 'polygon':
                if len(anno.polygon_points_img_coor) < 3:
                    messagebox.showinfo("Error", "Polygon incomplete.", parent=self.master)
                    return
                else:
                    self.command_parent._finish_polygon()
            else: # Assume they want to reset the whole image
                anno.selected_polygon_window = (0, scene.img.shape[0], 0, scene.img.shape[1])

        # Create new window
        self.zoom_window = ctk.CTkToplevel(self)
        self.zoom_window.transient(self)  # Set parent window
        self.zoom_window.attributes("-topmost", True)  # Always on top
        self.zoom_window.title("Reset annotations from label source")

        screen_width = self.zoom_window.winfo_screenwidth()
        screen_height = self.zoom_window.winfo_screenheight()
        # Use 75% of screen size, for example
        window_width = int(screen_width * 0.40)
        window_height = int(screen_height * 0.40)
        self.zoom_window.geometry(f"{window_width}x{window_height}")
        # self.zoom_window.geometry("800x600")
        self.zoom_window.protocol("WM_DELETE_WINDOW", self.zoom_window.destroy)

        # Canvas for visualization
        self.zoom_canvas = Canvas(self.zoom_window, bg='black')
        self.zoom_canvas.pack(fill=ctk.BOTH, expand=True)

        # Label source selection frame
        self.zoom_lbl_source_frame = ctk.CTkFrame(self.zoom_window)
        self.zoom_lbl_source_frame.pack(side=ctk.BOTTOM, fill=ctk.X, padx=5, pady=5)
        ctk.CTkLabel(self.zoom_lbl_source_frame, text="Seg Source").pack(side=ctk.LEFT, padx=5)

        self.zoom_mode_var_lbl_source = None
        self.zoom_lbl_source_buttons = {}
        for lbl_s in scene.lbl_sources:
            if lbl_s == 'Custom_Annotation':
                continue
            if self.zoom_mode_var_lbl_source is None:
                self.zoom_mode_var_lbl_source = ctk.StringVar(value=lbl_s)
            btn = ctk.CTkRadioButton(self.zoom_lbl_source_frame, text=lbl_s, 
                                   variable=self.zoom_mode_var_lbl_source, value=lbl_s,
                                   command=self.update_zoomed_display)
            btn.pack(side=ctk.LEFT, padx=5)
            self.zoom_lbl_source_buttons[lbl_s] = btn

        # Apply button
        ctk.CTkButton(self.zoom_lbl_source_frame, text="Apply", command=self.apply_label_source).pack(side=ctk.RIGHT, padx=5)

        # Display initial zoomed image
        self.after(100, self.update_zoomed_display)

    def update_zoomed_display(self):
        """Update the zoomed visualization based on current settings."""
        scene = self.app_state.scene
        overlay_state = self.app_state.overlay
        anno = self.app_state.anno

        # Current source not custom annotation
        key = self.zoom_mode_var_lbl_source.get()
        if key not in scene.predictions:
            return

        # Get the selected region
        img_y_min, img_y_max, img_x_min, img_x_max = anno.selected_polygon_window
        img_y_min = max(0, img_y_min-20)
        img_y_max = min(scene.predictions[key].shape[0], img_y_max+20)
        img_x_min = max(0, img_x_min-20)
        img_x_max = min(scene.predictions[key].shape[1], img_x_max+20)

        # Crop images
        pred_crop = scene.predictions[key][img_y_min:img_y_max, img_x_min:img_x_max].astype(np.float32)
        img_crop = scene.img[img_y_min:img_y_max, img_x_min:img_x_max].astype(np.float32)
        boundmask_crop = scene.boundmasks[key][img_y_min:img_y_max, img_x_min:img_x_max]
        landmask_crop = scene.land_nan_masks[key][img_y_min:img_y_max, img_x_min:img_x_max]

        # Resize to fit canvas
        self.zoom_window.update_idletasks()  # Let Tk finish geometry calculation
        canvas_width = self.zoom_canvas.winfo_width()
        canvas_height = self.zoom_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:  # Canvas not yet realized
            canvas_width = 800
            canvas_height = 600

        crop_width = img_x_max - img_x_min
        crop_height = img_y_max - img_y_min
        zoom_factor = min(canvas_width / crop_width, canvas_height / crop_height)

        zoomed_width = max(1, int(crop_width * zoom_factor))
        zoomed_height = max(1, int(crop_height * zoom_factor))

        pred_resized = cv2.resize(pred_crop, (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST)
        img_resized = cv2.resize(img_crop, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
        boundmask_resized = cv2.resize(boundmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        landmask_resized = cv2.resize(landmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)

        # Apply overlay
        overlay = blend_overlay(pred_resized, img_resized, boundmask_resized, 
                                                 landmask_resized, None, overlay_state.alpha)
        image = overlay if overlay_state.show_overlay else img_resized
        image = image.astype(np.uint8)

        # Convert to PhotoImage
        self.zoom_tk_image = ImageTk.PhotoImage(Image.fromarray(image))
        self.zoom_canvas.delete("all")

        self.zoom_canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                      anchor=ctk.CENTER, image=self.zoom_tk_image)

        # Centering offset
        x_offset = (canvas_width - zoomed_width) // 2
        y_offset = (canvas_height - zoomed_height) // 2

        if anno.polygon_points_img_coor:
            # Draw polygon
            if not anno.multiple_polygons:
                polygon_points_img_coor = [anno.polygon_points_img_coor]
            else:
                polygon_points_img_coor = anno.polygon_points_img_coor
            
            for p_img_coor in polygon_points_img_coor:
                polygon_points = [
                    ((x - img_x_min) * zoom_factor + x_offset,
                    (y - img_y_min) * zoom_factor + y_offset)
                    for x, y in p_img_coor
                ]
                self.zoom_canvas.create_polygon(
                    polygon_points, outline='yellow', width=1, fill=''
                )
        # If empty: No polygon to draw, assume whole picture reset


    def apply_label_source(self):
        """Apply the selected label source to the main canvas for the selected area."""
        scene = self.app_state.scene
        anno = self.app_state.anno
        key = self.zoom_mode_var_lbl_source.get()
        if key not in scene.predictions:
            messagebox.showinfo("Error", f"Invalid label source {key}.", parent=self.zoom_window)
            return

        # Update the prediction in the selected area
        if anno.selected_polygon_area_idx:
            if anno.undo_stack and len(anno.undo_stack) > anno.stack_limit:
                anno.undo_stack.pop(0)  # Remove oldest entry if stack limit exceeded
            anno.undo_stack.append((anno.selected_polygon_area_idx, scene.predictions[scene.active_source][anno.selected_polygon_area_idx].copy(), anno.selected_polygon_window))
            
            self.command_parent.deps.widgets["mode_var_lbl_source"].set("Custom_Annotation")
            main_key = self.command_parent.deps.widgets["mode_var_lbl_source"].get()
            # if main_key != "Custom_Annotation":
            #     messagebox.showinfo("Error", "Only 'Custom Annotation' segmentation source can be reset.", parent=self.zoom_window)
            #     return

            scene.predictions[scene.active_source][anno.selected_polygon_area_idx] = \
                scene.predictions[key][anno.selected_polygon_area_idx]
            scene.predictions[scene.active_source][scene.land_nan_masks[scene.active_source]] = [255, 255, 255]

            # Update boundaries
            img_y_min, img_y_max, img_x_min, img_x_max = anno.selected_polygon_window
            img_y_min = max(0, img_y_min-20)
            img_y_max = min(scene.predictions[scene.active_source].shape[0], img_y_max+20)
            img_x_min = max(0, img_x_min-20)
            img_x_max = min(scene.predictions[scene.active_source].shape[1], img_x_max+20)
            scene.boundmasks[scene.active_source][img_y_min:img_y_max, img_x_min:img_x_max] = \
                generate_boundaries(rgb2gray(
                    scene.predictions[scene.active_source][img_y_min:img_y_max, img_x_min:img_x_max]))

            # Update main display
            # Calling refresh_img function from visualzier which consists of crop_resize, set_overlay, display_img
            self.command_parent.refresh_view()
            if anno.polygon_points_img_coor: 
                self.command_parent.draw_polygon_on_canvas()

            # Close the zoom window
            self.zoom_window.destroy()

            self.command_parent.deps.widgets["lbl_source_btn"][main_key].configure(text=f"* {main_key}")
            self.unsaved_changes = True
            #self.save_button.configure(state=ctk.NORMAL)
            changed_area_mask = scene.predictions[scene.active_source][:,:,0] != scene.predictions[scene.lbl_sources[0]][:,:,0]
            self.command_parent.deps.minimap.show_changed_area(scene.img, changed_area_mask)
        else:
            # Whole area reset
            if not messagebox.askyesnocancel("Reset whole annotation", "Please note you are about to reset the entire annotation.\n" \
            "This action cannot be undone.", parent=self.master):
                return
            anno.undo_stack.clear() # Clear undo stack as we are replacing the whole thing, no point in undoing back to previous state
            scene.predictions[scene.active_source] = scene.predictions[key].copy()
            scene.land_nan_masks[scene.active_source] = scene.land_nan_masks[key].copy()
            scene.boundmasks[scene.active_source] = scene.boundmasks[key].copy()
            self.command_parent.refresh_view()

            # Close the zoom window
            self.zoom_window.destroy()

            self.command_parent.deps.widgets["lbl_source_btn"][scene.active_source].configure(text=f"* {scene.active_source}")
            self.unsaved_changes = True
            #self.save_button.configure(state=ctk.NORMAL)
            self.command_parent.deps.minimap.set_image(scene.img)
        
        anno.redo_stack.clear()

    def save_annotation(self):
        """
        Save the current annotation and notes to disk, and mark as saved.
         - Saves the annotation as an image file (.png)
         - Saves the notes in a JSON file with timestamp
         - Also saves a mask of the changed area compared to the original prediction for reference (.png)
        """
        notes = self.notes_text.get("1.0", "end").strip()

        key = "Custom_Annotation"
        try:
            file_path = self.annotation_viewmodel.save_annotation(notes)
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return False

        # mark as saved
        self.command_parent.deps.widgets["lbl_source_btn"][key].configure(text=key)
        self.unsaved_changes = False
        #self.save_button.configure(state=ctk.DISABLED)

        messagebox.showinfo("Saved", f"Evaluation saved to {file_path}", parent=self.master)
        return True

    def insert_existing_notes(self, notes):
        if self.notes_text.get("1.0", "end").strip() == "":
            self.notes_text.insert("1.0", f"{notes}")

    def clear_notes(self):
        self.notes_text.delete("1.0", "end")
        

# if __name__ == '__main__':
#     ctk.set_appearance_mode("System")  # "Dark", "Light", or "System"
#     ctk.set_default_color_theme("blue")  # or "green", "dark-blue", etc.

#     root = ctk.CTk()
#     root.title("Annotation Panel")

#     panel = AnnotationPanel(root)
#     panel.pack(padx=10, pady=10, fill="both", expand=True)


#     root.mainloop()
