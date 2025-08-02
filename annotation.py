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
from utils import blend_overlay, generate_boundaries, rgb2gray

class AnnotationPanel(ctk.CTkFrame):
    def __init__(self, parent, command_parent=None):
        super().__init__(parent)
        # self.configure(padx=10, pady=10)

        if command_parent is None:
            command_parent = self
        
        self.command_parent = command_parent

        # Drawing tools frame
        self.drawing_frame = ctk.CTkFrame(self)
        self.drawing_frame.grid(row=0, column=0, padx=5, pady=5)
        ctk.CTkLabel(self.drawing_frame, text="Draw Polygon").grid(row=0, column=0, columnspan=3, sticky="nsew", pady=5)
        ctk.CTkButton(self.drawing_frame, text="", image=ctk.CTkImage(Image.open("icons/rectangle.png"), size=(20, 20)), 
                      width=20, command=command_parent.draw_rectangle).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkButton(self.drawing_frame, text="", image=ctk.CTkImage(Image.open("icons/polygon.png"), size=(20, 20)), 
                      width=20, command=command_parent.draw_polygon).grid(row=1, column=1, padx=5, pady=5)
        
        # Labels for annotation
        self.labels_frame = ctk.CTkFrame(self)
        self.labels_frame.grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(self.labels_frame, text="Labels").grid(row=0, column=0, columnspan=3, sticky="nsew", pady=5)
        ctk.CTkButton(self.labels_frame, text="ice", width=20, fg_color="#bf803f", text_color="#000000",
                      command=command_parent.label_ice).grid(row=1, column=0, padx=5, pady=5)
        ctk.CTkButton(self.labels_frame, text="water", width=20, fg_color="#3fbfbf", text_color="#000000",
                      command=command_parent.label_water).grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(self.labels_frame, text="reset from", 
                      width=20, command=self.reset_label_from).grid(row=1, column=2, padx=5, pady=5)

    def reset_label_from(self):
        """Reset the label from available label sources."""
        if 'Custom Annotation' not in self.command_parent.lbl_source:
            messagebox.showinfo("Error", "There is no custom annotation registered.", parent=self.master)
            return

        if self.command_parent.selected_polygon_window is None:
            if self.command_parent.annotation_mode == 'polygon':
                if len(self.command_parent.polygon_points_img_coor) < 3:
                    messagebox.showinfo("Error", "Polygon incomplete.", parent=self.master)
                    return
                else:
                    self.command_parent._finish_polygon()
            else:
                messagebox.showinfo("Error", "Please select a polygon area first.", parent=self.master)
                return

        # Create new window
        self.zoom_window = ctk.CTkToplevel(self)
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
        for lbl_s in self.command_parent.lbl_source:
            if lbl_s == 'Custom Annotation':
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
        key = self.zoom_mode_var_lbl_source.get()
        if key not in self.command_parent.predictions:
            return

        # Get the selected region
        img_y_min, img_y_max, img_x_min, img_x_max = self.command_parent.selected_polygon_window
        img_y_min = max(0, img_y_min-20)
        img_y_max = min(self.command_parent.pred.shape[0], img_y_max+20)
        img_x_min = max(0, img_x_min-20)
        img_x_max = min(self.command_parent.pred.shape[1], img_x_max+20)

        # Crop images
        pred_crop = self.command_parent.predictions[key][img_y_min:img_y_max, img_x_min:img_x_max].astype(np.float32)
        img_crop = self.command_parent.img[img_y_min:img_y_max, img_x_min:img_x_max].astype(np.float32)
        boundmask_crop = self.command_parent.boundmasks[key][img_y_min:img_y_max, img_x_min:img_x_max]
        landmask_crop = self.command_parent.landmasks[key][img_y_min:img_y_max, img_x_min:img_x_max]

        # Resize to fit canvas
        # self.zoom_window.update_idletasks()  # Let Tk finish geometry calculation
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
                                                 landmask_resized, self.command_parent.alpha)
        image = overlay if self.command_parent.Segmentation_toggle_state else img_resized
        image = image.astype(np.uint8)

        # Convert to PhotoImage
        self.zoom_tk_image = ImageTk.PhotoImage(Image.fromarray(image))
        self.zoom_canvas.delete("all")
        self.zoom_canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                    anchor=ctk.CENTER, image=self.zoom_tk_image)

    def apply_label_source(self):
        """Apply the selected label source to the main canvas for the selected area."""
        key = self.zoom_mode_var_lbl_source.get()
        if key not in self.command_parent.predictions:
            messagebox.showinfo("Error", f"Invalid label source {key}.", parent=self.zoom_window)
            return

        # Update the prediction in the selected area
        if self.command_parent.selected_polygon_area_idx:
            self.command_parent.pred[self.command_parent.selected_polygon_area_idx] = \
                self.command_parent.predictions[key][self.command_parent.selected_polygon_area_idx]
            self.command_parent.pred[self.command_parent.landmask] = [255, 255, 255]

            # Update boundaries
            img_y_min, img_y_max, img_x_min, img_x_max = self.command_parent.selected_polygon_window
            img_y_min = max(0, img_y_min-20)
            img_y_max = min(self.command_parent.pred.shape[0], img_y_max+20)
            img_x_min = max(0, img_x_min-20)
            img_x_max = min(self.command_parent.pred.shape[1], img_x_max+20)
            self.command_parent.boundmask[img_y_min:img_y_max, img_x_min:img_x_max] = \
                generate_boundaries(rgb2gray(
                    self.command_parent.pred[img_y_min:img_y_max, img_x_min:img_x_max]))

            # Update current label source
            current_key = self.command_parent.mode_var_lbl_source.get()
            self.command_parent.predictions[current_key] = self.command_parent.pred.copy()
            self.command_parent.landmasks[current_key] = self.command_parent.landmask.copy()
            self.command_parent.boundmasks[current_key] = self.command_parent.boundmask.copy()

            # Update main display
            self.command_parent.crop_resize()
            self.command_parent.Overlay()
            self.command_parent.display_image()

            # Close the zoom window
            self.zoom_window.destroy()

    def draw_rectangle(self):
        pass
    def draw_polygon(self):
        pass
    def label_ice(self):
        pass
    def label_water(self):
        pass
        

if __name__ == '__main__':
    ctk.set_appearance_mode("System")  # "Dark", "Light", or "System"
    ctk.set_default_color_theme("blue")  # or "green", "dark-blue", etc.

    root = ctk.CTk()
    root.title("Annotation Panel")

    panel = AnnotationPanel(root)
    panel.pack(padx=10, pady=10, fill="both", expand=True)


    root.mainloop()
