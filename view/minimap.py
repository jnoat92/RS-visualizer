'''
Minimap panel setup and functions

Last modified: Jun 2026
'''

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
from collections import defaultdict
import numpy as np

class Minimap(ctk.CTkFrame):
    def __init__(self, parent, w=180, h=180, **kwargs):
        super().__init__(parent, width=w, height=h, corner_radius=12, **kwargs)
        self.pack_propagate(False)

        self.w, self.h = w, h
        self.canvas = tk.Canvas(self, width=w, height=h, bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.tk_img_ref = None
        self.img_item = None
        self.viewport_item = self.canvas.create_rectangle(0, 0, 0, 0, outline="yellow", width=2)

        self.stored_area_idx = None # Store annotated area indices for saving
        self.full_img_w = None # Placeholder until image is set
        self.full_img_h = None # Placeholder until image is set

        self.show_prev_anno = True # Flag to control whether to show previous annotations on minimap

    def resize_image(self, img):
        """Resize the input image to fit within the minimap dimensions while maintaining aspect ratio."""
        pil_img = Image.fromarray(img.astype('uint8'))
        pil_img.thumbnail((self.w, self.h), Image.Resampling.LANCZOS) # high-quality downsampling to fit image into minimap
        return pil_img

    def set_image(self, img):
        """Set the image to be displayed on the minimap, resizing it to fit while maintaining aspect ratio."""
        pil_img = self.resize_image(img)
        
        # Store original and minimap dimensions
        self.mini_img_w, self.mini_img_h = pil_img.size
        self.full_img_w, self.full_img_h = img.shape[1], img.shape[0]

        if self.stored_area_idx is None:
            self.stored_area_idx = np.zeros((self.full_img_h + 1, self.full_img_w + 1), dtype=np.uint8)

        minimap_img = ImageTk.PhotoImage(pil_img)

        # Keep a reference to the image without annotated areas
        self.tk_img_ref = minimap_img

        self.display_image(minimap_img)

    def display_image(self, minimap_img):
        """Display the given minimap image on the canvas."""
        if self.img_item is None:
            self.img_item = self.canvas.create_image(self.w // 2, self.h // 2, image=minimap_img, anchor="center")
        else:
            self.canvas.itemconfig(self.img_item, image=minimap_img)

        self.canvas.tag_raise(self.viewport_item)  # Ensure viewport rectangle is on top

    def set_viewport_rect(self, image, zoom_factor, offset_x, offset_y, canvas_width, canvas_height):
        """Calculate and update the viewport rectangle on the minimap based on the current view of the main image."""
        view_top, view_bottom, view_left, view_right = self.get_viewport_coords(
            image, zoom_factor, offset_x, offset_y, canvas_width, canvas_height)

        sx = self.mini_img_w / self.full_img_w
        sy = self.mini_img_h / self.full_img_h

        # Convert image coordinates to minimap coordinates
        x0 = view_left * sx
        y0 = view_bottom * sy
        x1 = view_right * sx
        y1 = view_top * sy

        # Center minimap image
        offset_x = (self.w - self.mini_img_w) / 2
        offset_y = (self.h - self.mini_img_h) / 2

        # Update rectangle position
        self.canvas.coords(self.viewport_item, 
                           x0 + offset_x, 
                           y0 + offset_y, 
                           x1 + offset_x, 
                           y1 + offset_y)


    def get_viewport_coords(self, image, zoom_factor, offset_x, offset_y, canvas_width, canvas_height):
        """Calculate the coordinates of the viewport rectangle on the original image based on the current zoom and pan state."""
        h, w = image.shape[:2]
        
        # Image coordinates of the viewport
        img_left = max(0, int(-offset_x / zoom_factor))
        img_top = max(0, int(-offset_y / zoom_factor))
        img_right = min(w, int((canvas_width - offset_x) / zoom_factor))
        img_bottom = min(h, int((canvas_height - offset_y) / zoom_factor))

        # Clamp values to image bounds
        view_top = max(0, min(h-1, img_top))
        view_bottom = max(0, min(h, img_bottom))
        view_left = max(0, min(w-1, img_left))
        view_right = max(0, min(w, img_right))

        return view_top, view_bottom, view_left, view_right

    
    def show_changed_area(self, img, changed_area_mask, color=[255,255,255]):
        """Update only changed areas on image to the specified color"""
        minimap_img = img.copy()
        minimap_img[changed_area_mask] = color
        self.set_image(minimap_img)
