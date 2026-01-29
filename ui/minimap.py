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
        self.viewport_item = self.canvas.create_rectangle(0, 0, 0, 0, outline="white", width=2)

        self.stored_area_idx = None  # Store annotated area indices for saving

    def set_image(self, img):
        pil_img = Image.fromarray(img)
        pil_img.thumbnail((self.w, self.h), Image.Resampling.LANCZOS) # high-quality downsampling to fit image into minimap

        # Store original and minimap dimensions
        self.mini_img_w, self.mini_img_h = pil_img.size
        self.full_img_w, self.full_img_h = img.shape[1], img.shape[0]

        if self.stored_area_idx is None:
            self.stored_area_idx = np.zeros((self.full_img_h, self.full_img_w), dtype=np.uint8)

        minimap_img = ImageTk.PhotoImage(pil_img)
        self.tk_img_ref = minimap_img  # keep reference
        
        if self.img_item is None:
            self.img_item = self.canvas.create_image(self.w // 2, self.h // 2, image=minimap_img, anchor="center")
        else:
            self.canvas.itemconfig(self.img_item, image=minimap_img)

        self.canvas.tag_raise(self.viewport_item)  # Ensure viewport rectangle is on top

    def set_viewport_rect(self, image, zoom_factor, offset_x, offset_y, canvas_width, canvas_height):
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
        
        self.canvas.tag_raise(self.viewport_item)

    def get_viewport_coords(self, image, zoom_factor, offset_x, offset_y, canvas_width, canvas_height):
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
        
    def polygon_to_minimap_coords(self, polygon_points):
        # Recieve polygon points as (y_coords, x_coords)
        y_coords, x_coords = polygon_points

        # Might be slower for large polygons, but easier to group by rows and display
        coord_rows = defaultdict(list)
        for x, y in zip(x_coords, y_coords):
            coord_rows[y].append(x)
            self.stored_area_idx[y, x] = 1  # Mark annotated area

        return coord_rows
    
    def show_annotated_area(self, polygon_area_idx, color=[255,255,255]):
        color = "#{:02x}{:02x}{:02x}".format(*color)
        minimap_coord_rows = {}
        if polygon_area_idx is not None:
            minimap_coord_rows = self.polygon_to_minimap_coords(polygon_area_idx)

            # Scale factors
            sx = self.mini_img_w / self.full_img_w
            sy = self.mini_img_h / self.full_img_h

            # Offset factors to center minimap image
            offset_x = (self.w - self.mini_img_w) / 2
            offset_y = (self.h - self.mini_img_h) / 2

            for y, xlist in minimap_coord_rows.items():
                xlist.sort()

                # compress each row into continuous segments
                start = prev = xlist[0]
                for x in xlist[1:] + [None]:
                    if x != prev + 1:
                        mx0 = start * sx + offset_x
                        mx1 = (prev + 1) * sx + offset_x
                        my0 = y * sy + offset_y
                        my1 = (y + 1) * sy + offset_y
                        self.canvas.create_rectangle(
                            mx0, my0, mx1, my1,
                            fill=color, outline="",
                            tags=("annotated_area",)
                        )
                        start = x
                    prev = x
        self.canvas.tag_raise("annotated_area")
        self.canvas.tag_raise(self.viewport_item)

    def save_annotated_area(self, filepath):
        np.savez_compressed(filepath, area_idx=self.stored_area_idx)
