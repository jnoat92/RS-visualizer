'''
No@
June 2025
'''
import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage.measure import find_contours
from skimage.morphology import binary_dilation, disk
from evaluation import EvaluationPanel
from numba import njit, prange, cuda
import cv2
import os
from parallel_stuff import Parallel
import multiprocessing

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

@cuda.jit
def blend_overlay_cuda(pred, img, boundmask, landmask, alpha, out):
    y, x = cuda.grid(2)
    h, w = pred.shape[:2]
    beta = 1 - alpha

    if y < h and x < w:
        if landmask[y, x]:
            for c in range(3):
                out[y, x, c] = 255
        elif boundmask[y, x]:
            for c in range(3):
                out[y, x, c] = pred[y, x, c]
        else:
            for c in range(3):
                out[y, x, c] = alpha * pred[y, x, c] + beta * img[y, x, c]

@njit(parallel=True)
def blend_overlay(pred, img, boundmask, landmask, alpha):
    h, w, c = pred.shape
    out = np.empty((h, w, c), dtype=np.float32)

    beta = 1 - alpha

    for y in prange(h):
        for x in range(w):
            if landmask[y, x]:
                out[y, x, 0] = 255
                out[y, x, 1] = 255
                out[y, x, 2] = 255
            elif boundmask[y, x]:
                out[y, x, :] = pred[y, x, :]
            else:
                for ch in range(c):
                    out[y, x, ch] = alpha * pred[y, x, ch] + beta * img[y, x, ch]

    return out

def PredictionLoader(iterator):
    key, filename = iterator

    try:
        pred = np.asarray(Image.open(filename)).copy()
    except FileNotFoundError as e:
        print(f"The selected directory does not contain the required prediction files. Please, select a valid directory.\n\n{e}")
        return key, None, None, None
    
    pred[(pred == [0, 0, 128]).all(axis=2)] = [0, 255, 255]
    pred[(pred == [128, 0, 0]).all(axis=2)] = [255, 130, 0]
    landmask = (pred == [255, 255, 255]).all(axis=2)

    boundmask = np.zeros_like(landmask, dtype=bool)    
    lbl = rgb2gray(pred)
    for lvl in np.unique(lbl):
        level_ctrs = find_contours(lbl, level=lvl)
        for c in level_ctrs:
            try:
                contours = np.concatenate((contours, c), axis=0)
            except:
                contours = c
    contours = np.uint16(contours)
    boundmask[contours[:,0], contours[:,1]] = True

    return key, pred, landmask, boundmask

class Visualizer(ctk.CTk):

    def __init__(self):

        super().__init__()

        # ==================== GUI DESIGN

        # ------- Visualizer settings
        self.title("Visualizer")

        ctk.set_appearance_mode("System")  # or "Dark", "Light"
        ctk.set_default_color_theme("blue")  # or another theme

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # Use 75% of screen size, for example
        window_width = int(screen_width * 0.80)
        window_height = int(screen_height * 0.80)
        self.geometry(f"{window_width}x{window_height}")

        #%% Initial state
        # Zoom state
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 20.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None
        self.zoom_select_mode = False
        self.selection_rect = None

        # Pan state
        self.select_start = None

        #%% Canvas
        self.canvas = Canvas(self, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down

        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._do_pan)

        self.canvas.bind("<ButtonPress-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)

        # Create a bottom container to hold control frames
        self.bottom_container = ctk.CTkFrame(self)
        self.bottom_container.pack(side=tk.BOTTOM, fill=tk.X)

        #%% Visualization panel
        self.control_frame = ctk.CTkFrame(self.bottom_container)
        self.control_frame.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Image selection frame
        self.select_image_frame = ctk.CTkFrame(self.control_frame)
        self.select_image_frame.grid(row=0, column=0, padx=5)

        # Choose SAR scene
        self.Choose_SAR_scene_toggle_btn = ctk.CTkButton(self.select_image_frame, text=f"Choose SAR scene", command=self.Choose_SAR_scene)
        self.Choose_SAR_scene_toggle_btn.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        # Choose channel (Switch)
        ctk.CTkLabel(self.select_image_frame, text="HH/HV").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.HH_HV_switch = ctk.CTkSwitch(self.select_image_frame, text="", command=self.HH_HV)
        self.HH_HV_switch.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Better Contrast ON/OFF
        ctk.CTkLabel(self.select_image_frame, text="Better contrast").grid(row=2, column=0, sticky="e", padx=5, pady=5)

        self.Better_contrast_toggle_state = True
        state = "ON" if self.Better_contrast_toggle_state else "OFF"
        self.Better_contrast_toggle_btn = ctk.CTkButton(self.select_image_frame, text=state, 
                                                    width=19,
                                                    command=self.Better_contrast_toggle)
        self.Better_contrast_toggle_btn.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.default_fg_color = self.Better_contrast_toggle_btn.cget("fg_color")
        self.default_hover_color = self.Better_contrast_toggle_btn.cget("hover_color")
        self.default_text_color = self.Better_contrast_toggle_btn.cget("text_color")        

        # Opacity slider
        self.labels_frame = ctk.CTkFrame(self.control_frame)
        self.labels_frame.grid(row=0, column=1, padx=5)

        self.Opacity_slider_value = 50  # Initial value
        ctk.CTkLabel(self.labels_frame, text="Opacity").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.slider = ctk.CTkSlider(self.labels_frame,
                                    from_=0, to=100,
                                    number_of_steps=20,
                                    width=100,
                                    command=self.Opacity_slider)
        self.slider.set(self.Opacity_slider_value)  # Set initial value
        self.slider.grid(row=0, column=1, pady=5)

        # Classes ON/OFF
        ctk.CTkLabel(self.labels_frame, text="Ice/Water Labels").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.Polygon_toggle_state = True
        state = "ON" if self.Polygon_toggle_state else "OFF"
        self.Polygon_toggle_btn = ctk.CTkButton(self.labels_frame, text=state, width=19, command=self.Polygon_toggle)
        self.Polygon_toggle_btn.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Zoom selection button
        self.zoom_frame = ctk.CTkFrame(self.control_frame)
        self.zoom_frame.grid(row=0, column=2, padx=5)
        self.zoom_select_btn = ctk.CTkButton(self.zoom_frame, text="Zoom to Selection Mode", command=self.enable_zoom_selection)
        self.zoom_select_btn.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Reset zoom button
        reset_btn = ctk.CTkButton(self.zoom_frame, text="Reset Zoom", command=self.reset_zoom)
        reset_btn.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        #%% Labels source
        self.lbl_source_frame = ctk.CTkFrame(self.bottom_container)
        ctk.CTkLabel(self.lbl_source_frame, text="Label Source", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.lbl_source_frame.grid(row=0, column=1, sticky="e", padx=5, pady=5)

        self.lbl_source = [
            'Unet+ITT_pixel', 
            # 'Unet+ITT_pixel+MV', 
            'Unet+ITT_region', 
            # 'Results_Major'
            ]
        filenames_ = [
                "colored_predict_cnn.png",
                # "CNN_colored_m_v_per_CC.png",
                "colored_predict_transformer.png",
                # "resnet.png"
                ]
        self.filenames = ["/{}/{}".format(lbl_s, file) for lbl_s, file in zip(self.lbl_source, filenames_)]

        # Radio buttons for explicit selection
        self.mode_var_lbl_source = ctk.StringVar(value=self.lbl_source[0])  # Default selection
        self.mode_var_lbl_source_prev = self.mode_var_lbl_source.get()
        for i, lbl_s in enumerate(self.lbl_source):
            ctk.CTkRadioButton(self.lbl_source_frame, text=lbl_s, variable=self.mode_var_lbl_source,
                        value=lbl_s, command=self.Choose_lbl_source).grid(row=i, column=0, sticky="w", pady=(10, 10))

        #%% Evaluation panel
        self.evaluation_frame = ctk.CTkFrame(self.bottom_container)
        self.evaluation_frame.grid(row=0, column=2, sticky="e", padx=5, pady=5)

        self.evaluation_panel = EvaluationPanel(self.evaluation_frame)
        self.evaluation_panel.pack(fill="x", pady=(10, 0))
        self.evaluation_panel.set_enabled(False)


        #%% INITIAL VISUALIZATION
        self.folder_path = ''
        self.alpha = 0.5
        self.channel = 0
        
        self._set_all_children_enabled(self.control_frame, False, exclude=[self.Choose_SAR_scene_toggle_btn])
        self._set_all_children_enabled(self.lbl_source_frame, False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(500, self.Choose_SAR_scene)

        
    def Load_Images(self):
        try:
            self.img_ = [np.tile(np.asarray(Image.open(self.folder_path + "/imagery_HH_UW_4_by_4_average.tif"))[:,:,np.newaxis], (1,1,3)), 
                        np.tile(np.asarray(Image.open(self.folder_path + "/imagery_HV_UW_4_by_4_average.tif"))[:,:,np.newaxis], (1,1,3))]

            self.img_Better_contrast = [np.tile(np.asarray(Image.open(self.folder_path + "/enhanced_images/imagery_HH_UW_4_by_4_average.png"))[:,:,np.newaxis], (1,1,3)),
                                        np.tile(np.asarray(Image.open(self.folder_path + "/enhanced_images/imagery_HV_UW_4_by_4_average.png"))[:,:,np.newaxis], (1,1,3))]
        except FileNotFoundError as e:
            messagebox.showinfo("Error", f"The selected directory does not contain the required files. Please, select a valid directory.\n\n{e}")
            return 0
        return 1

    def Load_pred(self):

        self.predictions = {}
        self.landmasks = {}
        self.boundmasks = {}

        filenames = [self.folder_path + f for f in self.filenames]
        
        if len(self.lbl_source) > 1:
            variables = Parallel(PredictionLoader, zip(self.lbl_source, filenames))
        else:
            variables = [PredictionLoader(zip(self.lbl_source, filenames))]
        
        # variables = [PredictionLoader(it) for it in zip(lbl_source, filenames)]
                
        for key, pred, landmask, boundmask in variables:
            self.predictions[key] = pred
            self.landmasks[key] = landmask
            self.boundmasks[key] = boundmask


    def Overlay_GPU(self):
        # Using Numba on the GPU to parallelize 
        h, w, c = self.pred_resized.shape
        pred = self.pred_resized.astype(np.float32)
        img = self.img_resized.astype(np.float32)

        d_pred = cuda.to_device(pred)
        d_img = cuda.to_device(img)
        d_boundmask = cuda.to_device(self.boundmask_resized)
        d_landmask = cuda.to_device(self.landmask_resized)
        d_out = cuda.device_array((h, w, c), dtype=np.float32)

        threadsperblock = (16, 16)
        blockspergrid_x = (w + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid_y = (h + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid = (blockspergrid_y, blockspergrid_x)

        blend_overlay_cuda[blockspergrid, threadsperblock](d_pred, d_img, d_boundmask, d_landmask, self.alpha, d_out)

        blended = d_out.copy_to_host()
        self.overlay = blended.astype(np.uint8)

    def Overlay(self):

        # alpha = self.alpha
        # beta = 1 - alpha
        # overlay = alpha * self.pred_resized + beta * self.img_resized
        # overlay = np.where(self.boundmask_resized[..., None], self.pred_resized, overlay)
        # overlay = np.where(self.landmask_resized[..., None], 255, overlay)        # Use float32 for fast computation

        # Using Numba on the CPU to parallelize 
        pred = self.pred_resized.astype(np.float32)
        img = self.img_resized.astype(np.float32)
        overlay = blend_overlay(pred, img, self.boundmask_resized, self.landmask_resized, self.alpha)

        self.overlay = overlay.astype(np.uint8)

    def Choose_image(self):
        if self.Better_contrast_toggle_state:
            self.img = self.img_Better_contrast[self.channel]
        else:
            self.img = self.img_[self.channel]

    def display_image(self):
        image = self.overlay if self.Polygon_toggle_state else self.img_resized.astype('uint8')

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image))

        self.canvas.delete("all")
        self.canvas.create_image(self.draw_x, self.draw_y, anchor=tk.NW, image=self.tk_image)

    def crop_resize(self):
        crop = self.get_zoomed_region(self.pred)
        if crop is None:
            return

        view_top, view_bottom, view_left, view_right = crop
        h, w = self.pred.shape[:2]

        # Clamp values to image bounds
        view_top = max(0, min(h-1, view_top))
        view_bottom = max(0, min(h, view_bottom))
        view_left = max(0, min(w-1, view_left))
        view_right = max(0, min(w, view_right))

        if view_bottom <= view_top or view_right <= view_left:
            return  # invalid crop

        pred_crop = self.pred[view_top:view_bottom, view_left:view_right].astype(np.float32)
        img_crop = self.img[view_top:view_bottom, view_left:view_right].astype(np.float32)
        boundmask_crop = self.boundmask[view_top:view_bottom, view_left:view_right]
        landmask_crop = self.landmask[view_top:view_bottom, view_left:view_right]

        # Determine canvas display size
        zoomed_width = max(1, int((view_right - view_left) * self.zoom_factor))
        zoomed_height = max(1, int((view_bottom - view_top) * self.zoom_factor))

        self.pred_resized = cv2.resize(pred_crop, (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST)
        self.img_resized = cv2.resize(img_crop, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
        self.boundmask_resized = cv2.resize(boundmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)
        self.landmask_resized = cv2.resize(landmask_crop.astype(np.uint8), (zoomed_width, zoomed_height), interpolation=cv2.INTER_NEAREST).astype(bool)

        self.boundmask_resized = np.uint8(binary_dilation(self.boundmask_resized.astype('uint8'), np.ones((2,2)).astype('uint8')))

        # Adjust where the image is drawn (canvas position)
        self.draw_x = int(self.offset_x + view_left * self.zoom_factor)
        self.draw_y = int(self.offset_y + view_top * self.zoom_factor)

    def get_zoomed_region(self, image):
        h, w = image.shape[:2]
        
        # Image coordinates of the viewport
        img_left = max(0, int(-self.offset_x / self.zoom_factor))
        img_top = max(0, int(-self.offset_y / self.zoom_factor))
        img_right = min(w, int((self.canvas.winfo_width() - self.offset_x) / self.zoom_factor))
        img_bottom = min(h, int((self.canvas.winfo_height() - self.offset_y) / self.zoom_factor))

        if img_right <= img_left or img_bottom <= img_top:
            return None

        return img_top, img_bottom, img_left, img_right

    def _on_mousewheel(self, event):
        scale = 1.1 if event.delta > 0 or event.num == 4 else 1 / 1.1
        old_zoom = self.zoom_factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * scale))

        if new_zoom == old_zoom:
            return  # no change

        # Mouse position in canvas
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Convert to image coordinates before zoom
        img_x = (canvas_x - self.offset_x) / old_zoom
        img_y = (canvas_y - self.offset_y) / old_zoom

        # Update zoom
        self.zoom_factor = new_zoom

        # Adjust offsets so the image pixel under the cursor stays at the same canvas position
        self.offset_x = canvas_x - img_x * self.zoom_factor
        self.offset_y = canvas_y - img_y * self.zoom_factor

        self.crop_resize()
        self.Overlay()
        self.display_image()

    def _start_pan(self, event):
        if self.zoom_select_mode:
            return  # prevent panning while selecting
        self.drag_start = (event.x, event.y)

    def _do_pan(self, event):
        if self.drag_start:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (event.x, event.y)
            
            self.crop_resize()
            self.Overlay()
            self.display_image()


    def reset_zoom(self):
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Use overlay or base image to get image size
        img_height, img_width = self.img.shape[:2]

        # Compute scale to fit the whole image
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.zoom_factor = min(scale_x, scale_y)

        # Center image in canvas
        new_width = int(img_width * self.zoom_factor)
        new_height = int(img_height * self.zoom_factor)
        self.offset_x = (canvas_width - new_width) // 2
        self.offset_y = (canvas_height - new_height) // 2

        self.crop_resize()
        self.Overlay()
        self.display_image()

    def Polygon_toggle(self):
        self.Polygon_toggle_state = not self.Polygon_toggle_state
        state = "ON" if self.Polygon_toggle_state else "OFF"
        self.Polygon_toggle_btn.configure(text=state)

        if self.Polygon_toggle_state:
            # Restore default appearance
            self.Polygon_toggle_btn.configure(
                fg_color=self.default_fg_color,  # Default customtkinter blue
                hover_color=self.default_hover_color,
                text_color=self.default_text_color
            )
        else:
            # Set to gray when OFF
            self.Polygon_toggle_btn.configure(
                fg_color="#888888",     # Gray background
                hover_color="#777777",  # Slightly darker on hover
                text_color="white"
            )

        self.display_image()

    def Better_contrast_toggle(self):
        self.Better_contrast_toggle_state = not self.Better_contrast_toggle_state
        state = "ON" if self.Better_contrast_toggle_state else "OFF"
        self.Better_contrast_toggle_btn.configure(text=state)

        if self.Better_contrast_toggle_state:
            # Restore default appearance
            self.Better_contrast_toggle_btn.configure(
                fg_color=self.default_fg_color,  # Default customtkinter blue
                hover_color=self.default_hover_color,
                text_color=self.default_text_color
            )
        else:
            # Set to gray when OFF
            self.Better_contrast_toggle_btn.configure(
                fg_color="#888888",     # Gray background
                hover_color="#777777",  # Slightly darker on hover
                text_color="white"
            )

        self.Choose_image()

        self.crop_resize()
        self.Overlay()
        self.display_image()

    def Choose_SAR_scene(self):

        if self.evaluation_panel.unsaved_changes:
            result = messagebox.askyesnocancel("Unsaved Changes", "You have unsaved evaluation data. Do you want to save before loading a new scene?")
            if result is None:  # Cancel
                return
            elif result:
                self.evaluation_panel.save_evaluation()
                if self.evaluation_panel.unsaved_changes:
                    return
                # self.evaluation_panel._show_silent_popup("Saved", "Evaluation has been saved.")

        
        prev_folder_path = self.folder_path

        root = ctk.CTk()
        root.withdraw()
        self.folder_path = filedialog.askdirectory(initialdir=os.path.dirname(prev_folder_path) if self.folder_path else os.getcwd(),
                                                   title='Select the dated directory containing HH/HV and segmentation results')
        root.destroy()

        if self.folder_path:

            if self.folder_path == prev_folder_path:
                return

            self.scene_name = self.folder_path.split('/')[-1]

            self.title(f"Scene {self.scene_name}-{"HV" if self.channel else "HH"}")
            if not self.Load_Images(): 
                self.folder_path = ''
                return
            
            self.Choose_image()
            self.Load_pred()
            if not self.Choose_lbl_source(plot=False):
                self.folder_path = ''
                return
            self.update_idletasks()
            self.after(100, self.reset_zoom)    # Delay the initial reset call with .after() so the canvas has its final size:
            
            self.evaluation_panel.set_scene_name(self.scene_name)
            self.evaluation_panel.set_enabled(True)
            self._set_all_children_enabled(self.control_frame, True)
            self._set_all_children_enabled(self.lbl_source_frame, True)

        else:
            self.folder_path = prev_folder_path
        

    def Choose_lbl_source(self, plot=True):
        key = self.mode_var_lbl_source.get()

        if self.predictions[key] is None:
            messagebox.showinfo("Error", f"The selected directory does not contain prediction files for {key}.")
            self.mode_var_lbl_source.set(self.mode_var_lbl_source_prev)
            return 0
        self.mode_var_lbl_source_prev = key

        self.pred = self.predictions[key]
        self.landmask = self.landmasks[key]
        self.boundmask = self.boundmasks[key]

        if plot:
            self.crop_resize()
            self.Overlay()
            self.display_image()
        
        return 1

    def Opacity_slider(self, val):
        # self.slider_label.config(text=f"{float(val):.2f}")
        self.alpha = float(val)/100
        self.Overlay()
        self.display_image()

    def HH_HV(self):
        self.channel = 1 if self.HH_HV_switch.get() else 0

        self.title(f"Scene {self.scene_name}-{"HV" if self.channel else "HH"}")
        
        self.Choose_image()

        self.crop_resize()
        self.Overlay()
        self.display_image()


    def enable_zoom_selection(self):
        self.zoom_select_mode = True
        self.canvas.config(cursor="crosshair")

    def _on_left_click(self, event):
        if self.zoom_select_mode:
            # Start selection
            self.select_start = (event.x, event.y)
            self.selection_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=2)
        else:
            # Start pan
            self.drag_start = (event.x, event.y)

    def _on_left_drag(self, event):
        if self.zoom_select_mode and self.select_start:
            # Update selection rectangle
            x0, y0 = self.select_start
            x1, y1 = event.x, event.y
            self.canvas.coords(self.selection_rect, x0, y0, x1, y1)
        elif self.drag_start:
            # Pan mode
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (event.x, event.y)
            
            self.crop_resize()
            self.Overlay()
            self.display_image()

    def _on_left_release(self, event):
        if self.zoom_select_mode and self.select_start:
            # Complete selection and zoom
            x0, y0 = self.select_start
            x1, y1 = event.x, event.y
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
            self.select_start = None
            self.zoom_select_mode = False
            self.canvas.config(cursor="")

            # Convert canvas to image coords
            x_min = min(x0, x1)
            y_min = min(y0, y1)
            x_max = max(x0, x1)
            y_max = max(y0, y1)

            if x_max - x_min < 10 or y_max - y_min < 10:
                return  # too small

            img_x_min = int((x_min - self.offset_x) / self.zoom_factor)
            img_y_min = int((y_min - self.offset_y) / self.zoom_factor)
            img_x_max = int((x_max - self.offset_x) / self.zoom_factor)
            img_y_max = int((y_max - self.offset_y) / self.zoom_factor)

            img_x_min = max(0, img_x_min)
            img_y_min = max(0, img_y_min)
            img_x_max = min(self.img.shape[1], img_x_max)
            img_y_max = min(self.img.shape[0], img_y_max)

            self.zoom_to_rectangle(img_x_min, img_y_min, img_x_max, img_y_max)

        elif self.drag_start:
            self.drag_start = None  # end pan

    def zoom_to_rectangle(self, x_min, y_min, x_max, y_max):
        rect_width = x_max - x_min
        rect_height = y_max - y_min

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        zoom_x = canvas_width / rect_width
        zoom_y = canvas_height / rect_height
        self.zoom_factor = min(zoom_x, zoom_y, self.max_zoom)

        center_x = x_min + rect_width / 2
        center_y = y_min + rect_height / 2

        self.offset_x = int(canvas_width / 2 - center_x * self.zoom_factor)
        self.offset_y = int(canvas_height / 2 - center_y * self.zoom_factor)

        self.crop_resize()
        self.Overlay()
        self.display_image()

    
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
        if self.evaluation_panel.unsaved_changes:
            result = messagebox.askyesnocancel("Unsaved Changes", "You have unsaved evaluation data. Do you want to save before exiting?")
            if result is None:
                return  # Cancel
            elif result:
                if not self.evaluation_panel.save_evaluation():
                    return  # Failed to save → don't close
        self.destroy()

if __name__ == '__main__':

    multiprocessing.freeze_support()
    
    app = Visualizer()
    app.mainloop()

