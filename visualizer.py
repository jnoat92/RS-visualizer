'''
No@
June 2025
'''
import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage.measure import find_contours, label

from skimage.morphology import binary_dilation, disk
from evaluation import EvaluationPanel
from annotation import AnnotationPanel
from utils import blend_overlay_cuda, blend_overlay, rgb2gray
from utils import generate_boundaries

from numba import cuda
import cv2
import os
from parallel_stuff import Parallel
import multiprocessing



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

    boundmask = generate_boundaries(rgb2gray(pred))

    return key, pred, landmask, boundmask

def get_segment_contours(pred, x, y):
    target_rgb = pred[x, y]
    mask = np.all(pred == target_rgb, axis=-1)
    labeled = label(mask, connectivity=2)
    region_label = labeled[x, y]

    if region_label == 0:
        return []

    segment_mask = labeled == region_label

    # Get list of contours — each a Nx2 array of [row, col]
    contours = find_contours(segment_mask.astype(np.uint8), level=0.5)

    return contours, segment_mask


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

        # Annotation state
        self.annotation_mode = None  # 'rectangle' or None
        self.selected_polygon = None   # Current canvas item being drawn
        self.reset_annotation()

        # Pan state
        self.select_start = None

        #%% Canvas
        self.canvas = Canvas(self, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down

        self.canvas.bind("<ButtonPress-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)

        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)

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
        ctk.CTkLabel(self.select_image_frame, 
                     text="HH/HV").grid(row=1, column=0, sticky="e", padx=5, pady=5)
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
        self.Segmentation_frame = ctk.CTkFrame(self.control_frame)
        self.Segmentation_frame.grid(row=0, column=1, padx=5)

        self.Opacity_slider_value = 50  # Initial value
        ctk.CTkLabel(self.Segmentation_frame, 
                     text="Opacity").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.slider = ctk.CTkSlider(self.Segmentation_frame,
                                    from_=0, to=100,
                                    number_of_steps=20,
                                    width=100,
                                    command=self.Opacity_slider)
        self.slider.set(self.Opacity_slider_value)  # Set initial value
        self.slider.grid(row=0, column=1, pady=5)

        # Classes ON/OFF
        ctk.CTkLabel(self.Segmentation_frame, text="Ice/Water Labels").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.Segmentation_toggle_state = True
        state = "ON" if self.Segmentation_toggle_state else "OFF"
        self.Segmentation_toggle_btn = ctk.CTkButton(self.Segmentation_frame, text=state, width=19, command=self.Segmentation_toggle)
        self.Segmentation_toggle_btn.grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Zoom selection button
        self.zoom_frame = ctk.CTkFrame(self.control_frame)
        self.zoom_frame.grid(row=0, column=2, padx=5)
        self.zoom_select_btn = ctk.CTkButton(self.zoom_frame, text="Zoom to Selection Mode", width=166, command=self.enable_zoom_selection)
        self.zoom_select_btn.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.zoom_btn_default_style = {     # store default style
            "fg_color": self.zoom_select_btn.cget("fg_color"),
            "hover_color": self.zoom_select_btn.cget("hover_color"),
            "text_color": self.zoom_select_btn.cget("text_color"),
            "font": self.zoom_select_btn.cget("font")
        }
        self.zoom_btn_active_style = {      # define active style
            "fg_color": "#1F6AA5",
            "hover_color": "#3B8ED0",
            "text_color": "white",
            "font": ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        }

        # Reset zoom button
        reset_btn = ctk.CTkButton(self.zoom_frame, text="Reset Zoom", command=self.reset_zoom)
        reset_btn.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        #%% Segmentation source
        self.lbl_source_frame = ctk.CTkFrame(self.bottom_container)
        self.lbl_source_frame.grid(row=0, column=1, sticky="e", padx=5, pady=5)
        ctk.CTkLabel(self.lbl_source_frame, 
                     text="Seg Source").grid(row=0, column=0, sticky="nsew", pady=5)

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
        self.lbl_source_buttom = {}
        self.mode_var_lbl_source = None
        self.mode_var_lbl_source_prev = None
        # Radio buttons for explicit selection
        for i, lbl_s in enumerate(self.lbl_source):
            self.update_label_source_widgets(lbl_s, i)


        #%% Operations
            
        self.operation_frame = ctk.CTkFrame(self.bottom_container)
        self.operation_frame.grid(row=0, column=2, sticky="e", padx=5, pady=5)

        # # # Evaluation panel
        self.evaluation_window = ctk.CTkToplevel(self)
        self.evaluation_window.transient(self)  # Set parent window
        self.evaluation_window.attributes("-topmost", True)  # Always on top
        self.evaluation_window.title("Evaluation Panel")
        self.evaluation_window.withdraw()  # Hide the window at start
        self.evaluation_window.protocol("WM_DELETE_WINDOW", self.close_evaluation_panel) # Hide window instead of destroying it on close

        self.evaluation_panel = EvaluationPanel(self.evaluation_window, self)
        self.evaluation_panel.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkButton(self.operation_frame, text="Evaluation", 
                      command=self.show_evaluation_panel) .grid(row=0, column=0, sticky="nsew", 
                                                                padx=5, pady=5)

        # # # Annotation panel
        self.annotation_window = ctk.CTkToplevel(self)
        self.annotation_window.transient(self)  # Set parent window
        self.annotation_window.attributes("-topmost", True)  # Always on top
        self.annotation_window.title("Annotation Panel")
        self.annotation_window.withdraw()  # Hide the window at start
        self.annotation_window.protocol("WM_DELETE_WINDOW", self.close_annotation_panel) # Hide window instead of destroying it on close

        self.annotation_panel = AnnotationPanel(self.annotation_window, self)
        self.annotation_panel.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkButton(self.operation_frame, text="Annotation", 
                      command=self.show_annotation_panel).grid(row=1, column=0, sticky="nsew", 
                                                               padx=5, pady=5)

        #%% INITIAL VISUALIZATION
        self.folder_path = ''
        self.alpha = 0.5
        self.channel = 0
        
        self._set_all_children_enabled(self.bottom_container, False, exclude=[self.Choose_SAR_scene_toggle_btn])
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(500, self.Choose_SAR_scene)


    # Load images
        
    def update_label_source_widgets(self, lbl_source, i):
        # Radio buttons for explicit selection
        if self.mode_var_lbl_source is None:
            self.mode_var_lbl_source = ctk.StringVar(value=lbl_source)  # Default selection
            self.mode_var_lbl_source_prev = self.mode_var_lbl_source.get()
        self.lbl_source_buttom[lbl_source] = ctk.CTkRadioButton(self.lbl_source_frame, 
                                                                text=lbl_source, 
                                                                variable=self.mode_var_lbl_source,
                                                                value=lbl_source, 
                                                                command=self.Choose_lbl_source)
        self.lbl_source_buttom[lbl_source].grid(row=i+1, column=0, sticky="w", pady=(10, 10))

    def Load_Images(self):
        try:
            self.img_ = [np.tile(np.asarray(Image.open(self.folder_path + "/imagery_HH_UW_4_by_4_average.tif"))[:,:,np.newaxis], (1,1,3)), 
                        np.tile(np.asarray(Image.open(self.folder_path + "/imagery_HV_UW_4_by_4_average.tif"))[:,:,np.newaxis], (1,1,3))]

            self.img_Better_contrast = [np.tile(np.asarray(Image.open(self.folder_path + "/enhanced_images/imagery_HH_UW_4_by_4_average.png"))[:,:,np.newaxis], (1,1,3)),
                                        np.tile(np.asarray(Image.open(self.folder_path + "/enhanced_images/imagery_HV_UW_4_by_4_average.png"))[:,:,np.newaxis], (1,1,3))]
        except FileNotFoundError as e:
            messagebox.showinfo("Error", f"The selected directory does not contain the required files. Please, select a valid directory.\n\n{e}", parent=self.master)
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
        
        # Reset label source radio buttons
        for key in self.lbl_source_buttom.keys():
            self.lbl_source_buttom[key].destroy()
        self.lbl_source_buttom = {}
        self.mode_var_lbl_source = None
        self.mode_var_lbl_source_prev = None

        # Add available label sources
        for i, (key, pred, landmask, boundmask) in enumerate(variables):
            if pred is None: 
                if key != 'Custom_Annotation':
                    messagebox.showinfo("Error", f"The selected scene does not contain prediction files for {key}.", parent=self.master)
                continue
            self.update_label_source_widgets(key, i)
            self.predictions[key] = pred
            self.landmasks[key] = landmask
            self.boundmasks[key] = boundmask


    # Display handle

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
        image = self.overlay if self.Segmentation_toggle_state else self.img_resized.astype('uint8')

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


    # Image selection handle

    def Choose_SAR_scene(self):

        self.close_evaluation_panel()
        self.close_annotation_panel()
        
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
            
            self._set_all_children_enabled(self.bottom_container, True)

        else:
            self.folder_path = prev_folder_path

    def HH_HV(self):
        self.channel = 1 if self.HH_HV_switch.get() else 0

        self.title(f"Scene {self.scene_name}-{"HV" if self.channel else "HH"}")
        
        self.Choose_image()

        self.crop_resize()
        self.Overlay()
        self.display_image()

        if self.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()
        
        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()

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

        if self.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()


    # Segmentation handle

    def Opacity_slider(self, val):
        # self.slider_label.config(text=f"{float(val):.2f}")
        self.alpha = float(val)/100
        self.Overlay()
        self.display_image()

        if self.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()

    def Segmentation_toggle(self):
        self.Segmentation_toggle_state = not self.Segmentation_toggle_state
        state = "ON" if self.Segmentation_toggle_state else "OFF"
        self.Segmentation_toggle_btn.configure(text=state)

        if self.Segmentation_toggle_state:
            # Restore default appearance
            self.Segmentation_toggle_btn.configure(
                fg_color=self.default_fg_color,  # Default customtkinter blue
                hover_color=self.default_hover_color,
                text_color=self.default_text_color
            )
        else:
            # Set to gray when OFF
            self.Segmentation_toggle_btn.configure(
                fg_color="#888888",     # Gray background
                hover_color="#777777",  # Slightly darker on hover
                text_color="white"
            )

        self.display_image()

        if self.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()


    # Zoom handle

    def enable_zoom_selection(self):
        self.zoom_select_mode = True
        self.zoom_select_btn.configure(**self.zoom_btn_active_style)
        self.canvas.config(cursor="crosshair")

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
        if self.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

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
        if self.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()


    # Label source handle

    def Choose_lbl_source(self, plot=True):

        key = self.mode_var_lbl_source.get()


        if self.predictions[key] is None:
            messagebox.showinfo("Error", f"The selected directory does not contain prediction files for {key}.", parent=self.master)
            self.mode_var_lbl_source.set(self.mode_var_lbl_source_prev)
            return 0
        self.mode_var_lbl_source_prev = key

        self.pred = self.predictions[key].copy()
        self.landmask = self.landmasks[key].copy()
        self.boundmask = self.boundmasks[key].copy()

        if plot:
            self.crop_resize()
            self.Overlay()
            self.display_image()

        self.reset_annotation()

        if self.evaluation_window.winfo_viewable():
            self.evaluation_panel.load_existing_evaluation()

        return 1


    # Canvas Events

    def _on_mousewheel(self, event):
        """Handle mouse wheel events for zooming in and out."""
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
        if self.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

    def _on_left_click(self, event):
        """Handle left mouse click for zoom selection, panning, rectangle, or polygon drawing."""
        if self.zoom_select_mode:
            # Start selection
            self.select_start = (event.x, event.y)
            self.selection_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=2)
        elif self.annotation_mode == 'rectangle':
                self.select_start = (event.x, event.y)
                self.selected_polygon = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='yellow', width=2)
        elif self.annotation_mode == 'polygon':
                self._add_polygon_point(event)
        else:
            # Start pan
            self.drag_start = (event.x, event.y)

    def _on_left_drag(self, event):
        """Handle mouse drag for zoom selection, panning, or rectangle drawing."""
        if self.zoom_select_mode and self.select_start:
            # Update selection rectangle
            x0, y0 = self.select_start
            x1, y1 = event.x, event.y
            self.canvas.coords(self.selection_rect, x0, y0, x1, y1)
        elif self.annotation_mode == 'rectangle' and self.select_start:
            x0, y0 = self.select_start
            x1, y1 = event.x, event.y
            self.canvas.coords(self.selected_polygon, x0, y0, x1, y1)
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
            if self.polygon_points_img_coor: 
                self.draw_polygon_on_canvas()

    def _on_left_release(self, event):
        """Handle left mouse button release to finalize zoom selection, rectangle, or polygon drawing."""	
        if self.zoom_select_mode and self.select_start:
            # Complete selection and zoom
            x0, y0 = self.select_start
            x1, y1 = event.x, event.y

            # Reset variables
            self.canvas.delete(self.selection_rect)
            self.selection_rect = None
            self.select_start = None
            self.zoom_select_mode = False
            self.zoom_select_btn.configure(**self.zoom_btn_default_style)
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
        
        elif self.annotation_mode == 'rectangle' and self.select_start:
            x0, y0 = self.select_start
            x1, y1 = event.x, event.y

            polygon_points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # Rectangle points
            for x, y in polygon_points:
                self.polygon_points_img_coor.append((int((x - self.offset_x) / self.zoom_factor),
                                                     int((y - self.offset_y) / self.zoom_factor)))
            self._finish_polygon()
            
            # Reset variables
            self.select_start = None

        elif self.drag_start:
            self.drag_start = None  # end pan

    def on_right_click(self, event):
        """Handle right-click to finish polygon drawing."""
        if self.annotation_mode == 'polygon':
            self._finish_polygon()

    def on_double_click(self, event):
        """Handle double-click to select polygon."""
        if self.annotation_window.winfo_viewable():

            if (hasattr(self.annotation_panel, 'zoom_window') and 
                self.annotation_panel.zoom_window is not None and 
                self.annotation_panel.zoom_window.winfo_exists()):
                if self.annotation_panel.zoom_window.winfo_viewable():            
                    self.annotation_panel.zoom_window.destroy()

            self.annotation_mode = 'selection'
            self.reset_annotation()

            x = int((event.x - self.offset_x) / self.zoom_factor)
            y = int((event.y - self.offset_y) / self.zoom_factor)

            h, w = self.pred.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                return
            
            contours, mask = get_segment_contours(self.pred, y, x)

            # select polygon area on image
            self.selected_polygon_area_idx = [(y, x) for y, x in zip(*np.where(mask))]
            img_y_min = np.asarray(self.selected_polygon_area_idx)[:,0].min()
            img_y_max = np.asarray(self.selected_polygon_area_idx)[:,0].max()
            img_x_min = np.asarray(self.selected_polygon_area_idx)[:,1].min()
            img_x_max = np.asarray(self.selected_polygon_area_idx)[:,1].max()
            self.selected_polygon_window = (img_y_min, img_y_max, img_x_min, img_x_max)
            self.selected_polygon_area_idx = tuple(zip(*self.selected_polygon_area_idx))

            # draw polygon(s) on canvas
            self.polygon_points_img_coor = [[(x, y) for y, x in c] for c in contours]
            self.multiple_polygons = True
            self.draw_polygon_on_canvas()


    # Operations
    
    def show_evaluation_panel(self):
        ann_flag = True
        if self.annotation_window.winfo_viewable():
            ann_flag = self.close_annotation_panel()
        if not ann_flag:
            return
        
        if self.evaluation_panel.scene_name != self.scene_name:
            self.evaluation_panel.set_scene_name(self.scene_name)
        
        self.evaluation_window.deiconify()
        self.evaluation_window.focus_force()
    
    def show_annotation_panel(self):
        eva_flag = True
        if self.evaluation_window.winfo_viewable():
            eva_flag = self.close_evaluation_panel()
        if not eva_flag:
            return
        
        self.annotation_window.deiconify()
        self.annotation_window.focus_force()

    def close_evaluation_panel(self):
        if self.evaluation_panel.unsaved_changes:
            result = messagebox.askyesnocancel("Unsaved Changes", "You have unsaved evaluation data. Do you want to save before exiting?")
            if result is None:
                return  0   # Cancel
            elif result:
                if not self.evaluation_panel.save_evaluation():
                    return  0   # Failed to save → don't close
        
        self.evaluation_panel.scene_name = ""
        self.evaluation_panel.reset_fields()
        self.evaluation_window.withdraw()
        return 1

    def close_annotation_panel(self):
        if self.annotation_panel.unsaved_changes:
            result = messagebox.askyesnocancel("Unsaved Changes", "Your 'Custom Annotation is unsaved'. Do you want to save before exiting?")
            if result is None:
                return  0   # Cancel
            elif result:
                if not self.annotation_panel.save_annotation():
                    return  0   # Failed to save → don't close
        
        # Remove custom annotation from seg sources
        if "Custom_Annotation" in self.lbl_source:
            for i in range(len(self.lbl_source)):
                if self.lbl_source[i] == 'Custom_Annotation':
                    self.lbl_source.pop(i)
                    break
            for key in self.lbl_source_buttom.keys():
                self.lbl_source_buttom[key].destroy()
            self.lbl_source_buttom = {}
            self.mode_var_lbl_source = None
            self.mode_var_lbl_source_prev = None

            for i, lbl_s in enumerate(self.lbl_source):
                self.update_label_source_widgets(lbl_s, i)
            
            self.Choose_lbl_source()

        self.reset_annotation()
        self.annotation_panel.unsaved_changes = False
        self.annotation_window.withdraw()

        return 1


    # Annotation options

    def draw_rectangle(self):
        """Enable rectangle drawing mode."""
        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.zoom_window.destroy()

        self.annotation_mode = 'rectangle'
        self.canvas.config(cursor="crosshair")
        self.reset_annotation()

    def draw_polygon(self):
        """Enable Polygon drawing mode."""
        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.zoom_window.destroy()

        self.annotation_mode = 'polygon'
        self.canvas.config(cursor="crosshair")
        self.reset_annotation()



    def _add_polygon_point(self, event):
        """Add a point to the polygon."""
        if self.annotation_mode == 'polygon':
            self.polygon_points_img_coor.append((int((event.x - self.offset_x) / self.zoom_factor), 
                                                 int((event.y - self.offset_y) / self.zoom_factor)))
            
            self.draw_polygon_on_canvas()
    
    def draw_polygon_on_canvas(self):

        if self.selected_polygon:
            if isinstance(self.selected_polygon, list):
                for poly in self.selected_polygon:
                    self.canvas.delete(poly)
            else:
                self.canvas.delete(self.selected_polygon)
            self.selected_polygon = None

        if not self.multiple_polygons:
            polygon_points_img_coor = [self.polygon_points_img_coor]
        else:
            polygon_points_img_coor = self.polygon_points_img_coor
        
        self.selected_polygon = []
        for p_img_coor in polygon_points_img_coor:
            polygon_points = [
                (x * self.zoom_factor + self.offset_x, 
                 y * self.zoom_factor + self.offset_y) for x, y in p_img_coor
            ]

            self.selected_polygon.append(self.draw_single_polygon_on_canvas(polygon_points))

    def draw_single_polygon_on_canvas(self, polygon_points):
        if len(polygon_points) == 1:
            x, y = polygon_points[0]
            r = 3  # radius for the point
            selected_polygon = self.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill='yellow', outline='yellow'
            )
        elif len(polygon_points) == 2:
            selected_polygon = self.canvas.create_line(
                *polygon_points, fill='yellow', width=2
            )
        elif len(polygon_points) >= 3:
            selected_polygon = self.canvas.create_polygon(
                polygon_points, outline='yellow', width=2, fill=''
            )

        return selected_polygon

    def _finish_polygon(self):
        """Finish drawing a polygon and store it."""
        img_points = self.polygon_points_img_coor
        if len(img_points) >= 3:

            img_x_min = max(0, min(x for x, y in img_points))
            img_y_min = max(0, min(y for x, y in img_points))
            img_x_max = min(self.img.shape[1], max(x for x, y in img_points))
            img_y_max = min(self.img.shape[0], max(y for x, y in img_points))
            if img_x_max > img_x_min and img_y_max > img_y_min:
                self.selected_polygon_window = (img_y_min, img_y_max, img_x_min, img_x_max)

                mask = np.zeros((img_y_max - img_y_min, img_x_max - img_x_min), dtype=np.uint8)
                shifted_points = [(x - img_x_min, y - img_y_min) for x, y in img_points]
                cv2.fillPoly(mask, [np.array(shifted_points, dtype=np.int32)], 255)
                self.selected_polygon_area_idx = [(y + img_y_min, x + img_x_min) for y, x in zip(*np.where(mask==255))]
                self.selected_polygon_area_idx = tuple(zip(*self.selected_polygon_area_idx))

            # Reset variables
            self.annotation_mode = None
            self.canvas.config(cursor="")

    def reset_annotation(self):
        """Reset the annotation state."""
        if self.selected_polygon:
            if isinstance(self.selected_polygon, list):
                for poly in self.selected_polygon:
                    self.canvas.delete(poly)
            else:
                self.canvas.delete(self.selected_polygon)
            self.selected_polygon = None

        self.polygon_points_img_coor = []
        self.selected_polygon_window = None
        self.selected_polygon_area_idx = None
        self.multiple_polygons = False


    def annotate_class(self, class_color):

        if self.selected_polygon_area_idx is None:
            if self.annotation_mode == 'polygon':
                if len(self.polygon_points_img_coor) < 3:
                    messagebox.showinfo("Error", "Polygon incomplete.", parent=self.master)
                    return
                else:
                    self._finish_polygon()
            else:
                messagebox.showinfo("Error", "Please select a polygon area first.", parent=self.master)
                return
        
        # Check if this area is already annotated with the selected class.
        if (self.pred[self.selected_polygon_area_idx] == class_color).all():
            self.reset_annotation()
            return
            
        key = "Custom_Annotation"
        if key not in self.lbl_source_buttom.keys():
            # Add custom annotation as and additional label source
            self.lbl_source.append(key)
            self.filenames.append("{}/{}/{}".format(self.lbl_source[-1], self.scene_name, "custom_annotation.png"))
            self.lbl_source_buttom[key] = ctk.CTkRadioButton(self.lbl_source_frame, 
                                                             text=f"* {key}", 
                                                             variable=self.mode_var_lbl_source, 
                                                             value=key, command=self.Choose_lbl_source)
            self.lbl_source_buttom[key].grid(row=len(self.lbl_source), column=0, sticky="w", pady=(10, 10))
        else:
            self.lbl_source_buttom[key].configure(text=f"* {key}")
        self.annotation_panel.unsaved_changes = True
        self.annotation_panel.save_button.configure(state=ctk.NORMAL)

        self.mode_var_lbl_source.set(key)   # set custom annotation as current label source
        self.pred[self.selected_polygon_area_idx] = class_color
        self.pred[self.landmask] = [255, 255, 255]

        img_y_min, img_y_max, img_x_min, img_x_max = self.selected_polygon_window
        img_y_min = max(0, img_y_min-20)
        img_y_max = min(self.pred.shape[0], img_y_max+20)
        img_x_min = max(0, img_x_min-20)
        img_x_max = min(self.pred.shape[1], img_x_max+20)
        self.boundmask[img_y_min: img_y_max, 
                       img_x_min: img_x_max] = generate_boundaries(rgb2gray(self.pred[img_y_min: img_y_max, 
                                                                                      img_x_min: img_x_max]))
        
        self.predictions[key] = self.pred.copy()
        self.landmasks[key] = self.landmask.copy()
        self.boundmasks[key] = self.boundmask.copy()

        self.crop_resize()
        self.Overlay()
        self.display_image()

        # Reset variables
        self.reset_annotation()

    def label_water(self):
        self.annotate_class([0, 255, 255])

    def label_ice(self):
        self.annotate_class([255, 130, 0])


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
        if self.annotation_window.winfo_viewable():
            ann_flag = self.close_annotation_panel()

        eva_flag = True
        if self.evaluation_window.winfo_viewable():
            eva_flag = self.close_evaluation_panel()
        
        if ann_flag and eva_flag:
            self.destroy()


if __name__ == '__main__':

    multiprocessing.freeze_support()
    
    app = Visualizer()
    app.mainloop()

