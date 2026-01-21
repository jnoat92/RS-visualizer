'''
Visualizer module for Remote Sensing Visualizer application

Contains the Visualizer class which manages the GUI and image processing.

Last modified: Jan 2026
'''
import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

from ui.evaluation import EvaluationPanel
from ui.annotation import AnnotationPanel
from core.utils import rgb2gray, generate_boundaries
from core.io import load_prediction, load_existing_annotation, load_base_images
from core.segmentation import get_segment_contours
from core.overlay import compose_overlay
from core.render import crop_resize, layer_imagery
from core.contrast_handler import enhance_outlier_slider
from app.state import AppState


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
        self.selected_polygon = None   # Current canvas item being drawn
        self.reset_annotation()


        # ==================== LAYOUT: SIDEBAR (LEFT) + CANVAS (RIGHT)

        # Main container holds sidebar and canvas
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True)

        # Left sidebar for controls
        self.sidebar = ctk.CTkFrame(self.main_container, width=200)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        # self.sidebar.pack_propagate(False)  # Prevent sidebar from resizing to fit contents

        # Canvas on the right
        self.canvas = Canvas(self.main_container, bg="black")
        self.canvas.pack(side="right", fill="both", expand=True)

        # Canvas bindings
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down

        self.canvas.bind("<ButtonPress-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)

        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)


        # ==================== CONTROL PANELS (STACKED VERTICALLY)

        #%% Visualization panel (scene/channel/opacity/zoom)
        self.control_frame = ctk.CTkFrame(self.sidebar)
        self.control_frame.pack(fill="x", padx=5, pady=5)

        # Image selection frame
        self.select_image_frame = ctk.CTkFrame(self.control_frame)
        self.select_image_frame.grid(row=0, column=0, padx=5, pady=(0, 5), sticky="nwe")

        # Choose SAR scene
        self.choose_SAR_scene_toggle_btn = ctk.CTkButton(
            self.select_image_frame,
            text="Choose SAR scene",
            command=self.choose_SAR_scene
        )
        self.choose_SAR_scene_toggle_btn.grid(row=0, column=0, columnspan=2,
                                            sticky="w", padx=5, pady=5)
        
        # Color composite selection
        self.mode_var_color_composite = ctk.StringVar(value=self.app_state.display.channel_mode)  # Default selection
        HH_HV = ctk.CTkRadioButton(self.select_image_frame,
                                      text="(HH/HV)", 
                                      variable=self.mode_var_color_composite,
                                      value="(HH/HV)", 
                                      command=self.color_composite)
        HH_HH_HV = ctk.CTkRadioButton(self.select_image_frame,
                                      text="(HH, HH, HV)", 
                                      variable=self.mode_var_color_composite,
                                      value="(HH, HH, HV)", 
                                      command=self.color_composite)
        HH_HV_HV = ctk.CTkRadioButton(self.select_image_frame,
                                      text="(HH, HV, HV)", 
                                      variable=self.mode_var_color_composite,
                                      value="(HH, HV, HV)", 
                                      command=self.color_composite)
        HH_HV.grid(   row=1, column=0, sticky="w", pady=(10, 10))
        HH_HH_HV.grid(row=2, column=0, sticky="w", pady=(10, 10), columnspan=2)
        HH_HV_HV.grid(row=3, column=0, sticky="w", pady=(10, 10), columnspan=2)

        self.HH_HV_switch = ctk.CTkSwitch(
            self.select_image_frame,
            text="",
            command=self.HH_HV
        )
        self.HH_HV_switch.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Contrast slider
        self.contrast_slider_value = 0  # Initial value
        ctk.CTkLabel(self.select_image_frame, text="Contrast").grid(
            row=5, column=0, sticky="e", padx=5, pady=5
        )
        self.contrast_slider = ctk.CTkSlider(
            self.select_image_frame,
            from_=20,
            to=100,
            number_of_steps=10,
            width=100,
            command=self.contrast_slider_handle
        )
        self.contrast_slider.set(self.contrast_slider_value)  # Set initial value
        self.contrast_slider.grid(row=5, column=1, pady=5, padx=5, sticky="w")
        self.contrast_slider._canvas.bind("<Button-3>", self.right_click_contrast_reset)

        # Brightness slider
        self.brightness_slider_value = 0  # Initial value
        ctk.CTkLabel(self.select_image_frame, text="Brightness").grid(
            row=6, column=0, sticky="e", padx=5, pady=5
        )
        self.brightness_slider = ctk.CTkSlider(
            self.select_image_frame,
            from_=-100,
            to=100,
            number_of_steps=20,
            width=100,
            command=self.brightness_slider_handle
        )
        self.brightness_slider.set(self.brightness_slider_value)  # Set initial value
        self.brightness_slider.grid(row=6, column=1, pady=5, padx=5, sticky="w")
        self.brightness_slider._canvas.bind("<Button-3>", self.right_click_brightness_reset)

        # Opacity + segmentation controls in same block
        self.segmentation_frame = ctk.CTkFrame(self.control_frame)
        self.segmentation_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nwe")

        self.opacity_slider_value = 50  # Initial value
        ctk.CTkLabel(self.segmentation_frame, text="Opacity").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        self.opacity_slider = ctk.CTkSlider(
            self.segmentation_frame,
            from_=0,
            to=100,
            number_of_steps=20,
            width=100,
            command=self.opacity_slider_handle
        )
        self.opacity_slider.set(self.opacity_slider_value)  # Set initial value
        self.opacity_slider.grid(row=0, column=1, pady=5, padx=5, sticky="w")

        # Classes ON/OFF
        ctk.CTkLabel(self.segmentation_frame, text="Ice/Water Labels").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        self.app_state.overlay.show_overlay = True
        state = "ON" if self.app_state.overlay.show_overlay else "OFF"
        self.segmentation_toggle_btn = ctk.CTkButton(
            self.segmentation_frame,
            text=state,
            width=19,
            command=self.segmentation_toggle
        )
        self.segmentation_toggle_btn.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.default_fg_color = self.segmentation_toggle_btn.cget("fg_color")
        self.default_hover_color = self.segmentation_toggle_btn.cget("hover_color")
        self.default_text_color = self.segmentation_toggle_btn.cget("text_color")

        # Zoom controls
        self.zoom_frame = ctk.CTkFrame(self.control_frame)
        self.zoom_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nwe")

        self.zoom_select_btn = ctk.CTkButton(
            self.zoom_frame,
            text="Zoom to Selection Mode",
            width=166,
            command=self.enable_zoom_selection
        )
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
        reset_btn = ctk.CTkButton(
            self.zoom_frame,
            text="Reset Zoom",
            command=self.reset_zoom
        )
        reset_btn.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        #%% Segmentation source (second block in sidebar)
        self.lbl_source_frame = ctk.CTkFrame(self.sidebar)
        self.lbl_source_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(self.lbl_source_frame, text="Seg Source").grid(
            row=0, column=0, sticky="nsew", pady=5
        )

        # Might want to move this out into another file for easy editing
        self.app_state.scene.lbl_sources = [
            "Unet+ITT_pixel",
            # "Unet+ITT_pixel+MV",
            "Unet+ITT_region",
            # "Results_Major"
        ]
        filenames_ = [
            "colored_predict_cnn.png",
            # "CNN_colored_m_v_per_CC.png",
            "colored_predict_transformer.png",
            # "resnet.png"
        ]
        self.app_state.scene.filenames = ["/{}/{}".format(lbl_s, file)
                        for lbl_s, file in zip(self.app_state.scene.lbl_sources, filenames_)]
        self.lbl_source_btn = {}
        self.mode_var_lbl_source = None
        self.mode_var_lbl_source_prev = None

        # Radio buttons for explicit selection
        for i, lbl_s in enumerate(self.app_state.scene.lbl_sources):
            self.update_label_source_widgets(lbl_s, i)

        #%% Operations (third block in sidebar)
        self.operation_frame = ctk.CTkFrame(self.sidebar)
        self.operation_frame.pack(fill="x", padx=5, pady=5)

        # # # Evaluation panel
        self.evaluation_window = ctk.CTkToplevel(self)
        self.evaluation_window.transient(self)  # Set parent window
        self.evaluation_window.attributes("-topmost", True)  # Always on top
        self.evaluation_window.title("Evaluation Panel")
        self.evaluation_window.withdraw()  # Hide the window at start
        self.evaluation_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_evaluation_panel
        )  # Hide window instead of destroying it on close

        self.evaluation_panel = EvaluationPanel(self.evaluation_window, self)
        self.evaluation_panel.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkButton(
            self.operation_frame,
            text="Evaluation",
            command=self.show_evaluation_panel
        ).grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # # # Annotation panel
        self.annotation_window = ctk.CTkToplevel(self)
        self.annotation_window.transient(self)  # Set parent window
        self.annotation_window.attributes("-topmost", True)  # Always on top
        self.annotation_window.title("Annotation Panel")
        self.annotation_window.withdraw()  # Hide the window at start
        self.annotation_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_annotation_panel
        )  # Hide window instead of destroying it on close

        self.annotation_panel = AnnotationPanel(self.annotation_window, self)
        self.annotation_panel.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkButton(
            self.operation_frame,
            text="Annotation",
            command=self.show_annotation_panel
        ).grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Layout behavior inside bottom_container
        self.sidebar.grid_rowconfigure(0, weight=0)
        self.sidebar.grid_rowconfigure(1, weight=0)
        self.sidebar.grid_rowconfigure(2, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)

        #%% INITIAL VISUALIZATION / STATE

        display.channel_mode = self.mode_var_color_composite.get()
        if display.channel_mode == "(HH/HV)":
            display.channel_mode = "HV" if self.HH_HV_switch.get() else "HH"

        # Disable everything until SAR scene is chosen
        self._set_all_children_enabled(
            self.sidebar,
            False,
            exclude=[self.choose_SAR_scene_toggle_btn]
        )

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(500, self.choose_SAR_scene)


    # Load images
        
    def update_label_source_widgets(self, lbl_source, i):
        # Radio buttons for explicit selection
        if self.mode_var_lbl_source is None:
            self.mode_var_lbl_source = ctk.StringVar(value=lbl_source)  # Default selection
            self.mode_var_lbl_source_prev = self.mode_var_lbl_source.get()
        self.lbl_source_btn[lbl_source] = ctk.CTkRadioButton(self.lbl_source_frame, 
                                                                text=lbl_source, 
                                                                variable=self.mode_var_lbl_source,
                                                                value=lbl_source, 
                                                                command=self.choose_lbl_source)
        self.lbl_source_btn[lbl_source].grid(row=i+1, column=0, sticky="w", pady=(10, 10))

    def load_pred(self):

        scene = self.app_state.scene
        anno = self.app_state.anno

        scene.predictions = {}
        scene.landmasks = {}
        scene.boundmasks = {}

        variables = load_prediction(scene.folder_path, scene.filenames, scene.lbl_sources)
        existing_anno, anno.annotation_notes = load_existing_annotation(scene.scene_name)

        if existing_anno is not None:
            variables.append(existing_anno)
            scene.lbl_sources.append("Custom_Annotation")
            scene.filenames.append("{}/{}/{}".format(scene.lbl_sources[-1], scene.scene_name, "custom_annotation.png"))
        self.annotation_panel.clear_notes()
        
        # variables = [PredictionLoader(it) for it in zip(lbl_source, filenames)]
        
        # Reset label source radio buttons
        for key in self.lbl_source_btn.keys():
            self.lbl_source_btn[key].destroy()
        self.lbl_source_btn = {}
        self.mode_var_lbl_source = None
        self.mode_var_lbl_source_prev = None

        # Add available label sources
        for i, (key, pred, landmask, boundmask) in enumerate(variables):
            if pred is None: 
                if key != 'Custom_Annotation':
                    messagebox.showinfo("Error", f"The selected scene does not contain prediction files for {key}.", parent=self.master)
                continue
            self.update_label_source_widgets(key, i)
            scene.predictions[key] = pred
            scene.landmasks[key] = landmask
            scene.boundmasks[key] = boundmask

        custom_anno = "Custom_Annotation"
        if custom_anno in scene.lbl_sources:
            result = messagebox.askyesno("Custom Annotation Found",
                                         "An existing custom annotation was found for this scene. Do you want to view it?",
                                         parent=self.master)
            if result:
                scene.active_source = custom_anno
                self.mode_var_lbl_source.set(custom_anno)


    # Display handle

    def set_overlay(self):
        self.overlay = compose_overlay(self.pred_resized, self.img_resized, self.boundmask_resized, self.landmask_resized, 
                                self.app_state.overlay.alpha)

    def choose_image(self):
        scene = self.app_state.scene
        display = self.app_state.display
        scene.img = self.img_[display.channel_mode]

    def display_image(self):
        image = self.overlay if self.app_state.overlay.show_overlay else self.img_resized.astype('uint8')

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image))

        self.canvas.delete("all")
        self.canvas.create_image(self.draw_x, self.draw_y, anchor=tk.NW, image=self.tk_image)

    
    def refresh_view(self):

        view = self.app_state.view
        scene = self.app_state.scene
        display = self.app_state.display
        # NEXT STEP: Group the returns
        self.pred_resized, self.img_resized, self.boundmask_resized, self.landmask_resized, self.draw_x, self.draw_y = crop_resize(
                    scene.predictions[scene.active_source], scene.img, scene.boundmasks[scene.active_source], scene.landmasks[scene.active_source], 
                    view.zoom_factor, view.offset_x, view.offset_y, display.brightness,
                    self.canvas.winfo_width(), self.canvas.winfo_height())
        self.set_overlay()
        self.display_image()


    # Image selection handle

    def choose_SAR_scene(self):

        scene = self.app_state.scene
        display = self.app_state.display

        self.close_evaluation_panel()
        self.close_annotation_panel()
        
        prev_folder_path = scene.folder_path

        root = ctk.CTk()
        root.withdraw()
        scene.folder_path = filedialog.askdirectory(initialdir=os.path.dirname(prev_folder_path) if scene.folder_path else os.getcwd(),
                                                   title='Select the dated directory containing HH/HV and segmentation results')
        root.destroy()

        if scene.folder_path:

            if scene.folder_path == prev_folder_path:
                return

            scene.scene_name = scene.folder_path.split('/')[-1]

            self.title(f"Scene {scene.scene_name}-{display.channel_mode}")

            raw_img, orig_img, sorted_data, nan_mask = load_base_images(scene.folder_path)
            # Save raw images to app state for later use (e.g., layering)
            scene.raw_img = raw_img
            scene.orig_img = orig_img
            scene.sorted_data = sorted_data
            scene.nan_mask = nan_mask

            if isinstance(raw_img, FileNotFoundError):
                messagebox.showinfo("Error", f"The selected directory does not contain the required files. Please, select a valid directory.\n\n{raw_img}", parent=self.master)
                scene.folder_path = ''
                return
            else:
                self.img_ = orig_img
                
                self.img_["(HH, HH, HV)"] = layer_imagery(
                    orig_img["HH"],
                    orig_img["HV"],
                    stack="(HH, HH, HV)"
                )
                self.img_["(HH, HV, HV)"] = layer_imagery(
                    orig_img["HH"],
                    orig_img["HV"],
                    stack="(HH, HV, HV)"
                )
            
            # Handle switching scenes with existing custom annotation to one without
            if "Custom_Annotation" in scene.lbl_sources:
                scene.filenames.pop()
                scene.lbl_sources.pop()

            self.choose_image()
            self.load_pred()
            if not self.choose_lbl_source(plot=False):
                scene.folder_path = ''
                return
            self.update_idletasks()
            self.after(100, self.reset_zoom)    # Delay the initial reset call with .after() so the canvas has its final size:
            
            self._set_all_children_enabled(self.sidebar, True)
            if display.channel_mode in ["(HH, HH, HV)", "(HH, HV, HV)"]:
                self.HH_HV_switch.configure(state=ctk.DISABLED)

        else:
            scene.folder_path = prev_folder_path

        self.contrast_slider.set(20) # reset to default
        self.app_state.display.contrast = 0.0
        self.brightness_slider.set(0) # reset to default
        self.app_state.display.brightness = 0.0

    def color_composite(self):
        display = self.app_state.display
        display.channel_mode = self.mode_var_color_composite.get()

        if display.channel_mode == "(HH/HV)":
            self.HH_HV_switch.configure(state=ctk.NORMAL)
            self.HH_HV(get_channel=True)
        else:
            self.HH_HV_switch.configure(state=ctk.DISABLED)
            self.HH_HV(get_channel=False)

        
    def HH_HV(self, get_channel=True):
        display = self.app_state.display
        scene = self.app_state.scene

        if get_channel:
            display.channel_mode = "HV" if self.HH_HV_switch.get() else "HH"

        self.contrast_slider.set(0)  # Reset contrast slider
        self.contrast_slider_handle(0)

        self.title(f"Scene {scene.scene_name}-{display.channel_mode}")
        self.choose_image()

        self.refresh_view()

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()

    # Image handle
    def contrast_slider_handle(self, val):
        scene = self.app_state.scene
        display = self.app_state.display
        if val <= 20: # Initial state
            val = 0.0
        display.contrast = float(val)/1000

        if display.channel_mode in ["(HH, HH, HV)", "(HH, HV, HV)"]:
            HH_contrasted = enhance_outlier_slider(
                img=scene.orig_img["HH"], # Pass raw image for faster processing
                sorted_data=scene.sorted_data["HH"],
                land_nan_mask=scene.nan_mask["HH"],
                s=display.contrast,
                s_max=0.1,
                ksize=5,
                output_dtype=np.uint8
            )

            HV_contrasted = enhance_outlier_slider(
                img=scene.orig_img["HV"], # Pass raw image for faster processing
                sorted_data=scene.sorted_data["HV"],
                land_nan_mask=scene.nan_mask["HV"],
                s=display.contrast,
                s_max=0.1,
                ksize=5,
                output_dtype=np.uint8
            )

            # Re-layer the imagery with new contrast
            scene.img = layer_imagery(
                HH_contrasted,
                HV_contrasted,
                display.channel_mode
            )
        else:
            scene.img = enhance_outlier_slider(
                img=scene.orig_img[display.channel_mode], # Pass raw image for faster processing
                sorted_data=scene.sorted_data[display.channel_mode],
                land_nan_mask=scene.nan_mask[display.channel_mode],
                s=display.contrast,
                s_max=0.1,
                ksize=5,
                output_dtype=np.uint8
            )

        self.refresh_view()

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()

    def right_click_contrast_reset(self, event):
        self.contrast_slider.set(20) # reset to default
        self.app_state.display.contrast = 0.0
        self.contrast_slider_handle(20)
        self.refresh_view()

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()

    def brightness_slider_handle(self,val):
        self.app_state.display.brightness = float(val)/100
        self.refresh_view()

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()

    def right_click_brightness_reset(self, event):
        self.brightness_slider.set(0) # reset to default
        self.app_state.display.brightness = 0.0
        self.refresh_view()

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()

    # Segmentation handle

    def opacity_slider_handle(self, val):
        # self.slider_label.config(text=f"{float(val):.2f}")
        self.app_state.overlay.alpha = float(val)/100
        self.set_overlay()
        self.display_image()

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()

    def segmentation_toggle(self):
        overlay_state = self.app_state.overlay
        overlay_state.show_overlay = not overlay_state.show_overlay
        state = "ON" if overlay_state.show_overlay else "OFF"
        self.segmentation_toggle_btn.configure(text=state)

        if overlay_state.show_overlay:
            # Restore default appearance
            self.segmentation_toggle_btn.configure(
                fg_color=self.default_fg_color,  # Default customtkinter blue
                hover_color=self.default_hover_color,
                text_color=self.default_text_color
            )
        else:
            # Set to gray when OFF
            self.segmentation_toggle_btn.configure(
                fg_color="#888888",     # Gray background
                hover_color="#777777",  # Slightly darker on hover
                text_color="white"
            )

        self.display_image()

        if self.app_state.anno.polygon_points_img_coor: 
            self.draw_polygon_on_canvas()

        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.update_zoomed_display()


    # Zoom handle

    def enable_zoom_selection(self):
        view = self.app_state.view
        view.zoom_select_mode = True
        self.zoom_select_btn.configure(**self.zoom_btn_active_style)
        self.canvas.config(cursor="crosshair")

    def zoom_to_rectangle(self, x_min, y_min, x_max, y_max):

        view = self.app_state.view
        rect_width = x_max - x_min
        rect_height = y_max - y_min

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

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

    def reset_zoom(self):

        view = self.app_state.view
        scene = self.app_state.scene
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

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

        scene = self.app_state.scene

        scene.active_source = self.mode_var_lbl_source.get()
        key = scene.active_source

        if scene.predictions[key] is None:
            messagebox.showinfo("Error", f"The selected directory does not contain prediction files for {key}.", parent=self.master)
            self.mode_var_lbl_source.set(self.mode_var_lbl_source_prev)
            return 0
        self.mode_var_lbl_source_prev = key

        if plot:
            self.refresh_view()

        self.reset_annotation()

        if self.evaluation_window.winfo_viewable():
            self.evaluation_panel.load_existing_evaluation()

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
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

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

    def _on_left_click(self, event):
        """Handle left mouse click for zoom selection, panning, rectangle, or polygon drawing."""
        view = self.app_state.view
        anno = self.app_state.anno
        if view.zoom_select_mode:
            # Start selection
            self.selection_start_coord = (event.x, event.y)
            self.selection_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=2)
        elif anno.annotation_mode == 'rectangle':
                self.selection_start_coord = (event.x, event.y)
                self.selected_polygon = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='yellow', width=2)
        elif anno.annotation_mode == 'polygon':
                self._add_polygon_point(event)
        else:
            # Start pan
            view.pan_start_screen = (event.x, event.y)

    def _on_left_drag(self, event):
        """Handle mouse drag for zoom selection, panning, or rectangle drawing."""

        view = self.app_state.view
        anno = self.app_state.anno
        if view.zoom_select_mode and self.selection_start_coord:
            # Update selection rectangle
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y
            self.canvas.coords(self.selection_rect_id, x0, y0, x1, y1)
        elif anno.annotation_mode == 'rectangle' and self.selection_start_coord:
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y
            self.canvas.coords(self.selected_polygon, x0, y0, x1, y1)
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
        anno = self.app_state.anno
        if view.zoom_select_mode and self.selection_start_coord:
            # Complete selection and zoom
            x0, y0 = self.selection_start_coord
            x1, y1 = event.x, event.y

            # Reset variables
            self.canvas.delete(self.selection_rect_id)
            self.selection_rect_id = None
            self.selection_start_coord = None
            view.zoom_select_mode = False
            self.zoom_select_btn.configure(**self.zoom_btn_default_style)
            self.canvas.config(cursor="")

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

            self.zoom_to_rectangle(img_x_min, img_y_min, img_x_max, img_y_max)
        
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

    def on_right_click(self, event):
        """Handle right-click to finish polygon drawing."""
        if self.app_state.anno.annotation_mode == 'polygon':
            self._finish_polygon()

    def on_double_click(self, event):
        """Handle double-click to select polygon."""
        view = self.app_state.view
        scene = self.app_state.scene
        anno = self.app_state.anno
        if self.annotation_window.winfo_viewable():

            if (hasattr(self.annotation_panel, 'zoom_window') and 
                self.annotation_panel.zoom_window is not None and 
                self.annotation_panel.zoom_window.winfo_exists()):
                if self.annotation_panel.zoom_window.winfo_viewable():            
                    self.annotation_panel.zoom_window.destroy()

            anno.annotation_mode = 'selection'
            self.reset_annotation()

            x = int((event.x - view.offset_x) / view.zoom_factor)
            y = int((event.y - view.offset_y) / view.zoom_factor)

            h, w = scene.predictions[scene.active_source].shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                return
            
            contours, mask = get_segment_contours(scene.predictions[scene.active_source], y, x)

            # select polygon area on image
            anno.selected_polygon_area_idx = [(y, x) for y, x in zip(*np.where(mask))]
            img_y_min = np.asarray(anno.selected_polygon_area_idx)[:,0].min()
            img_y_max = np.asarray(anno.selected_polygon_area_idx)[:,0].max()
            img_x_min = np.asarray(anno.selected_polygon_area_idx)[:,1].min()
            img_x_max = np.asarray(anno.selected_polygon_area_idx)[:,1].max()
            anno.selected_polygon_window = (img_y_min, img_y_max, img_x_min, img_x_max)
            anno.selected_polygon_area_idx = tuple(zip(*anno.selected_polygon_area_idx))

            # draw polygon(s) on canvas
            anno.polygon_points_img_coor = [[(x, y) for y, x in c] for c in contours]
            anno.multiple_polygons = True
            self.draw_polygon_on_canvas()


    # Operations
    
    def show_evaluation_panel(self):
        ann_flag = True
        if self.annotation_window.winfo_viewable():
            ann_flag = self.close_annotation_panel()
        if not ann_flag:
            return
        
        if self.evaluation_panel.scene_name != self.app_state.scene.scene_name:
            self.evaluation_panel.set_scene_name(self.app_state.scene.scene_name)
        
        self.evaluation_window.deiconify()
        self.evaluation_window.focus_force()
    
    def show_annotation_panel(self):
        eva_flag = True
        if self.evaluation_window.winfo_viewable():
            eva_flag = self.close_evaluation_panel()
        if not eva_flag:
            return
        
        annotation_loaded = self.check_existing_annotation()

        if annotation_loaded:
            self.annotation_panel.insert_existing_notes(self.app_state.anno.annotation_notes)
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
        scene = self.app_state.scene
        if self.annotation_panel.unsaved_changes:
            result = messagebox.askyesnocancel("Unsaved Changes", "Your 'Custom Annotation is unsaved'. Do you want to save before exiting?")
            if result is None:
                return  0   # Cancel
            elif result:
                if not self.annotation_panel.save_annotation():
                    return  0   # Failed to save → don't close

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

        self.app_state.anno.annotation_mode = 'rectangle'
        self.canvas.config(cursor="crosshair")
        self.reset_annotation()

    def draw_polygon(self):
        """Enable Polygon drawing mode."""
        if (hasattr(self.annotation_panel, 'zoom_window') and 
            self.annotation_panel.zoom_window is not None and 
            self.annotation_panel.zoom_window.winfo_exists()):
            if self.annotation_panel.zoom_window.winfo_viewable():            
                self.annotation_panel.zoom_window.destroy()

        self.app_state.anno.annotation_mode = 'polygon'
        self.canvas.config(cursor="crosshair")
        self.reset_annotation()



    def _add_polygon_point(self, event):
        view = self.app_state.view
        anno = self.app_state.anno
        """Add a point to the polygon."""
        if anno.annotation_mode == 'polygon':
            anno.polygon_points_img_coor.append((int((event.x - view.offset_x) / view.zoom_factor), 
                                                 int((event.y - view.offset_y) / view.zoom_factor)))
            
            self.draw_polygon_on_canvas()
    
    def draw_polygon_on_canvas(self):
        view = self.app_state.view
        anno = self.app_state.anno
        if self.selected_polygon:
            if isinstance(self.selected_polygon, list):
                for poly in self.selected_polygon:
                    self.canvas.delete(poly)
            else:
                self.canvas.delete(self.selected_polygon)
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
            self.canvas.config(cursor="")

    def reset_annotation(self):
        """Reset the annotation state."""
        anno = self.app_state.anno
        if self.selected_polygon:
            if isinstance(self.selected_polygon, list):
                for poly in self.selected_polygon:
                    self.canvas.delete(poly)
            else:
                self.canvas.delete(self.selected_polygon)
            self.selected_polygon = None

        anno.polygon_points_img_coor = []
        anno.selected_polygon_window = None
        anno.selected_polygon_area_idx = None
        anno.multiple_polygons = False


    def annotate_class(self, class_color=[0, 0, 0]):
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
        
        # Check if this area is already annotated with the selected class.
        if (scene.predictions[scene.active_source][anno.selected_polygon_area_idx] == class_color).all():
            self.reset_annotation()
            return
        
        key = "Custom_Annotation"

        scene.predictions[key] = scene.predictions[scene.active_source].copy()
        scene.landmasks[key] = scene.landmasks[scene.active_source].copy()
        scene.boundmasks[key] = scene.boundmasks[scene.active_source].copy()
        scene.active_source = key

        if key not in self.lbl_source_btn.keys():
            # Add custom annotation as and additional label source
            scene.lbl_sources.append(key)
            scene.filenames.append("{}/{}/{}".format(scene.lbl_sources[-1], scene.scene_name, "custom_annotation.png"))
            self.lbl_source_btn[key] = ctk.CTkRadioButton(self.lbl_source_frame, 
                                                                text=f"* {key}", 
                                                                variable=self.mode_var_lbl_source, 
                                                                value=key, command=self.choose_lbl_source)
            self.lbl_source_btn[key].grid(row=len(scene.lbl_sources), column=0, sticky="w", pady=(10, 10))
            
        else:
            self.lbl_source_btn[key].configure(text=f"* {key}")
            
        self.annotation_panel.unsaved_changes = True
        self.annotation_panel.save_button.configure(state=ctk.NORMAL)

        self.mode_var_lbl_source.set(key)   # set custom annotation as current label source
        scene.predictions[scene.active_source][anno.selected_polygon_area_idx] = class_color
        scene.predictions[scene.active_source][scene.landmasks[scene.active_source]] = [255, 255, 255]

        img_y_min, img_y_max, img_x_min, img_x_max = anno.selected_polygon_window
        img_y_min = max(0, img_y_min-20)
        img_y_max = min(scene.predictions[scene.active_source].shape[0], img_y_max+20)
        img_x_min = max(0, img_x_min-20)
        img_x_max = min(scene.predictions[scene.active_source].shape[1], img_x_max+20)
        scene.boundmasks[scene.active_source][img_y_min: img_y_max, 
                    img_x_min: img_x_max] = generate_boundaries(rgb2gray(scene.predictions[scene.active_source][img_y_min: img_y_max, 
                                                                                    img_x_min: img_x_max]))

        self.refresh_view()

        # Reset variables
        self.reset_annotation()

    def check_existing_annotation(self):
        scene = self.app_state.scene
        key = "Custom_Annotation"

        # Duplicate scene for new/updated custom annotation scene
        if key != scene.active_source and key in self.lbl_source_btn.keys():
            result = messagebox.askyesnocancel("Existing annotation", "You have an existing custom annotation. Do you want to use it?")
            if result is None:
                self.reset_annotation()
                return 0 # Cancel
            elif not result:  # No, create new annotation from choice of overlay
                self.annotation_panel.reset_label_from()

            scene.active_source = key
            self.mode_var_lbl_source.set(key)
            self.refresh_view()
        return 1

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


# if __name__ == '__main__':

#     multiprocessing.freeze_support()
    
#     app = Visualizer()
#     app.mainloop()

