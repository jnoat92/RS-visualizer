from dataclasses import dataclass
import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas

from ui.evaluation import EvaluationPanel
from ui.annotation import AnnotationPanel
from ui.minimap import Minimap

@dataclass
class VisualizerLayout:
    canvas: Canvas
    sidebar: ctk.CTkFrame
    minimap: Minimap
    minimap_window_id: int
    status_bar: ctk.CTkLabel
    annotation_panel: AnnotationPanel
    evaluation_panel: EvaluationPanel
    annotation_window: ctk.CTkToplevel
    evaluation_window: ctk.CTkToplevel
    loading_bar: ctk.CTkProgressBar
    loading_bar_label: ctk.CTkLabel
    widgets: dict

def build_visualizer_layout(app, app_state) -> VisualizerLayout:
    widgets = {}

    main_container = ctk.CTkFrame(app)
    main_container.pack(fill="both", expand=True)

    sidebar = ctk.CTkFrame(main_container, width=200)
    sidebar.pack(side="left", fill="y", padx=0, pady=0)

    canvas = Canvas(main_container, bg="black")
    canvas.pack(side="right", fill="both", expand=True)

    # build control frames, sliders, radio buttons, buttons, windows...
    # store them in widgets dict

    # Canvas bindings
    canvas.bind("<MouseWheel>", app._on_mousewheel)  # Windows
    canvas.bind("<Button-4>", app._on_mousewheel)    # Linux scroll up
    canvas.bind("<Button-5>", app._on_mousewheel)    # Linux scroll down

    canvas.bind("<ButtonPress-1>", app._on_left_click_await)
    canvas.bind("<B1-Motion>", app._on_left_drag)
    canvas.bind("<ButtonRelease-1>", app._on_left_release)

    canvas.bind("<Button-3>", app._on_right_click)
    canvas.bind("<Double-Button-1>", app._on_double_click_set_flag)
    canvas.bind("<Motion>", app._on_mouse_move)

    app.bind("<Escape>", app._on_escape_key)

    app.bind("<Control-z>", app._on_ctrl_z)
    app.bind("<Control-y>", app._on_ctrl_y)
    app.bind("<Control-Shift-Z>", app._on_ctrl_y)  # Some systems use Ctrl+Shift+Z for redo

    double_click_flag = False


    # ==================== CONTROL PANELS (STACKED VERTICALLY)

    #%% Visualization panel (scene/channel/opacity/zoom)
    control_frame = ctk.CTkFrame(sidebar)
    control_frame.pack(fill="x", padx=5, pady=5)
    widgets['control_frame'] = control_frame

    # Image selection frame
    select_image_frame = ctk.CTkFrame(control_frame)
    select_image_frame.grid(row=0, column=0, padx=5, pady=(0, 5), sticky="nwe")
    widgets['select_image_frame'] = select_image_frame

    # Choose SAR scene
    choose_SAR_scene_toggle_btn = ctk.CTkButton(
        select_image_frame,
        text="Choose SAR scene",
        command=app.choose_SAR_scene
    )
    choose_SAR_scene_toggle_btn.grid(row=0, column=0, columnspan=2,
                                        sticky="w", padx=5, pady=5)
    widgets['choose_SAR_scene_toggle_btn'] = choose_SAR_scene_toggle_btn
    
    # Color composite selection
    mode_var_color_composite = ctk.StringVar(value=app_state.display.channel_mode)  # Default selection
    HH_HV = ctk.CTkRadioButton(select_image_frame,
                                    text="(HH/HV)", 
                                    variable=mode_var_color_composite,
                                    value="(HH/HV)", 
                                    command=app.color_composite)
    HH_HH_HV = ctk.CTkRadioButton(select_image_frame,
                                    text="(HH, HH, HV)", 
                                    variable=mode_var_color_composite,
                                    value="(HH, HH, HV)", 
                                    command=app.color_composite)
    HH_HV_HV = ctk.CTkRadioButton(select_image_frame,
                                    text="(HH, HV, HV)", 
                                    variable=mode_var_color_composite,
                                    value="(HH, HV, HV)", 
                                    command=app.color_composite)
    HH_HV.grid(   row=1, column=0, sticky="w", pady=(10, 10))
    HH_HH_HV.grid(row=2, column=0, sticky="w", pady=(10, 10), columnspan=2)
    HH_HV_HV.grid(row=3, column=0, sticky="w", pady=(10, 10), columnspan=2)
    widgets['mode_var_color_composite'] = mode_var_color_composite
    widgets['hh_hv_radio'] = HH_HV
    widgets['hh_hh_hv_radio'] = HH_HH_HV
    widgets['hh_hv_hv_radio'] = HH_HV_HV

    HH_HV_switch = ctk.CTkSwitch(
        select_image_frame,
        text="",
        command=app.HH_HV
    )
    HH_HV_switch.grid(row=1, column=1, sticky="w", padx=5, pady=5)
    widgets['hh_hv_switch'] = HH_HV_switch

    # Contrast slider
    contrast_slider_value = 0  # Initial value
    ctk.CTkLabel(select_image_frame, text="Contrast").grid(
        row=5, column=0, sticky="e", padx=5, pady=5
    )
    contrast_slider = ctk.CTkSlider(
        select_image_frame,
        from_=0,
        to=200,
        number_of_steps=200,
        width=100,
        command=app.contrast_slider_handle
    )
    contrast_slider.set(contrast_slider_value)  # Set initial value
    contrast_slider.grid(row=5, column=1, pady=5, padx=5, sticky="w")
    contrast_slider._canvas.bind("<Button-3>", app.right_click_contrast_reset)
    widgets['contrast_slider'] = contrast_slider

    # Brightness slider
    brightness_slider_value = 0  # Initial value
    ctk.CTkLabel(select_image_frame, text="Brightness").grid(
        row=6, column=0, sticky="e", padx=5, pady=5
    )
    brightness_slider = ctk.CTkSlider(
        select_image_frame,
        from_=-100,
        to=100,
        number_of_steps=20,
        width=100,
        command=app.brightness_slider_handle
    )
    brightness_slider.set(brightness_slider_value)  # Set initial value
    brightness_slider.grid(row=6, column=1, pady=5, padx=5, sticky="w")
    brightness_slider._canvas.bind("<Button-3>", app.right_click_brightness_reset)
    widgets['brightness_slider'] = brightness_slider

    # Opacity + segmentation controls in same block
    segmentation_frame = ctk.CTkFrame(control_frame)
    segmentation_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nwe")
    widgets['segmentation_frame'] = segmentation_frame

    opacity_slider_value = 50  # Initial value
    ctk.CTkLabel(segmentation_frame, text="Opacity").grid(
        row=0, column=0, sticky="e", padx=5, pady=5
    )
    opacity_slider = ctk.CTkSlider(
        segmentation_frame,
        from_=0,
        to=100,
        number_of_steps=20,
        width=100,
        command=app.opacity_slider_handle
    )
    opacity_slider.set(opacity_slider_value)  # Set initial value
    opacity_slider.grid(row=0, column=1, pady=5, padx=5, sticky="w")
    widgets['opacity_slider'] = opacity_slider

    # Classes ON/OFF
    ctk.CTkLabel(segmentation_frame, text="Ice/Water Labels").grid(
        row=1, column=0, sticky="e", padx=5, pady=5
    )
    app_state.overlay.show_overlay = True
    state = "ON" if app_state.overlay.show_overlay else "OFF"
    segmentation_toggle_btn = ctk.CTkButton(
        segmentation_frame,
        text=state,
        width=19,
        command=app.segmentation_toggle
    )
    segmentation_toggle_btn.grid(row=1, column=1, sticky="w", padx=5, pady=5)
    default_fg_color = segmentation_toggle_btn.cget("fg_color")
    default_hover_color = segmentation_toggle_btn.cget("hover_color")
    default_text_color = segmentation_toggle_btn.cget("text_color")
    widgets['segmentation_toggle_btn'] = segmentation_toggle_btn
    widgets['default_fg_color'] = default_fg_color
    widgets['default_hover_color'] = default_hover_color
    widgets['default_text_color'] = default_text_color

    # Zoom controls
    zoom_frame = ctk.CTkFrame(control_frame)
    zoom_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nwe")
    widgets['zoom_frame'] = zoom_frame

    zoom_select_btn = ctk.CTkButton(
        zoom_frame,
        text="Zoom to Selection Mode",
        width=166,
        command=app.enable_zoom_selection
    )
    zoom_select_btn.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    widgets['zoom_select_btn'] = zoom_select_btn

    zoom_btn_default_style = {     # store default style
        "fg_color": zoom_select_btn.cget("fg_color"),
        "hover_color": zoom_select_btn.cget("hover_color"),
        "text_color": zoom_select_btn.cget("text_color"),
        "font": zoom_select_btn.cget("font")
    }
    zoom_btn_active_style = {      # define active style
        "fg_color": "#1F6AA5",
        "hover_color": "#3B8ED0",
        "text_color": "white",
        "font": ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
    }
    widgets['zoom_btn_default_style'] = zoom_btn_default_style
    widgets['zoom_btn_active_style'] = zoom_btn_active_style

    # Reset zoom button
    reset_btn = ctk.CTkButton(
        zoom_frame,
        text="Reset Zoom",
        command=app.reset_zoom
    )
    reset_btn.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    widgets['reset_btn'] = reset_btn

    #%% Segmentation source (second block in sidebar)
    lbl_source_frame = ctk.CTkFrame(sidebar)
    lbl_source_frame.pack(fill="x", padx=5, pady=5)
    widgets['lbl_source_frame'] = lbl_source_frame

    ctk.CTkLabel(lbl_source_frame, text="Seg Source").grid(
        row=0, column=0, sticky="nsew", pady=5
    )

    # Might want to move this out into another file for easy editing
    app_state.scene.lbl_sources = [
        "Unet+ITT_pixel",
        # "Unet+ITT_pixel+MV",
        # "Unet+ITT_region",
        # "Results_Major"
    ]
    filenames_ = [
        "colored_predict_cnn.png",
        # "CNN_colored_m_v_per_CC.png",
        # "colored_predict_transformer.png",
        # "resnet.png"
    ]
    app_state.scene.filenames = ["/{}/{}".format(lbl_s, file)
                    for lbl_s, file in zip(app_state.scene.lbl_sources, filenames_)]
    lbl_source_btn = {}
    mode_var_lbl_source = None
    mode_var_lbl_source_prev = None
    widgets['lbl_source_btn'] = lbl_source_btn
    widgets['mode_var_lbl_source'] = mode_var_lbl_source
    widgets['mode_var_lbl_source_prev'] = mode_var_lbl_source_prev

    # Radio buttons for explicit selection
    for i, lbl_s in enumerate(app_state.scene.lbl_sources):
        widgets = update_label_source_widgets(app, widgets, lbl_s, i)

    #%% Operations (third block in sidebar)
    operation_frame = ctk.CTkFrame(sidebar)
    operation_frame.pack(fill="x", padx=5, pady=5)
    widgets['operation_frame'] = operation_frame

    # # # Evaluation panel
    evaluation_window = ctk.CTkToplevel(app)
    evaluation_window.transient(app)  # Set parent window
    evaluation_window.attributes("-topmost", True)  # Always on top
    evaluation_window.title("Evaluation Panel")
    evaluation_window.withdraw()  # Hide the window at start
    evaluation_window.protocol(
        "WM_DELETE_WINDOW",
        app.close_evaluation_panel
    )  # Hide window instead of destroying it on close

    evaluation_panel = EvaluationPanel(evaluation_window, app)
    evaluation_panel.pack(fill="both", expand=True, padx=10, pady=10)

    ctk.CTkButton(
        operation_frame,
        text="Evaluation",
        command=app.show_evaluation_panel
    ).grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # # # Annotation panel
    annotation_window = ctk.CTkToplevel(app)
    annotation_window.transient(app)  # Set parent window
    annotation_window.attributes("-topmost", True)  # Always on top
    annotation_window.title("Annotation Panel")
    annotation_window.withdraw()  # Hide the window at start
    annotation_window.protocol(
        "WM_DELETE_WINDOW",
        app.close_annotation_panel
    )  # Hide window instead of destroying it on close

    annotation_panel = AnnotationPanel(annotation_window, app)
    annotation_panel.pack(fill="both", expand=True, padx=10, pady=10)

    ctk.CTkButton(
        operation_frame,
        text="Annotation",
        command=app.show_annotation_panel
    ).grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    # Loading bar at the bottom of the sidebar
    loading_bar_frame = ctk.CTkFrame(sidebar)
    loading_bar_frame.pack(fill="x", padx=5, pady=5, side="bottom")

    # Loading bar label
    loading_bar_label = ctk.CTkLabel(loading_bar_frame, text="", font=ctk.CTkFont(size=12))
    loading_bar = ctk.CTkProgressBar(loading_bar_frame, orientation="horizontal", mode="determinate")
    loading_bar.set(0)

    # Set up in grid for hiding and showing
    loading_bar_label.grid(row=0, column=0, padx=5, pady=5, sticky="we")
    loading_bar.grid(row=1, column=0, padx=5, pady=5, sticky="we")

    loading_bar_label.grid_remove() # Hide loading bar after short delay
    loading_bar.grid_remove() # Hide loading bar after short delay

    # Minimap frame housing minimap and status bar
    minimap_frame = ctk.CTkFrame(canvas, width=200, height=200, corner_radius=12)
    
    # Coords on top of minimap
    status_bar = ctk.CTkLabel(minimap_frame, width=200,text="-, -", bg_color="black", text_color="white", font=ctk.CTkFont(size=12))
    status_bar.pack(fill="both", expand=True, anchor = "center")

    # Minimap in bottom-right corner of canvas
    minimap = Minimap(minimap_frame, w=200, h=200)
    minimap.pack(fill="both", expand=True)
    minimap_window_id = canvas.create_window(0, 0, window=minimap_frame, anchor="se", tags=("minimap"))
    canvas.bind("<Configure>", app._update_minimap_position)
    
    # Add toggle for showing previous annotations on minimap right below annotation button
    show_prev_anno_switch = ctk.CTkSwitch(
        operation_frame,
        text="Show Annotations on Minimap",
        command=app.toggle_show_anno_on_minimap
    )
    show_prev_anno_switch.select()  # Default to showing previous annotations
    show_prev_anno_switch.grid(row=2, column=0, padx=5, pady=5, sticky="w")
    widgets['show_prev_anno_switch'] = show_prev_anno_switch

    # Layout behavior inside bottom_container
    sidebar.grid_rowconfigure(0, weight=0)
    sidebar.grid_rowconfigure(1, weight=0)
    sidebar.grid_rowconfigure(2, weight=1)
    sidebar.grid_rowconfigure(3, weight=1)
    sidebar.grid_columnconfigure(0, weight=1)

    return VisualizerLayout(
        canvas=canvas,
        sidebar=sidebar,
        minimap=minimap,
        minimap_window_id=minimap_window_id,
        status_bar=status_bar,
        annotation_panel=annotation_panel,
        evaluation_panel=evaluation_panel,
        annotation_window=annotation_window,
        evaluation_window=evaluation_window,
        loading_bar=loading_bar,
        loading_bar_label=loading_bar_label,
        widgets=widgets,
    )

def update_label_source_widgets(app, widgets, lbl_source, i):
    """
    Update the label source selection widgets (button) when new label sources are loaded.
    """
    # Radio buttons for explicit selection
    if widgets['mode_var_lbl_source'] is None:
        widgets['mode_var_lbl_source'] = ctk.StringVar(value=lbl_source)  # Default selection
        widgets['mode_var_lbl_source_prev'] = widgets['mode_var_lbl_source'].get()
    widgets['lbl_source_btn'][lbl_source] = ctk.CTkRadioButton(widgets['lbl_source_frame'], 
                                                                            text=lbl_source, 
                                                                            variable=widgets['mode_var_lbl_source'],
                                                            value=lbl_source, 
                                                            command=app.choose_lbl_source)
    widgets['lbl_source_btn'][lbl_source].grid(row=i+1, column=0, sticky="w", pady=(10, 10))

    return widgets