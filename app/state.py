'''
Data classes for application state management

Last modified: Jan 2026
'''

import numpy as np
from dataclasses import dataclass, field

# Data class that houses values that when changed requires a refresh_view()
@dataclass(slots=True)
class ViewState:
    zoom_factor: float = 1.0
    min_zoom: float = 0.1
    max_zoom: float = 20.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    # Keep track of pan interaction
    is_panning: bool = False
    pan_start_screen: tuple[int, int] = None

    # Selection zoom
    zoom_select_mode: bool = False


# Data class that houses the currently loaded scene and everything tied to that data
@dataclass(slots=True)
class SceneState:
    scene_name: str = ""
    folder_path: str = ""
    filenames: list[str] = field(default_factory=list)
    current_index: int = 0

    # Current image data
    img: np.ndarray = None
    img_name: str = ""

    # Store original images (HH, HV)
    raw_img: dict[str, np.ndarray] = field(default_factory=dict)
    orig_img: dict[str, np.ndarray] = field(default_factory=dict)

    # Store histogram data for contrast enhancement
    contrast_img: dict[str, np.ndarray] = field(default_factory=dict)
    nan_mask: dict[str, np.ndarray] = field(default_factory=dict)
    cum_hist: dict[str, list[np.ndarray]] = field(default_factory=dict)
    bin_list: dict[str, list[np.ndarray]] = field(default_factory=dict)
    bands: dict[str, int] = field(default_factory=dict)

    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    landmasks: dict[str, np.ndarray] = field(default_factory=dict)
    boundmasks: dict[str, np.ndarray] = field(default_factory=dict)

    # Store currently selected prediction
    active_source: str = ""

    # Store sources available
    lbl_sources: list[str] = field(default_factory=list)

    # Caching
    last_render_key: tuple = None
    last_render_img: np.ndarray = None

# Data class that houses values that affect the base image pixels before overlay
@dataclass(slots=True)
class DisplaySettings:
    brightness: float = 0.0
    contrast: float = 1.0 # Placeholder for later
    gamma: float = 1.0 # Placeholder for later
    clip: bool = True

    better_contrast: bool = False # Alt contrast method for now
    channel_mode: str = "(HH/HV)" # Store current channel (image type)

# Data class for values that impact the overlay
@dataclass(slots=True)
class OverlaySettings:
    alpha: float = 0.5 # Opacity of overlay
    show_overlay: bool = True # Segmentation overlay visibility
    show_boundaries: bool = True
    show_landmask: bool = False
    use_gpu: bool = False # CPU quicker based on tests to update overlay

# Data class for annotation functions
@dataclass(slots=True)
class AnnotationState:
    annotation_mode: str = None
    active_label: str = "water"

    # Drawing points
    polygon_points_img_coor: list[tuple[int, int]] = field(default_factory=list)

    # Select created polygon
    selected_polygon_window: tuple[int, int, int, int] = None
    selected_polygon_area_idx: int = None
    multiple_polygons: bool = False

    unsaved_changes: bool = False

    # Annotation text storage
    annotation_notes: str = ""

    # Zoom window state
    zoom_window_open: bool = False
    zoom_bbox_img: tuple[int,int,int,int] = None

@dataclass(slots=True)
class AppState:
    view: ViewState = field(default_factory=ViewState)
    scene: SceneState = field(default_factory=SceneState)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    anno: AnnotationState = field(default_factory=AnnotationState)


