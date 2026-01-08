'''
Data classes for application state management

Last modified: Jan 2026
'''

import numpy as np
from dataclasses import dataclass, field

# Data class that houses values that determine the orientation of the image
@dataclass(slots=True)
class ViewState:
    zoom_factor: float = 1.0
    min_zoom: float = 0.1
    max_zoom: float = 10.0
    offset_x: float = 0.0
    offset_y: float = 0.0

# Data class that houses the currently loaded scene and everything tied to that data
@dataclass(slots=True)
class SceneState:
    folder_path: str = ""
    filenames: list[str] = field(default_factory=list)
    current_index: int = 0

    # Current image data
    img: np.ndarray = None
    img_name: str = ""

    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    landmasks: dict[str, np.ndarray] = field(default_factory=dict)
    boundmasks: dict[str, np.ndarray] = field(default_factory=dict)

    # Store currently selected prediction
    active_source: str = ""

    # Caching
    last_render_key: tuple = None
    last_render_img: np.ndarray = None

# Data class that houses values that affect the base image pixels before overlay
@dataclass(slots=True)
class DisplaySettings:
    brightness: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    clip: bool = True

# Data class for values that impact the overlay
@dataclass(slots=True)
class OverlaySettings:
    opacity: float = 0.0

# Data class for annotation functions
@dataclass(slots=True)
class AnnotationState:
    annotation_mode: str = ""
    polygon_points_img_coor: list = field(default_factory=list)
    selected_polygon_window: tuple = None
    selected_polygon_area_idx: tuple = None

@dataclass(slots=True)
class AppState:
    view: ViewState = field(default_factory=ViewState)
    scene: SceneState = field(default_factory=SceneState)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    anno: AnnotationState = field(default_factory=AnnotationState)


