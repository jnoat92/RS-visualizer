'''
SceneController manages loading and switching between different scenes 
in the application. It serves as the central point for coordinating 
scene-related actions and ensuring a smooth user experience when
navigating through different images and their associated data.

Last modified: Mar 2026
'''

import os
import customtkinter as ctk
from tkinter import filedialog, messagebox

from core.io import (
    load_existing_annotation,
    load_rcm_product,
    run_pred_model,
    scale_hh_hv,
    build_land_masks,
    normalize_and_prepare_images,
    resource_path,
)
from core.render import layer_imagery
from core.utils import tiepoints_1d_to_grid, make_pix2ll

# Move choose_SAR_scene, load_pred for sure
# Potentially also move update_label_source_widgets and choose_lbl_source

class SceneController:
    def __init__(self, visualizer):
        self.visualizer = visualizer