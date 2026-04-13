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


class SceneController:
    def __init__(self, deps):
        self.deps = deps

    def choose_SAR_scene(self):
        """
        Open a file dialog to select a SAR scene directory, load the images that scene, generate the predictions, and update the display.
        """

        scene = self.deps.app_state.scene
        display = self.deps.app_state.display
        anno = self.deps.app_state.anno

        self.deps.app.close_evaluation_panel()
        self.deps.app.close_annotation_panel()
        
        prev_folder_path = scene.folder_path

        root = ctk.CTk()
        root.withdraw()
        scene.folder_path = filedialog.askdirectory(initialdir=os.path.dirname(prev_folder_path) if scene.folder_path else os.getcwd(),
                                                   title='Select the dated directory containing HH/HV images')
        root.destroy()

        if scene.folder_path:

            if scene.folder_path == prev_folder_path:
                return

            scene.scene_name = scene.folder_path.split('/')[-1]

            self.deps.app.title(f"Scene {scene.scene_name}-{display.channel_mode}")

            # Show loading bar
            self.deps.loading_bar_label.grid(row=0, column=0)
            self.deps.loading_bar.grid(row=1, column=0)
            self.deps.app.update_idletasks()

            self.deps.loading_bar.set(0) # Update loading bar after loading images
            self.deps.loading_bar_label.configure(text="Loading images...")
            self.deps.app.update_idletasks() # Force UI update to show loading bar progress

            try:
                rcm_data = load_rcm_product(scene.folder_path)
            except (FileNotFoundError, ValueError) as e:
                messagebox.showinfo("Error", f"The selected directory does not contain the required files. Please, select a valid directory.\n\n{e}", parent=self.deps.app.master)
                scene.folder_path = ''
                self.deps.loading_bar.set(0) # Update loading bar after loading images
                self.deps.loading_bar_label.configure(text="Error loading images")
                self.deps.app.update_idletasks()
                return

            self.deps.loading_bar.set(0.2) # Update loading bar after loading images
            self.deps.loading_bar_label.configure(text="Scaling images...")
            self.deps.app.update_idletasks()      

            # Scale image
            rcm_200m_data, rcm_scaled_data = scale_hh_hv(rcm_data, target_spacing=scene.target_spacing)

            self.deps.loading_bar.set(0.35) # Update loading bar after loading images
            self.deps.loading_bar_label.configure(text="Building land mask...")
            self.deps.app.update_idletasks()

            # Build land masks
            land_mask = build_land_masks(rcm_scaled_data)


            self.deps.loading_bar.set(0.5) # Update loading bar after loading images
            self.deps.loading_bar_label.configure(text="Normalizing data...")
            self.deps.app.update_idletasks()  

            # Normalize and prepare images
            raw_img, orig_img, hist, n_valid, nan_mask, geo_coord_helpers = normalize_and_prepare_images(rcm_scaled_data, scene.normalization_method)

            # Save raw images to app state for later use (e.g., layering)
            scene.raw_img = raw_img
            scene.orig_img = orig_img
            scene.hist = hist
            scene.n_valid = n_valid
            scene.nan_mask = nan_mask
            scene.base_land_mask = land_mask
            scene.rcm_200m_data = rcm_200m_data
            scene.rcm_scaled_data = rcm_scaled_data

            # Save geo coord helpers to app state for later use
            scene.geo_coord_helpers = geo_coord_helpers
            scene.tie_lines = rcm_data.get("tie_lines", None)
            scene.tie_pixels = rcm_data.get("tie_pixels", None)
            scene.tie_lats = rcm_data.get("tie_lats", None)
            scene.tie_lons = rcm_data.get("tie_lons", None)

            # Build tiepoint grid interpolator if available
            if scene.tie_lines is not None:
                rows, cols, lat_grid, lon_grid = tiepoints_1d_to_grid(scene.tie_lines, scene.tie_pixels, scene.tie_lats, scene.tie_lons)
                scene.pix2ll = make_pix2ll(rows, cols, lat_grid, lon_grid)

            scene.color_composites = orig_img
            
            scene.color_composites["(HH, HH, HV)"] = layer_imagery(
                orig_img["HH"],
                orig_img["HV"],
                stack="(HH, HH, HV)"
            )
            scene.color_composites["(HH, HV, HV)"] = layer_imagery(
                orig_img["HH"],
                orig_img["HV"],
                stack="(HH, HV, HV)"
            )
            
            # Handle switching scenes with existing custom annotation to one without
            if "Custom_Annotation" in scene.lbl_sources:
                scene.filenames.pop()
                scene.lbl_sources.pop()

            # Should be from display_controller later
            self.deps.app.choose_image()

            self.deps.loading_bar.set(0.6)
            self.deps.loading_bar_label.configure(text="Generating prediction...")
            self.deps.app.update_idletasks()

            self.load_pred()

            if not self.deps.app.choose_lbl_source(plot=False):
                scene.folder_path = ''
                return
            self.deps.app.update_idletasks()
            self.deps.app.after(100, self.deps.app.reset_zoom)    # Delay the initial reset call with .after() so the canvas has its final size:
            
            self.deps.app._set_all_children_enabled(self.deps.sidebar, True)
            if display.channel_mode in ["(HH, HH, HV)", "(HH, HV, HV)"]:
                self.deps.widgets["hh_hv_switch"].configure(state=ctk.DISABLED)

            # Reset annotation stacks
            anno.undo_stack.clear()
            anno.redo_stack.clear()

            self.deps.widgets["contrast_slider"].set(0) # reset to default
            self.deps.app_state.display.contrast = 0.0
            self.deps.widgets["brightness_slider"].set(0) # reset to default
            self.deps.app_state.display.brightness = 0.0

            self.deps.loading_bar.set(1)
            self.deps.loading_bar_label.configure(text="Inference complete")
            self.deps.app.update_idletasks()

            self.deps.app.after(3000, self.deps.loading_bar_label.grid_remove) # Hide loading bar after short delay
            self.deps.app.after(3000, self.deps.loading_bar.grid_remove) # Hide loading bar after short delay

        else:
            scene.folder_path = prev_folder_path

    def load_pred(self):
        """
        Load prediction files for the current scene, check if custom annotation already exists and load it
        Also updates the label source selection widgets based on available label sources.
        """

        scene = self.deps.app_state.scene
        anno = self.deps.app_state.anno

        scene.predictions = {}
        scene.land_nan_masks = {}
        scene.boundmasks = {}

        if scene.model_path is None or not os.path.isfile(scene.model_path):
            model_paths = []
            model_folder = resource_path("model")

            for file in os.listdir(model_folder):
                if file.endswith(".pt"):
                    model_paths.append(file)

            model_path = model_paths[0] if model_paths else None
            model_path = os.path.join(model_folder, model_path) if model_path else None
        else:
            model_path = scene.model_path
        
        variables = run_pred_model(scene.lbl_sources[0], scene.rcm_200m_data, scene.base_land_mask, 
                                                                  model_path=model_path, 
                                                                  target_width = scene.rcm_scaled_data["dst_width"],
                                                                  target_height = scene.rcm_scaled_data["dst_height"],
                                                                  target_spacing=scene.target_spacing, device='cpu')
        
        existing_anno, anno.annotation_notes = load_existing_annotation(scene.scene_name)

        if existing_anno is not None:
            variables.append(existing_anno)
            scene.lbl_sources.append("Custom_Annotation")
            scene.filenames.append("{}/{}/{}".format(scene.lbl_sources[-1], scene.scene_name, "custom_annotation.png"))
        self.deps.annotation_panel.clear_notes()
        
        # Reset label source radio buttons
        for key in self.deps.widgets['lbl_source_btn'].keys():
            self.deps.widgets['lbl_source_btn'][key].destroy()
        self.deps.widgets['lbl_source_btn'] = {}
        self.deps.widgets['mode_var_lbl_source'] = None
        self.deps.widgets['mode_var_lbl_source_prev'] = None

        # Add available label sources
        for i, (key, pred, land_nan_mask, boundmask) in enumerate(variables):
            if pred is None: 
                if key != 'Custom_Annotation':
                    messagebox.showinfo("Error", f"The selected scene does not contain prediction files for {key}.", parent=self.deps.app.master)
                continue
            self.deps.app.update_label_source_widgets(key, i)
            scene.predictions[key] = pred
            scene.land_nan_masks[key] = land_nan_mask
            scene.boundmasks[key] = boundmask

        custom_anno = "Custom_Annotation"
        if custom_anno in scene.lbl_sources:
            result = messagebox.askyesno("Custom Annotation Found",
                                         "An existing custom annotation was found for this scene. Do you want to view it?",
                                         parent=self.deps.app.master)
            if result:
                scene.active_source = custom_anno
                self.deps.widgets['mode_var_lbl_source'].set(custom_anno)

            self.deps.app.choose_image() # Refresh image to show annotation on minimap

    def set_resolution_level(self, level):
        """
        Set the resolution level for display. Save in app state.
        """
        resolution = int(level.split("m")[0])
        self.deps.app_state.scene.target_spacing = resolution

    def set_normalization_method(self, method):
        """
        Set the normalization method for display. Save in app state.
        """
        self.deps.app_state.scene.normalization_method = method.lower()

    def set_model_file(self, model_path):
        """
        Set the model file for prediction. Save in app state.
        """
        self.deps.app_state.scene.model_path = model_path
        