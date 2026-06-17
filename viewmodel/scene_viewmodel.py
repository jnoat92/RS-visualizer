'''
SceneViewModel handles loading SAR scenes, land masks, prepared display imagery, 
and predictions for the application.

Last modified: Jun 2026
'''

import os

from model.io import (
    load_existing_annotation,
    load_rcm_product,
    run_pred_model,
    scale_hh_hv,
    build_land_masks,
    normalize_and_prepare_images,
    resource_path,
)
from model.render import layer_imagery
from model.utils import tiepoints_1d_to_grid, make_pix2ll


class SceneViewModel:
    """Loads SAR scenes, land masks, prepared display imagery, and predictions."""

    def __init__(self, app_state):
        self.app_state = app_state

    def load_scene(self, folder_path, progress=None):
        scene = self.app_state.scene

        if progress:
            progress(0.0, "Loading images...")

        rcm_data = load_rcm_product(folder_path)
        if rcm_data is None:
            raise ValueError("Unable to load the selected RCM product.")

        scene.folder_path = folder_path
        scene.scene_name = os.path.basename(os.path.normpath(folder_path))
        scene.sar_img = rcm_data["sar_img"]
        rcm_data["sar_img"] = None  # Free memory by removing the dataset from rcm_data dict
        
        if progress:
            progress(0.2, "Scaling images...")
        rcm_200m_data = scale_hh_hv(
            rcm_data,
            target_spacing_m=scene.target_spacing,
        )

        if progress:
            progress(0.35, "Building land mask...")
        land_mask = build_land_masks(rcm_200m_data)

        if progress:
            progress(0.5, "Normalizing data...")
        raw_img, orig_img, hist, n_valid, nan_mask, geo_coord_helpers = (
            normalize_and_prepare_images(rcm_200m_data, scene.normalization_method)
        )

        scene.raw_img = raw_img
        scene.orig_img = orig_img
        scene.hist = hist
        scene.n_valid = n_valid
        scene.nan_mask = nan_mask
        scene.base_land_mask = land_mask
        scene.rcm_200m_data = rcm_200m_data
        scene.rcm_scaled_data = None
        scene.scale_factor = round(scene.sar_img.width / rcm_200m_data["dst_width"], 2)  # Calculate scale factor based on original and target widths
        
        scene.geo_coord_helpers = geo_coord_helpers
        scene.tie_lines = rcm_data.get("tie_lines", None)
        scene.tie_pixels = rcm_data.get("tie_pixels", None)
        scene.tie_lats = rcm_data.get("tie_lats", None)
        scene.tie_lons = rcm_data.get("tie_lons", None)

        if scene.tie_lines is not None:
            rows, cols, lat_grid, lon_grid = tiepoints_1d_to_grid(
                scene.tie_lines,
                scene.tie_pixels,
                scene.tie_lats,
                scene.tie_lons,
            )
            scene.pix2ll = make_pix2ll(rows, cols, lat_grid, lon_grid)

        scene.color_composites = orig_img
        scene.color_composites["(HH, HH, HV)"] = layer_imagery(
            orig_img["HH"],
            orig_img["HV"],
            stack="(HH, HH, HV)",
        )
        scene.color_composites["(HH, HV, HV)"] = layer_imagery(
            orig_img["HH"],
            orig_img["HV"],
            stack="(HH, HV, HV)",
        )

        if "Custom_Annotation" in scene.lbl_sources:
            scene.filenames.pop()
            scene.lbl_sources.pop()

    def load_predictions(self, progress=None):
        scene = self.app_state.scene
        anno = self.app_state.anno

        if progress:
            progress(0.6, "Generating prediction...")

        scene.predictions = {}
        scene.land_nan_masks = {}
        scene.boundmasks = {}

        if scene.model_path is None or not os.path.isfile(scene.model_path):
            model_paths = []
            model_folder = resource_path("model\prediction_model")

            for file in os.listdir(model_folder):
                if file.endswith((".h5", ".pt", ".pth")):
                    model_paths.append(file)
            model_path = model_paths[0] if model_paths else None
            model_path = os.path.join(model_folder, model_path) if model_path else None
        else:
            model_path = scene.model_path

        variables = run_pred_model(
            scene.lbl_sources[0],
            scene.rcm_200m_data,
            scene.base_land_mask,
            model_path=model_path,
            target_width=scene.rcm_200m_data["dst_width"],
            target_height=scene.rcm_200m_data["dst_height"],
            target_spacing=scene.target_spacing,
            device="cpu",
        )

        existing_anno, anno.annotation_notes = load_existing_annotation(scene.scene_name)
        has_existing_annotation = existing_anno is not None

        if has_existing_annotation:
            variables.append(existing_anno)
            scene.lbl_sources.append("Custom_Annotation")
            scene.filenames.append(
                "{}/{}/{}".format(
                    scene.lbl_sources[-1],
                    scene.scene_name,
                    "custom_annotation.png",
                )
            )

        loaded_sources = []
        missing_sources = []
        for key, pred, land_nan_mask, boundmask in variables:
            if pred is None:
                missing_sources.append(key)
                continue
            scene.predictions[key] = pred
            scene.land_nan_masks[key] = land_nan_mask
            scene.boundmasks[key] = boundmask
            loaded_sources.append(key)

        return loaded_sources, missing_sources, has_existing_annotation

