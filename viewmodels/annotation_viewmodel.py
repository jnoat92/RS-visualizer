import datetime
import json
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2gray

from core.utils import generate_boundaries


class AnnotationViewModel:
    """Owns annotation data changes while the controller owns canvas drawing."""

    CUSTOM_SOURCE = "Custom_Annotation"

    def __init__(self, app_state):
        self.app_state = app_state

    def reset_selection(self):
        anno = self.app_state.anno
        anno.polygon_points_img_coor = []
        anno.selected_polygon_window = None
        anno.selected_polygon_area_idx = None
        anno.multiple_polygons = False

    def ensure_custom_annotation_source(self):
        scene = self.app_state.scene
        key = self.CUSTOM_SOURCE
        created = key not in scene.predictions
        scene.predictions[key] = scene.predictions[scene.active_source].copy()
        scene.land_nan_masks[key] = scene.land_nan_masks[scene.active_source].copy()
        scene.boundmasks[key] = scene.boundmasks[scene.active_source].copy()
        scene.active_source = key

        if key not in scene.lbl_sources:
            scene.lbl_sources.append(key)
            scene.filenames.append(
                "{}/{}/{}".format(key, scene.scene_name, "custom_annotation.png")
            )
            created = True
        return created

    def selected_area_is_land(self):
        scene = self.app_state.scene
        anno = self.app_state.anno
        return scene.land_nan_masks[scene.active_source][
            anno.selected_polygon_area_idx
        ].all()

    def selected_area_matches_color(self, class_color):
        scene = self.app_state.scene
        anno = self.app_state.anno
        return (
            scene.predictions[scene.active_source][anno.selected_polygon_area_idx]
            == class_color
        ).all()

    def apply_class_to_selection(self, class_color):
        scene = self.app_state.scene
        anno = self.app_state.anno
        key = self.CUSTOM_SOURCE

        self.ensure_custom_annotation_source()

        if anno.undo_stack and len(anno.undo_stack) > anno.stack_limit:
            anno.undo_stack.pop(0)
        anno.undo_stack.append(
            (
                anno.selected_polygon_area_idx,
                scene.predictions[scene.active_source][
                    anno.selected_polygon_area_idx
                ].copy(),
                anno.selected_polygon_window,
            )
        )
        anno.redo_stack.clear()

        scene.predictions[key][anno.selected_polygon_area_idx] = class_color
        scene.predictions[key][scene.land_nan_masks[key]] = [255, 255, 255]
        self.update_boundaries_for_window(anno.selected_polygon_window)
        return key

    def update_boundaries_for_window(self, window):
        scene = self.app_state.scene
        img_y_min, img_y_max, img_x_min, img_x_max = window
        img_y_min = max(0, img_y_min - 20)
        img_y_max = min(scene.predictions[scene.active_source].shape[0], img_y_max + 20)
        img_x_min = max(0, img_x_min - 20)
        img_x_max = min(scene.predictions[scene.active_source].shape[1], img_x_max + 20)

        scene.boundmasks[scene.active_source][
            img_y_min:img_y_max,
            img_x_min:img_x_max,
        ] = generate_boundaries(
            rgb2gray(
                scene.predictions[scene.active_source][
                    img_y_min:img_y_max,
                    img_x_min:img_x_max,
                ]
            )
        )

    def changed_area_mask(self):
        scene = self.app_state.scene
        key = self.CUSTOM_SOURCE
        if key not in scene.predictions or not scene.lbl_sources:
            return None
        return scene.predictions[key][:, :, 0] != scene.predictions[scene.lbl_sources[0]][
            :,
            :,
            0,
        ]

    def undo_redo_annotation(self, last_polygon_area_idx, last_colours, last_window):
        scene = self.app_state.scene
        scene.predictions[scene.active_source][last_polygon_area_idx] = last_colours
        self.update_boundaries_for_window(last_window)
        self.reset_selection()

    def set_selected_segment(self, contours, mask):
        anno = self.app_state.anno
        anno.selected_polygon_area_idx = [(y, x) for y, x in zip(*np.where(mask))]
        points = np.asarray(anno.selected_polygon_area_idx)
        img_y_min = points[:, 0].min()
        img_y_max = points[:, 0].max()
        img_x_min = points[:, 1].min()
        img_x_max = points[:, 1].max()
        anno.selected_polygon_window = (img_y_min, img_y_max, img_x_min, img_x_max)
        anno.selected_polygon_area_idx = tuple(zip(*anno.selected_polygon_area_idx))
        anno.polygon_points_img_coor = [[(x, y) for y, x in c] for c in contours]
        anno.multiple_polygons = True

    def save_annotation(self, notes):
        scene = self.app_state.scene
        anno = self.app_state.anno
        key = self.CUSTOM_SOURCE

        if key not in scene.predictions:
            raise ValueError(f"There is no {key} to save.")

        file_path = scene.filenames[list(scene.predictions).index(key)]
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)

        img = scene.predictions[key].copy()
        img[(img == [0, 255, 255]).all(axis=2)] = [0, 0, 128]
        img[(img == [255, 130, 0]).all(axis=2)] = [128, 0, 0]
        Image.fromarray(img).save(file_path)

        new_note = {
            scene.scene_name: {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notes": notes,
            }
        }

        notes_file_folder = os.path.split(os.path.split(file_path)[0])[0]
        notes_file_path = os.path.join(notes_file_folder, "annotation_notes.json")

        if os.path.exists(notes_file_path):
            with open(notes_file_path, "r") as f:
                try:
                    existing_notes = json.load(f)
                except json.JSONDecodeError:
                    existing_notes = {}
        else:
            existing_notes = {}

        existing_notes[scene.scene_name] = new_note[scene.scene_name]

        with open(notes_file_path, "w") as f:
            json.dump(existing_notes, f, indent=4)

        anno.annotation_notes = notes

        changed_area_mask = self.changed_area_mask()
        annotated_area_file_folder = os.path.split(file_path)[0]
        annotated_area_file_path = os.path.join(
            annotated_area_file_folder,
            "changed_area.png",
        )
        Image.fromarray(changed_area_mask).save(annotated_area_file_path)

        return file_path
