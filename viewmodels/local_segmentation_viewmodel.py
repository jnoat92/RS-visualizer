import numpy as np

from core.segmentation import IRGS, remove_edge_touching_polygons


class LocalSegmentationViewModel:
    """Runs local segmentation and stores the resulting masks in app state."""

    def __init__(self, app_state):
        self.app_state = app_state

    def run(self, x_min, y_min, x_max, y_max, progress=None):
        overlay = self.app_state.overlay
        scene = self.app_state.scene

        overlay.local_segmentation_area = np.stack(
            [
                scene.raw_img[overlay.local_segmentation_source],
                scene.raw_img[overlay.local_segmentation_source],
            ],
            axis=-1,
        )[y_min:y_max, x_min:x_max]

        overlay.local_segmentation_limits = (x_min, y_min, x_max, y_max)
        land_nan_mask_crop = scene.land_nan_masks[scene.active_source][
            y_min:y_max,
            x_min:x_max,
        ]
        overlay.select_local_segmentation = False

        if progress:
            progress(0.0, "Running local segmentation...")
        irgs_output, boundaries = IRGS(
            overlay.local_segmentation_area,
            n_classes=overlay.local_seg_n_classes,
            n_iter=120,
            mask=~land_nan_mask_crop,
        )

        if progress:
            progress(0.4, "Clearing border polygons...")
        irgs_output, boundaries = remove_edge_touching_polygons(irgs_output)

        if progress:
            progress(0.7, "Applying segmentation on overlay...")
        overlay.local_segmentation_mask = np.zeros_like(
            scene.boundmasks[scene.active_source],
            dtype=np.uint8,
        )
        overlay.local_segmentation_mask[y_min:y_max, x_min:x_max] = irgs_output
        overlay.local_segmentation_mask = np.tile(
            overlay.local_segmentation_mask[:, :, np.newaxis],
            (1, 1, 3),
        )

        overlay.local_segmentation_bounds = np.zeros_like(
            scene.boundmasks[scene.active_source],
            dtype=bool,
        )
        boundaries_bool = boundaries != 1
        overlay.local_segmentation_bounds[y_min:y_max, x_min:x_max] = boundaries_bool
        overlay.show_local_segmentation = True

    def clear(self):
        overlay = self.app_state.overlay
        if not overlay.show_local_segmentation:
            return False
        overlay.local_segmentation_area = None
        overlay.local_segmentation_mask = None
        overlay.local_segmentation_bounds = None
        overlay.show_local_segmentation = False
        return True

    def set_source(self, source):
        self.app_state.overlay.local_segmentation_source = source

    def set_n_classes(self, value):
        self.app_state.overlay.local_seg_n_classes = int(value)

