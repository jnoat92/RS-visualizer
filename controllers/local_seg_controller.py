'''
LocalSegController handles the local segmentation mode, allowing users 
to select a region of interest for segmentation and annotation. It 
manages the state of local segmentation selection and interacts with the 
CanvasEventsController to enable zoom selection when in local 
segmentation mode. It interacts with AnnotationController to allow 
annotation within the selected local segmentation area. It also updates 
the display to show the local segmentation overlay when enabled.

Last modified: Apr 2026
'''

import numpy as np
from core.segmentation import IRGS, remove_edge_touching_polygons

class LocalSegController:
    def __init__(self, deps, display_controller, annotation_controller, 
                 canvas_events_controller, zoom_controller):
        self.deps = deps
        self.display_controller = display_controller
        self.annotation_controller = annotation_controller
        self.canvas_events_controller = canvas_events_controller
        self.zoom_controller = zoom_controller

    def run_local_segmentation(self, x_min, y_min, x_max, y_max):
        """
        Run local segmentation (IRGS) on the area selected by the user, 
        update the overlay with the local segmentation results, and refresh the display.
        """
        overlay = self.deps.app_state.overlay
        scene = self.deps.app_state.scene

        overlay.local_segmentation_area = np.stack([scene.raw_img[overlay.local_segmentation_source], 
                                                    scene.raw_img[overlay.local_segmentation_source]], axis=-1)[y_min:y_max, x_min:x_max]

        overlay.local_segmentation_limits = (x_min, y_min, x_max, y_max)
        land_nan_mask_crop = scene.land_nan_masks[scene.active_source][y_min:y_max, x_min:x_max]
        # Disable select local segmentation mode after selection
        overlay.select_local_segmentation = False

        self.deps.canvas.delete(self.canvas_events_controller.selection_rect_id)
        self.canvas_events_controller.selection_rect_id = None
        self.canvas_events_controller.selection_start_coord = None

        # Show loading bar
        self.deps.loading_bar_label.grid(row=0, column=0)
        self.deps.loading_bar.grid(row=1, column=0)
        self.deps.app.update_idletasks()

        self.deps.loading_bar.set(0)
        self.deps.loading_bar_label.configure(text="Running local segmentation...")
        self.deps.app.update_idletasks()

        # Run IRGS on the selected area
        irgs_output, boundaries = IRGS(overlay.local_segmentation_area, n_classes=overlay.local_seg_n_classes, n_iter=120, mask=~land_nan_mask_crop)

        self.deps.loading_bar.set(0.4)
        self.deps.loading_bar_label.configure(text="Clearing border polygons...")
        self.deps.app.update_idletasks()

        irgs_output, boundaries = remove_edge_touching_polygons(irgs_output)

        self.deps.loading_bar.set(0.7)
        self.deps.loading_bar_label.configure(text="Applying segmentation on overlay...")
        self.deps.app.update_idletasks()

        overlay.local_segmentation_mask = np.zeros_like(scene.boundmasks[scene.active_source], dtype=np.uint8)
        overlay.local_segmentation_mask[y_min:y_max, x_min:x_max] = irgs_output
        overlay.local_segmentation_mask = np.tile(overlay.local_segmentation_mask[:, :, np.newaxis], (1, 1, 3))

        overlay.local_segmentation_bounds = np.zeros_like(scene.boundmasks[scene.active_source], dtype=bool)
        boundaries_bool = boundaries != 1
        overlay.local_segmentation_bounds[y_min:y_max, x_min:x_max] = boundaries_bool
        overlay.show_local_segmentation = True

        self.annotation_controller.reset_annotation() # Reset annotation to prevent annotation on old local segmentation
        self.display_controller.refresh_view()

        self.deps.loading_bar.set(1)
        self.deps.loading_bar_label.configure(text="Local segmentation applied")
        self.deps.app.update_idletasks()

        self.deps.app.after(3000, self.deps.loading_bar_label.grid_remove)
        self.deps.app.after(3000, self.deps.loading_bar.grid_remove)
        self.deps.app.update_idletasks()

    # Change this function name later
    def select_area_local_seg(self):
        overlay = self.deps.app_state.overlay
        overlay.select_local_segmentation = True
        # Using zoom selection for local segmentation area selection
        self.zoom_controller.enable_zoom_selection()

    def toggle_local_seg_source(self):
        """Toggle the source for local segmentation between HV and HH, and rerun local segmentation if area is already selected."""
        overlay = self.deps.app_state.overlay
        if self.deps.annotation_panel.local_seg_switch.get():
            overlay.local_segmentation_source = "HV"
        else:
            overlay.local_segmentation_source = "HH"

        if overlay.local_segmentation_area is not None and overlay.show_local_segmentation:
            x_min, y_min, x_max, y_max = overlay.local_segmentation_limits
            self.run_local_segmentation(x_min, y_min, x_max, y_max)

    def clear_local_seg(self):
        """Clear local segmentation results, reset related variables, exit local segmentation view, and refresh display."""
        overlay = self.deps.app_state.overlay
        if not overlay.show_local_segmentation:
            return
        overlay.local_segmentation_area = None
        overlay.local_segmentation_mask = None
        overlay.local_segmentation_bounds = None
        overlay.show_local_segmentation = False
        self.annotation_controller.reset_annotation()
        self.display_controller.refresh_view()

    def update_local_seg_n_classes(self, value):
        """Update the number of classes for local segmentation, rerun local segmentation if area is already selected."""
        overlay = self.deps.app_state.overlay
        overlay.local_seg_n_classes = int(value)
        if overlay.local_segmentation_area is not None and overlay.show_local_segmentation:
            x_min, y_min, x_max, y_max = overlay.local_segmentation_limits
            self.run_local_segmentation(x_min, y_min, x_max, y_max)