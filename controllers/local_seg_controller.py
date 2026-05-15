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

class LocalSegController:
    def __init__(self, deps, display_controller, annotation_controller, 
                 canvas_events_controller, zoom_controller,
                 local_segmentation_viewmodel):
        self.deps = deps
        self.display_controller = display_controller
        self.annotation_controller = annotation_controller
        self.canvas_events_controller = canvas_events_controller
        self.zoom_controller = zoom_controller
        self.local_segmentation_viewmodel = local_segmentation_viewmodel

    def _set_progress(self, value, text):
        self.deps.loading_bar.set(value)
        self.deps.loading_bar_label.configure(text=text)
        self.deps.app.update_idletasks()

    def run_local_segmentation(self, x_min, y_min, x_max, y_max):
        """
        Run local segmentation (IRGS) on the area selected by the user, 
        update the overlay with the local segmentation results, and refresh the display.
        """
        self.deps.canvas.delete(self.canvas_events_controller.selection_rect_id)
        self.canvas_events_controller.selection_rect_id = None
        self.canvas_events_controller.selection_start_coord = None

        # Show loading bar
        self.deps.loading_bar_label.grid(row=0, column=0)
        self.deps.loading_bar.grid(row=1, column=0)
        self.deps.app.update_idletasks()

        self.local_segmentation_viewmodel.run(
            x_min,
            y_min,
            x_max,
            y_max,
            progress=self._set_progress,
        )

        self.annotation_controller.reset_annotation() # Reset annotation to prevent annotation on old local segmentation
        self.display_controller.refresh_view()

        self._set_progress(1, "Local segmentation applied")

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
        source = "HV" if self.deps.annotation_panel.local_seg_switch.get() else "HH"
        self.local_segmentation_viewmodel.set_source(source)

        if overlay.local_segmentation_area is not None and overlay.show_local_segmentation:
            x_min, y_min, x_max, y_max = overlay.local_segmentation_limits
            self.run_local_segmentation(x_min, y_min, x_max, y_max)

    def clear_local_seg(self):
        """Clear local segmentation results, reset related variables, exit local segmentation view, and refresh display."""
        if not self.local_segmentation_viewmodel.clear():
            return
        self.annotation_controller.reset_annotation()
        self.display_controller.refresh_view()

    def update_local_seg_n_classes(self, value):
        """Update the number of classes for local segmentation, rerun local segmentation if area is already selected."""
        overlay = self.deps.app_state.overlay
        self.local_segmentation_viewmodel.set_n_classes(value)
        if overlay.local_segmentation_area is not None and overlay.show_local_segmentation:
            x_min, y_min, x_max, y_max = overlay.local_segmentation_limits
            self.run_local_segmentation(x_min, y_min, x_max, y_max)
