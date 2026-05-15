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


class SceneController:
    def __init__(self, deps, scene_viewmodel):
        self.deps = deps
        self.scene_viewmodel = scene_viewmodel

    def _set_progress(self, value, text):
        self.deps.loading_bar.set(value)
        self.deps.loading_bar_label.configure(text=text)
        self.deps.app.update_idletasks()

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

            scene.scene_name = os.path.basename(os.path.normpath(scene.folder_path))

            self.deps.app.title(f"Scene {scene.scene_name}-{display.channel_mode}")

            # Show loading bar
            self.deps.loading_bar_label.grid(row=0, column=0)
            self.deps.loading_bar.grid(row=1, column=0)
            self.deps.app.update_idletasks()

            self._set_progress(0, "Loading images...")

            try:
                self.scene_viewmodel.load_scene(scene.folder_path, progress=self._set_progress)
            except (FileNotFoundError, ValueError) as e:
                messagebox.showinfo("Error", f"The selected directory does not contain the required files. Please, select a valid directory.\n\n{e}", parent=self.deps.app.master)
                scene.folder_path = ''
                self._set_progress(0, "Error loading images")
                return

            # Should be from display_controller later
            self.deps.app.choose_image()

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

            self._set_progress(1, "Inference complete")

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
        loaded_sources, missing_sources, has_existing_annotation = (
            self.scene_viewmodel.load_predictions(progress=self._set_progress)
        )
        self.deps.annotation_panel.clear_notes()
        
        # Reset label source radio buttons
        for key in self.deps.widgets['lbl_source_btn'].keys():
            self.deps.widgets['lbl_source_btn'][key].destroy()
        self.deps.widgets['lbl_source_btn'] = {}
        self.deps.widgets['mode_var_lbl_source'] = None
        self.deps.widgets['mode_var_lbl_source_prev'] = None

        for key in missing_sources:
            if key != 'Custom_Annotation':
                messagebox.showinfo("Error", f"The selected scene does not contain prediction files for {key}.", parent=self.deps.app.master)

        # Add available label sources
        for i, key in enumerate(loaded_sources):
            self.deps.app.update_label_source_widgets(key, i)

        custom_anno = "Custom_Annotation"
        if has_existing_annotation and custom_anno in scene.lbl_sources:
            result = messagebox.askyesno("Custom Annotation Found",
                                         "An existing custom annotation was found for this scene. Do you want to view it?",
                                         parent=self.deps.app.master)
            if result:
                scene.active_source = custom_anno
                self.deps.widgets['mode_var_lbl_source'].set(custom_anno)

            self.deps.app.choose_image() # Refresh image to show annotation on minimap
