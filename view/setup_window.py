"""
Setup Window ran when the user clicks the "Setup" button in the menu and 
on app start.

This window allows users to choose preprocessing steps such as selecting 
the image source (in their local directory) and choosing the resolution 
level for display. It also provides options for selecting the 
normalization method to apply to the image (mean-std, min-max).
Users can also choose which model to use for predictions.

Last modified: Jun 2026
"""

import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog

class SetupWindow():
    def __init__(self, deps, scene_controller):
        self.deps = deps
        self.scene_controller = scene_controller
        self.window = None

    def open(self):
        if self.window is not None:
            return  # Window is already open

        self.window = ctk.CTkToplevel(self.deps.app)
        self.window.title("Setup")
        self.window.geometry("400x200")

        normalization_label = ctk.CTkLabel(self.window, text="Select Normalization Method:")
        normalization_label.pack(pady=10)
        default_normalization = ctk.StringVar(value=self.deps.app_state.scene.normalization_method)
        normalization_dropdown = ctk.CTkOptionMenu(self.window, 
                                            values=["mean-std", "min-max"], 
                                            variable=default_normalization,
                                            command=self.scene_controller.set_normalization_method)
        normalization_dropdown.pack(pady=10)

        model_label = ctk.CTkLabel(self.window, text="Select Model for Predictions:")
        model_label.pack(pady=10)
        model_file_btn = ctk.CTkButton(self.window, text="Choose Model File", command=self.choose_model_file)
        model_file_btn.pack(pady=10)

        # Example: Button to select image source
        select_image_btn = ctk.CTkButton(self.window, 
                                         text="Select Image Source", 
                                         command=self.scene_controller.choose_SAR_scene)
        select_image_btn.pack(pady=20)

        # Bring this window to the front
        self.window.grab_set()

        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def choose_model_file(self):
        file_path = filedialog.askopenfilename(title="Select Model File", 
                                               filetypes=[("Model Files", "*.h5 *.pt *.pth"), ("All Files", "*.*")])
        if file_path:
            self.scene_controller.set_model_file(file_path)

    def close(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None