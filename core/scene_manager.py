import os
import numpy as np

from core.io import load_images, load_prediction

class SceneManager:
    def load_folder(self, app_state, folder_path):
        scene = app_state.scene
        scene.folder_path = folder_path
        scene.filenames = self._scan_folder(folder_path)
        scene.current_index = 0

        # clear current scene data
        scene.img = None
        scene.img_name = ""
        scene.predictions.clear()
        scene.landmasks.clear()
        scene.boundmasks.clear()
        scene.active_source = ""

    def scan_folder(self, folder_path: str) -> list[str]:
        valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
        files.sort()
        return files

    def load_current_scene(self, app_state) -> None:
        scene = app_state.scene
        disp = app_state.display

        if not scene.folder_path:
            raise ValueError("No folder selected.")
        if not scene.filenames:
            raise ValueError("No images found in folder.")

        filename = scene.filenames[scene.current_index]

        # 1) Load images for folder (or scene) using your core/io.py
        images_dict = load_images(scene.folder_path)

        # 2) Choose which image to display based on disp.channel_mode / better_contrast
        scene.img, scene.img_name = self._choose_image(images_dict, disp, fallback=filename)

        # 3) Load predictions/masks using your existing core/io.py helper
        preds, land, bound = load_prediction(
            folder_path=scene.folder_path,
            lbl_sources=scene.lbl_sources,
            # any other args your load_predictions needs (filenames/templates/etc.)
        )

        scene.predictions = preds
        scene.landmasks = land
        scene.boundmasks = bound

        # 4) Set active source safely
        scene.active_source = self._choose_active_source(scene.active_source, scene.predictions)

    def choose_image(self, images_dict, disp, fallback: str):
        """
        Implement your current 'Choose_image' logic here.
        For now: use disp.channel_mode to pick a key.
        """
        key = disp.channel_mode

        # if your dict keys don't match channel_mode exactly, map them here
        if key in images_dict:
            return images_dict[key], key

        # fallback: first entry
        first_key = next(iter(images_dict))
        return images_dict[first_key], first_key

    def choose_active_source(self, preferred: str, preds: dict) -> str:
        if preferred and preferred in preds:
            return preferred
        if preds:
            return next(iter(preds.keys()))
        return ""

    def step(self, app_state, step: int) -> None:
        scene = app_state.scene
        if not scene.filenames:
            return
        scene.current_index = (scene.current_index + step) % len(scene.filenames)
        self.load_current_scene(app_state)
