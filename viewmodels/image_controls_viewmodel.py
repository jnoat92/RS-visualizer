from core.contrast_handler import enhance_outlier_slider
from core.render import layer_imagery


class ImageControlsViewModel:
    """Owns image-control state changes and derived image arrays."""

    def __init__(self, app_state, display_viewmodel):
        self.app_state = app_state
        self.display_viewmodel = display_viewmodel

    def set_color_composite(self, channel_mode, hh_hv_switch_on=False):
        display = self.app_state.display
        display.channel_mode = channel_mode
        if display.channel_mode == "(HH/HV)":
            display.channel_mode = "HV" if hh_hv_switch_on else "HH"
            return True
        return False

    def set_hh_hv_channel(self, use_hv):
        self.app_state.display.channel_mode = "HV" if use_hv else "HH"

    def apply_contrast(self, value):
        scene = self.app_state.scene
        display = self.app_state.display
        display.contrast = (float(value) / 200) * 0.15

        if display.channel_mode in ["(HH, HH, HV)", "(HH, HV, HV)"]:
            hh_contrasted = enhance_outlier_slider(
                img_u8=scene.orig_img["HH"],
                hist=scene.hist["HH"],
                n_valid=scene.n_valid["HH"],
                s=display.contrast,
            )
            hv_contrasted = enhance_outlier_slider(
                img_u8=scene.orig_img["HV"],
                hist=scene.hist["HV"],
                n_valid=scene.n_valid["HV"],
                s=display.contrast,
            )
            scene.img = layer_imagery(
                hh_contrasted,
                hv_contrasted,
                display.channel_mode,
            )
        else:
            scene.img = enhance_outlier_slider(
                img_u8=scene.orig_img[display.channel_mode],
                hist=scene.hist[display.channel_mode],
                n_valid=scene.n_valid[display.channel_mode],
                s=display.contrast,
            )

    def reset_contrast(self):
        self.app_state.display.contrast = 0.0
        self.apply_contrast(0)

    def set_brightness(self, value):
        self.app_state.display.brightness = float(value) / 100

    def reset_brightness(self):
        self.app_state.display.brightness = 0.0

    def set_opacity(self, value):
        self.app_state.overlay.alpha = float(value) / 100

    def toggle_segmentation_overlay(self):
        overlay = self.app_state.overlay
        overlay.show_overlay = not overlay.show_overlay
        return overlay.show_overlay

