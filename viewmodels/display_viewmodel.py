from dataclasses import dataclass

from core.overlay import compose_overlay
from core.render import crop_resize


@dataclass
class RenderResult:
    pred_resized: object
    img_resized: object
    boundmask_resized: object
    landmask_resized: object
    local_boundmask_resized: object
    draw_x: int
    draw_y: int


class DisplayViewModel:
    """Creates display-ready image arrays without depending on Tk widgets."""

    def __init__(self, app_state):
        self.app_state = app_state
        self.render_result = None
        self.overlay_image = None

    def choose_image(self):
        scene = self.app_state.scene
        display = self.app_state.display
        scene.img = scene.color_composites[display.channel_mode]
        return scene.img

    def get_minimap_data(self, show_previous_annotation):
        scene = self.app_state.scene
        custom_anno = "Custom_Annotation"
        if (
            custom_anno in scene.lbl_sources
            and show_previous_annotation
            and custom_anno in scene.predictions
        ):
            changed_area_mask = (
                scene.predictions[custom_anno][:, :, 0]
                != scene.predictions[scene.lbl_sources[0]][:, :, 0]
            )
            return scene.img, changed_area_mask
        return scene.img, None

    def render(self, canvas_width, canvas_height):
        view = self.app_state.view
        scene = self.app_state.scene
        display = self.app_state.display
        overlay = self.app_state.overlay

        result = crop_resize(
            scene.predictions[scene.active_source],
            scene.img,
            scene.boundmasks[scene.active_source],
            scene.land_nan_masks[scene.active_source],
            overlay.local_segmentation_bounds,
            scene.nan_mask["HH"],
            view.zoom_factor,
            view.offset_x,
            view.offset_y,
            display.brightness,
            canvas_width,
            canvas_height,
            overlay.show_local_segmentation,
            scene.sar_img,
        )
        if result is None:
            self.render_result = None
            self.overlay_image = None
            return None

        self.render_result = RenderResult(*result)
        self.overlay_image = compose_overlay(
            self.render_result.pred_resized,
            self.render_result.img_resized,
            self.render_result.boundmask_resized,
            self.render_result.landmask_resized,
            self.render_result.local_boundmask_resized,
            overlay.alpha,
        )
        return self.render_result

    def current_display_image(self):
        overlay = self.app_state.overlay
        if self.render_result is None:
            return None
        if overlay.show_overlay:
            return self.overlay_image
        return self.render_result.img_resized.astype("uint8")

    def update_overlay_only(self):
        if self.render_result is None:
            return None
        overlay = self.app_state.overlay
        self.overlay_image = compose_overlay(
            self.render_result.pred_resized,
            self.render_result.img_resized,
            self.render_result.boundmask_resized,
            self.render_result.landmask_resized,
            self.render_result.local_boundmask_resized,
            overlay.alpha,
        )
        return self.overlay_image

