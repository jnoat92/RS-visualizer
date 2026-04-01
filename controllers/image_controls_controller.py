'''
ImageControlsController manages the interactions between the image 
controls (like sliders, buttons) and the display, scene, and annotation 
controllers. It handles user inputs related to image manipulation, such 
as adjusting contrast, adjusting brightness, toggling overlay, and 
selecting channel. 

They share the "UI control changed -> update state -> rerender" flow.

Last modified: Apr 2026
'''

import customtkinter as ctk
from core.render import layer_imagery
from core.contrast_handler import enhance_outlier_slider

# Move color_composite, HH_HV, contrast_slider_handle, right_click_contrast_rest
# brightness_slider_handle, right_click_brightness_reset, opacity_slider_handle, segmentation_toggle

class ImageControlsController:
    def __init__(self, deps, display_controller):
        self.deps = deps
        self.display_controller = display_controller

    def color_composite(self):
        """
        Handle color composite selection changes, enable/disable HH/HV switch accordingly, and update the displayed image.
        """
        display = self.deps.app_state.display
        display.channel_mode = self.deps.widgets["mode_var_color_composite"].get()

        if display.channel_mode == "(HH/HV)":
            self.deps.widgets["hh_hv_switch"].configure(state=ctk.NORMAL)
            self.HH_HV(get_channel=True)
        else:
            self.deps.widgets["hh_hv_switch"].configure(state=ctk.DISABLED)
            self.HH_HV(get_channel=False)

        
    def HH_HV(self, get_channel=True):
        """
        Handle color composite changes, update the displayed image based on the selected channel, and reset contrast slider.
        """
        display = self.deps.app_state.display
        scene = self.deps.app_state.scene

        if get_channel:
            display.channel_mode = "HV" if self.deps.widgets["hh_hv_switch"].get() else "HH"

        self.deps.widgets["contrast_slider"].set(0)  # Reset contrast slider
        self.contrast_slider_handle(0)

        self.deps.app.title(f"Scene {scene.scene_name}-{display.channel_mode}")
        self.display_controller.choose_image()

        self.display_controller.refresh_all()

    # Image handle
    def contrast_slider_handle(self, val):
        """
        Handle contrast slider changes, apply contrast enhancement to the current image based on the selected channel(s), and refresh the display.
        """
        scene = self.deps.app_state.scene
        display = self.deps.app_state.display

        display.contrast = (val/200) * 0.15

        if display.channel_mode in ["(HH, HH, HV)", "(HH, HV, HV)"]:
            HH_contrasted = enhance_outlier_slider(
                img_u8=scene.orig_img["HH"], # Pass raw image for faster processing
                hist=scene.hist["HH"],
                n_valid=scene.n_valid["HH"],
                s=display.contrast
            )

            HV_contrasted = enhance_outlier_slider(
                img_u8=scene.orig_img["HV"], # Pass raw image for faster processing
                hist=scene.hist["HV"],
                n_valid=scene.n_valid["HV"],
                s=display.contrast
            )

            # Re-layer the imagery with new contrast
            scene.img = layer_imagery(
                HH_contrasted,
                HV_contrasted,
                display.channel_mode
            )
        else:
            scene.img = enhance_outlier_slider(
                img_u8=scene.orig_img[display.channel_mode], # Pass raw image for faster processing
                hist=scene.hist[display.channel_mode],
                n_valid=scene.n_valid[display.channel_mode],
                s=display.contrast
            )

        self.display_controller.refresh_all()

    def right_click_contrast_reset(self, event):
        """
        Handle right-click on contrast slider to reset contrast to default, refresh the display.
        """
        self.deps.widgets["contrast_slider"].set(0) # reset to default
        self.deps.app_state.display.contrast = 0.0
        self.contrast_slider_handle(0)
        self.display_controller.refresh_all()

    def brightness_slider_handle(self,val):
        """
        Handle brightness slider changes, update the displayed image based on the selected channel, and refresh the display.
        """
        self.deps.app_state.display.brightness = float(val)/100
        self.display_controller.refresh_all()

    def right_click_brightness_reset(self, event):
        """
        Handle right-click on brightness slider to reset brightness to default, refresh the display.
        """
        self.deps.widgets["brightness_slider"].set(0) # reset to default
        self.deps.app_state.display.brightness = 0.0
        self.display_controller.refresh_all()

    # Segmentation handle

    def opacity_slider_handle(self, val):
        """
        Handle opacity slider changes, update the overlay opacity, and refresh the display.
        """
        self.deps.app_state.overlay.alpha = float(val)/100
        self.display_controller.set_overlay()
        self.display_controller.display_image()

        if self.deps.app_state.anno.polygon_points_img_coor: 
            self.deps.app.draw_polygon_on_canvas()

        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                self.deps.annotation_panel.update_zoomed_display()

    def segmentation_toggle(self):
        """
        Handle segmentation overlay toggle, update the button appearance based on the state, and refresh the display.
        When 'OFF' just show base image, when 'ON' show overlay
        """
        overlay_state = self.deps.app_state.overlay
        overlay_state.show_overlay = not overlay_state.show_overlay
        state = "ON" if overlay_state.show_overlay else "OFF"
        self.deps.widgets["segmentation_toggle_btn"].configure(text=state)

        self.display_controller.display_image()

        if overlay_state.show_overlay:
            # Restore default appearance
            self.deps.widgets["segmentation_toggle_btn"].configure(
                fg_color=self.deps.widgets["default_fg_color"],  # Default customtkinter blue
                hover_color=self.deps.widgets["default_hover_color"],
                text_color=self.deps.widgets["default_text_color"]
            )
        else:
            # Set to gray when OFF
            self.deps.widgets["segmentation_toggle_btn"].configure(
                fg_color="#888888",     # Gray background
                hover_color="#777777",  # Slightly darker on hover
                text_color="white"
            )

        if self.deps.app_state.anno.polygon_points_img_coor: 
            self.deps.app.draw_polygon_on_canvas()

        if (hasattr(self.deps.annotation_panel, 'zoom_window') and 
            self.deps.annotation_panel.zoom_window is not None and 
            self.deps.annotation_panel.zoom_window.winfo_exists()):
            if self.deps.annotation_panel.zoom_window.winfo_viewable():            
                self.deps.annotation_panel.update_zoomed_display()