'''
ZoomController manages zooming and panning interactions on the image 
canvas. It handles mouse events to allow users to zoom in/out and pan 
around the image, ensuring that the display updates accordingly. 
This controller interacts closely with the DisplayController to adjust 
the view based on user input.

Last modified: Mar 2026
'''

# Move enable_zoom_selection, zoom_to_rectangle, reset_zoom
# Most of _on_mousewheel
# pan parts of _on_left_drag

class ZoomController:
    def __init__(self, deps, display_controller):
        self.deps = deps
        self.display = display_controller
