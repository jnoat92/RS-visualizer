'''
CanvasEventsController: Handles mouse events on the canvas, including 
clicks, double-clicks, and movements. It interacts with the display, 
zoom, annotation, and panel controllers to perform actions based on 
user input.

Last modified: Mar 2026
'''

class CanvasEventsController:
    '''Controller for handling canvas events such as mouse clicks and movements.'''
    def __init__(self, deps, display_controller, zoom_controller, 
                 annotation_controller, panel_controller):
        self.deps = deps
        self.display = display_controller
        self.zoom = zoom_controller
        self.annotation = annotation_controller
        self.panels = panel_controller
        self.double_click_flag = False
        self.selection_start_coord = None
        self.selection_rect_id = None

    