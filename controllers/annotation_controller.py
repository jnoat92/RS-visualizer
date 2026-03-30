'''
AnnotationController manages the annotation process, including handling 
user interactions for drawing and labeling. It contains methods for
polygon drawing, rectangle drawing, bucket fill, and label assignment.
It also manages the iteractions with local segmentation and 
undo/redo functionality.

Last modified: Mar 2026
'''

import numpy as np
from tkinter import messagebox
from core.segmentation import get_segment_contours

class AnnotationController:
    def __init__(self, deps, display_controller):
        self.deps = deps
        self.display = display_controller
        self.selected_polygon = None

    # Move draw_rectangle, draw_polygon, _add_polygon_point,
    # draw_polygon_on_canvas, draw_single_polygon_on_canvas,
    # _finish_polygon, reset_annotation, annotate_class,
    # undo_redo_annotation, check_existing_annotation,
    # bucket_fill, bucket_fill_polygon_area, exit_bucket_fill
    # label-specfic methods