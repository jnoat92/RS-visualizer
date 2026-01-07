import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage.measure import find_contours, label

from skimage.morphology import binary_dilation, disk
from ui.evaluation import EvaluationPanel
from ui.annotation import AnnotationPanel
from utils import blend_overlay_cuda, blend_overlay, rgb2gray
from utils import generate_boundaries

from numba import cuda
import cv2
import os
from parallel_stuff import Parallel
import multiprocessing

def get_segment_contours(pred, x, y):
    target_rgb = pred[x, y]
    mask = np.all(pred == target_rgb, axis=-1)
    labeled = label(mask, connectivity=2)
    region_label = labeled[x, y]

    if region_label == 0:
        return []

    segment_mask = labeled == region_label

    # Get list of contours — each a Nx2 array of [row, col]
    contours = find_contours(segment_mask.astype(np.uint8), level=0.5)

    return contours, segment_mask