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

