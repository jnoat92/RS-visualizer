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

def PredictionLoader(iterator):
    key, filename = iterator

    try:
        pred = np.asarray(Image.open(filename)).copy()
    except FileNotFoundError as e:
        print(f"The selected directory does not contain the required prediction files. Please, select a valid directory.\n\n{e}")
        return key, None, None, None
    
    pred[(pred == [0, 0, 128]).all(axis=2)] = [0, 255, 255]
    pred[(pred == [128, 0, 0]).all(axis=2)] = [255, 130, 0]
    landmask = (pred == [255, 255, 255]).all(axis=2)

    boundmask = generate_boundaries(rgb2gray(pred))

    return key, pred, landmask, boundmask

def load_images(self):

    try:
        HH = np.asarray(Image.open(self.folder_path + "/imagery_HH_UW_4_by_4_average.tif")) 
        HV = np.asarray(Image.open(self.folder_path + "/imagery_HV_UW_4_by_4_average.tif"))

        HH_better_contrast = np.asarray(Image.open(self.folder_path + "/enhanced_images/imagery_HH_UW_4_by_4_average.png"))
        HV_better_contrast = np.asarray(Image.open(self.folder_path + "/enhanced_images/imagery_HV_UW_4_by_4_average.png"))

    except FileNotFoundError as e:
        messagebox.showinfo("Error", f"The selected directory does not contain the required files. Please, select a valid directory.\n\n{e}", parent=self.master)
        return 0

    self.img_ = {}
    self.img_["HH"] = np.tile(HH[:,:,np.newaxis], (1,1,3))
    self.img_["HV"] = np.tile(HV[:,:,np.newaxis], (1,1,3))
    self.img_["(HH, HH, HV)"] = np.stack([HH, HH, HV], axis=-1)
    self.img_["(HH, HV, HV)"] = np.stack([HH, HV, HV], axis=-1)

    self.img_Better_contrast = {}
    self.img_Better_contrast["HH"] = np.tile(HH_better_contrast[:,:,np.newaxis], (1,1,3))
    self.img_Better_contrast["HV"] = np.tile(HV_better_contrast[:,:,np.newaxis], (1,1,3))
    self.img_Better_contrast["(HH, HH, HV)"] = np.stack([HH_better_contrast, HH_better_contrast, HV_better_contrast], axis=-1)
    self.img_Better_contrast["(HH, HV, HV)"] = np.stack([HH_better_contrast, HV_better_contrast, HV_better_contrast], axis=-1)

    return 1

def load_prediction(self):

    self.predictions = {}
    self.landmasks = {}
    self.boundmasks = {}

    filenames = [self.folder_path + f for f in self.filenames]
        
    if len(self.lbl_source) > 1:
        variables = Parallel(PredictionLoader, zip(self.lbl_source, filenames))
    else:
        variables = [PredictionLoader(zip(self.lbl_source, filenames))]
        
    # variables = [PredictionLoader(it) for it in zip(lbl_source, filenames)]
        
    # Reset label source radio buttons
    for key in self.lbl_source_buttom.keys():
        self.lbl_source_buttom[key].destroy()
    self.lbl_source_buttom = {}
    self.mode_var_lbl_source = None
    self.mode_var_lbl_source_prev = None

    # Add available label sources
    for i, (key, pred, landmask, boundmask) in enumerate(variables):
        if pred is None: 
            if key != 'Custom_Annotation':
                messagebox.showinfo("Error", f"The selected scene does not contain prediction files for {key}.", parent=self.master)
            continue
        self.update_label_source_widgets(key, i)
        self.predictions[key] = pred
        self.landmasks[key] = landmask
        self.boundmasks[key] = boundmask