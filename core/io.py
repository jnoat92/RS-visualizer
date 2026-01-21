'''
Image input/output related functions
Handles loading of images and predictions and path to external resources

Last modified: Jan 2026
'''

from PIL import Image
import numpy as np
import os
import json
import sys

from core.utils import rgb2gray
from core.utils import generate_boundaries
from core.parallel_stuff import Parallel
from core.contrast_handler import prepare_sorted_data

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


# Should change the file path to be more robust, possibly reading a template file to find the images to grab
def load_images(folder_path):

    try:
        HH = np.asarray(Image.open(folder_path + "/imagery_HH_UW_4_by_4_average.tif")) 
        HV = np.asarray(Image.open(folder_path + "/imagery_HV_UW_4_by_4_average.tif"))

        HH_better_contrast = np.asarray(Image.open(folder_path + "/enhanced_images/imagery_HH_UW_4_by_4_average.png"))
        HV_better_contrast = np.asarray(Image.open(folder_path + "/enhanced_images/imagery_HV_UW_4_by_4_average.png"))

    except FileNotFoundError as e:
        return e

    img_base = {}
    img_base["HH"] = np.tile(HH[:,:,np.newaxis], (1,1,3))
    img_base["HV"] = np.tile(HV[:,:,np.newaxis], (1,1,3))
    img_base["(HH, HH, HV)"] = np.stack([HH, HH, HV], axis=-1)
    img_base["(HH, HV, HV)"] = np.stack([HH, HV, HV], axis=-1)

    img_better_contrast = {}
    img_better_contrast["HH"] = np.tile(HH_better_contrast[:,:,np.newaxis], (1,1,3))
    img_better_contrast["HV"] = np.tile(HV_better_contrast[:,:,np.newaxis], (1,1,3))
    img_better_contrast["(HH, HH, HV)"] = np.stack([HH_better_contrast, HH_better_contrast, HV_better_contrast], axis=-1)
    img_better_contrast["(HH, HV, HV)"] = np.stack([HH_better_contrast, HV_better_contrast, HV_better_contrast], axis=-1)

    return (img_base, img_better_contrast)

def load_base_images(folder_path):
    try:
        HH = np.asarray(Image.open(folder_path + "/imagery_HH_UW_4_by_4_average.tif")) 
        HV = np.asarray(Image.open(folder_path + "/imagery_HV_UW_4_by_4_average.tif"))

    except FileNotFoundError as e:
        return e, {}, {}, {}, {}, {}, {}

    raw_img = {}
    raw_img["HH"] = HH
    raw_img["HV"] = HV

    img_base = {}
    img_base["HH"] = np.tile(HH[:,:,np.newaxis], (1,1,3))
    img_base["HV"] = np.tile(HV[:,:,np.newaxis], (1,1,3))

    nan_mask = {}
    nan_mask["HH"] = np.isnan(HH)
    nan_mask["HV"] = np.isnan(HV)

    data_sorted = {}
    for img_type in img_base.keys():
        data_sorted[img_type] = prepare_sorted_data(img_base[img_type], valid_mask=~nan_mask[img_type])
    return raw_img, img_base, data_sorted, nan_mask


def load_prediction(folder_path, filenames, lbl_source):

    file_names = [folder_path + f for f in filenames]
    
    if len(lbl_source) > 1:
        variables = Parallel(PredictionLoader, zip(lbl_source, file_names))
    else:
        variables = [PredictionLoader(zip(lbl_source[0], file_names[0]))]

    return variables


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Future combine with loading evaluation?
def load_existing_annotation(scene_name):
    folder_name = "Custom_Annotation"
    folder_path = os.path.join(folder_name, scene_name)
    file_path = folder_path + "/custom_annotation.png"
    notes_file_path = folder_name + "/annotation_notes.json"
    notes = ""
    if os.path.exists(folder_path) and os.path.exists(file_path):
        #print(folder_path)
        annotation_file = os.path.join(folder_path, "custom_annotation.png")
        custom_anno_variable = PredictionLoader(("Custom_Annotation", annotation_file))
        if os.path.exists(notes_file_path):
            with open(notes_file_path, 'r') as f:
                try:
                    existing_notes = json.load(f)
                    if scene_name in existing_notes:
                        notes = existing_notes[scene_name].get("notes", "").strip()
                except json.JSONDecodeError:
                    pass 
        return custom_anno_variable, notes
    else:
        return None, notes
        
