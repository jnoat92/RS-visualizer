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
import rasterio
from lxml import etree
from pathlib import Path
import cv2

from core.utils import rgb2gray, generate_boundaries, prepare_sorted_data_numba
from core.parallel_stuff import Parallel
from core.contrast_handler import precompute_valid_hist_u8

def PredictionLoader(iterator, resize=False, img_shape=None):
    key, filename = iterator

    try:
        pred = np.asarray(Image.open(filename)).copy()
    except FileNotFoundError as e:
        print(f"The selected directory does not contain the required prediction files. Please, select a valid directory.\n\n{e}")
        return key, None, None, None

    pred[(pred == [0, 0, 128]).all(axis=2)] = [0, 255, 255]
    pred[(pred == [128, 0, 0]).all(axis=2)] = [255, 130, 0]

    if resize:
        pred = cv2.resize(pred, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
        pred = np.ascontiguousarray(pred)

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

    return setup_base_images(HH, HV)

def setup_base_images(HH, HV):
    raw_img = {}
    raw_img["HH"] = HH
    raw_img["HV"] = HV

    img_base = {}
    img_base["HH"] = np.tile(HH[:,:,np.newaxis], (1,1,3))
    img_base["HV"] = np.tile(HV[:,:,np.newaxis], (1,1,3))

    nan_mask = {}
    nan_mask["HH"] = np.isnan(HH)
    nan_mask["HV"] = np.isnan(HV)

    hist = {}
    n_valid = {}
    for img_type in img_base.keys():
        print("Preparing hist for", img_type)
        hist[img_type], n_valid[img_type] = precompute_valid_hist_u8(img_base[img_type], valid_mask=~nan_mask[img_type])
    return raw_img, img_base, hist, n_valid, nan_mask


def load_prediction(folder_path, filenames, lbl_source, img_shape):
    resize_img = False
    file_names = [folder_path + f for f in filenames]

    if folder_path.split("/")[-1].startswith("RCM"):
        resize_img = True
    
    if len(lbl_source) > 1:
        variables = Parallel(PredictionLoader, zip(lbl_source, file_names))
    else:
        variables = [PredictionLoader((lbl_source[0], file_names[0]), resize=resize_img, img_shape=img_shape)]

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
        
def load_rcm_product(data_dir):
    """
    Load and parse RCM (Radarsat Constellation Mission) SAR product data.
    
    Reads a single RCM product directory containing a dual-polarization SAR image
    (.img file with HH and HV bands) and its associated XML metadata file. Extracts
    polarimetric bands, georeferencing information, and product metadata.
    
    Parameters
    ----------
    data_dir : str or Path
        Path to the RCM product directory containing:
        - One .img file (HFA raster format with HH and HV bands)
        - One product.xml file (RCM metadata with geolocation and product info)
    
    Returns
    -------
    dict or None
        Dictionary containing extracted product data if successful:
        - 'folder_name': Input directory path
        - 'product_id': RCM product identifier from metadata
        - 'hh': HH band (co-polarized, 2D numpy array)
        - 'hv': HV band (cross-polarized, 2D numpy array)
        - 'pixel_spacing': Dict with 'range_m' and 'azimuth_m' (meters)
        - 'geocoded_points': List of dicts with 'latitude' and 'longitude' (WGS84)
        - 'xml': Parsed XML root element (lxml Element)
        
        Returns None if loading fails, with error message printed to console.
    
    Raises
    ------
    ValueError
        If directory doesn't contain exactly one .img file or one product.xml file,
        or if .img file doesn't have exactly 2 bands.
    """

    img_file = list(Path(data_dir).glob("*.img"))
    xml_file = list(Path(data_dir).glob("product.xml"))

    if len(img_file) != 1:
        raise ValueError("expected one .img file")
    if len(xml_file) != 1:
        raise ValueError("expected one product.xml file")
    img_path = img_file[0]
    xml_file = xml_file[0]

    
    try:
        # read HH & HV
        with rasterio.open(img_path) as src:
            if src.count != 2:
                raise ValueError(f"expected 2 bands, found {src.count}")
            hh = src.read(1)
            hv = src.read(2)
        
        # parse product.xml
        xml_root = etree.parse(str(xml_file)).getroot()
        ns = {"rcm": xml_root.tag.split("}")[0].strip("{")}

        # metadata extraction

        # product ID
        product_id_elem = xml_root.find(".//rcm:productId", ns)
        product_id = (
            product_id_elem.text.strip() if product_id_elem is not None else None
        )

        # pixel spacing
        range_spacing_elem = xml_root.find(".//rcm:sampledPixelSpacing", ns)
        azimuth_spacing_elem = xml_root.find(".//rcm:sampledLineSpacing", ns)

        pixel_spacing = {
            "range_m": float(range_spacing_elem.text)
            if range_spacing_elem is not None else None,
            "azimuth_m": float(azimuth_spacing_elem.text)
            if azimuth_spacing_elem is not None else None,
        }

        # geocoded grid points (lon/lat)
        geocoded_points = []
        geodetic_coords = xml_root.findall(".//rcm:geodeticCoordinate", ns)

        for coord in geodetic_coords:
            lat = coord.find("rcm:latitude", ns)
            lon = coord.find("rcm:longitude", ns)

            if lat is not None and lon is not None:
                geocoded_points.append({
                    "latitude": float(lat.text),
                    "longitude": float(lon.text)
                })

        print("RCM product loaded successfully.")
        
        # Normalize HH band to uint8 for visualization
        land_nan_mask_hh = np.isnan(hh)
        min_ = hh[~land_nan_mask_hh].min(0)
        max_ = hh[~land_nan_mask_hh].max(0)
        hh = np.uint8(255*((hh - min_) / (max_ - min_)))

        # Normalize HV band to uint8 for visualization
        land_nan_mask_hv = np.isnan(hv)
        min_ = hv[~land_nan_mask_hv].min(0)
        max_ = hv[~land_nan_mask_hv].max(0)
        hv = np.uint8(255*((hv - min_) / (max_ - min_)))

        raw_img, img_base, hist, n_valid, nan_mask = setup_base_images(hh, hv)

        # save_hh_img = Image.fromarray(hh)
        # save_hh_img.save("hh_test.png")

        # save_hv_img = Image.fromarray(hv)
        # save_hv_img.save("hv_test.png")
        print("Returning RCM product data.")

        return {
            "folder_name": data_dir,
            "product_id": product_id,
            "hh": hh,
            "hv": hv,
            "pixel_spacing": pixel_spacing,
            "geocoded_points": geocoded_points,
            "xml": xml_root
        }, raw_img, img_base, hist, n_valid, nan_mask

    except Exception as e:
        print(f"Skipping {data_dir}: {e}")
        return None