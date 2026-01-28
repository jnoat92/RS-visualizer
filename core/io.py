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

from core.utils import rgb2gray, generate_boundaries, prepare_sorted_data_numba
from core.parallel_stuff import Parallel
from core.contrast_handler import prepare_sorted_data

def PredictionLoader(iterator):
    key, filename = iterator

    try:
        pred = np.asarray(Image.open(filename)).copy()
    except FileNotFoundError as e:
        print(f"The selected directory does not contain the required prediction files. Please, select a valid directory.\n\n{e}")
        return key, None, None, None
    
    print("Pred shape:", pred.shape)
    
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

    data_sorted = {}
    # for img_type in img_base.keys():
    #     print("Preparing sorted data for ", img_type)
    #     data_sorted[img_type] = prepare_sorted_data_numba(img_base[img_type], valid_mask=~nan_mask[img_type])
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
            print(f"hh shape: {hh.shape}, hv shape: {hv.shape}")
        
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

        print("RCM product loaded successfully. Returning data...")

        raw_img, img_base, data_sorted, nan_mask = setup_base_images(np.array(hh), np.array(hv))
        print(type(hh[1,1]), type(hv[1,1]))
        save_hh_img = Image.fromarray(hh, mode="F")
        save_hh_img.save("hh_test.tiff")

        save_hv_img = Image.fromarray(hv, mode="F")
        save_hv_img.save("hv_test.tiff")

        return {
            "folder_name": data_dir,
            "product_id": product_id,
            "hh": hh,
            "hv": hv,
            "pixel_spacing": pixel_spacing,
            "geocoded_points": geocoded_points,
            "xml": xml_root
        }, raw_img, img_base, data_sorted, nan_mask

    except Exception as e:
        print(f"Skipping {data_dir}: {e}")
        return None