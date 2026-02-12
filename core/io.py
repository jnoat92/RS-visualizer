'''
Image input/output related functions
Handles loading of images and predictions and path to external resources

Last modified: Feb 2026
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
from rasterio.warp import reproject, Resampling, calculate_default_transform
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
from rasterio.transform import xy, rowcol
from pyproj import Transformer
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from model.model_helper import Normalize_min_max, load_model, forward_model

from core.utils import rgb2gray, generate_boundaries
from core.parallel_handler import Parallel
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

def setup_base_images(HH, HV, nan_mask_hh, nan_mask_hv):
    raw_img = {}
    raw_img["HH"] = HH
    raw_img["HV"] = HV

    img_base = {}
    img_base["HH"] = np.tile(HH[:,:,np.newaxis], (1,1,3))
    img_base["HV"] = np.tile(HV[:,:,np.newaxis], (1,1,3))

    nan_mask = {}
    nan_mask["HH"] = nan_mask_hh
    nan_mask["HV"] = nan_mask_hv

    hist = {}
    n_valid = {}
    for img_type in img_base.keys():
        hist[img_type], n_valid[img_type] = precompute_valid_hist_u8(img_base[img_type], valid_mask=~nan_mask[img_type])
    return raw_img, img_base, hist, n_valid, nan_mask

# Keeping for future when we have more models
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
    annotated_area_path = folder_path + "/annotated_area.npz"
    notes = ""
    minimap_area_idx = None
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
        if os.path.exists(annotated_area_path):
            try:
                data = np.load(annotated_area_path)
                minimap_area_idx = data["area_idx"]
            except Exception as e:
                print("Error loading annotated area:", e)

        return custom_anno_variable, notes, minimap_area_idx
    else:
        return None, notes, minimap_area_idx
    
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
    if len(img_file) != 1:
        img_file = list(Path(data_dir).glob("*.tif"))
        if len(img_file) != 1:
            raise ValueError("expected one .img or .tif file")

    xml_file = list(Path(data_dir).glob("product.xml"))

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
            src_transform = src.transform
            src_crs = src.crs
            src_bounds = src.bounds
            nodata_hh = src.nodatavals[0]
            nodata_hv = src.nodatavals[1]
       
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

        # find tie-points
        tie_pts = xml_root.findall(".//rcm:geolocationGrid/rcm:imageTiePoint", ns)

        if len(tie_pts) == 0:
            print(f"Skipping {img_path}: no tie-points found")
            # continue

        geocoded_points = []

        for tp in tie_pts:
            img = tp.find("rcm:imageCoordinate", ns)
            geo = tp.find("rcm:geodeticCoordinate", ns)

            geocoded_points.append({
                "line": float(img.find("rcm:line", ns).text),
                "pixel": float(img.find("rcm:pixel", ns).text),
                "latitude": float(geo.find("rcm:latitude", ns).text),
                "longitude": float(geo.find("rcm:longitude", ns).text)
            })
 
        return {
            "folder_name": data_dir,
            "product_id": product_id,
            "hh": hh,
            "hv": hv,
            "pixel_spacing": pixel_spacing,
            "geocoded_points": geocoded_points,
            "xml": xml_root,
            "src_transform": src_transform,
            "src_crs": src_crs,
            "src_bounds": src_bounds,
            "nodata_hh": nodata_hh,
            "nodata_hv": nodata_hv,
        }
 
    except Exception as e:
        print(f"Skipping {data_dir}: {e}")
        return None
    
def scale_hh_hv_to_200m(rcm_data, target_spacing_m=200):
    """
    Loop over all RCM product folders in data_dir, rescale HH/HV to 200 m,
    and save the rescaled .img next to the original .img.
    """
    # calculate target transform & shape
    dst_transform, dst_width, dst_height = calculate_default_transform(
        rcm_data["src_crs"],
        rcm_data["src_crs"],
        rcm_data["hh"].shape[1],    # cols,
        rcm_data["hh"].shape[0],    # rows,
        *rcm_data["src_bounds"],
        resolution=target_spacing_m
    )

    # allocate outputs
    hh_200m = np.empty((dst_height, dst_width), dtype=np.float32)
    hv_200m = np.empty((dst_height, dst_width), dtype=np.float32)

    # resample HH
    reproject(
        source=rcm_data["hh"],
        destination=hh_200m,
        src_transform=rcm_data["src_transform"],
        src_crs=rcm_data["src_crs"],
        dst_transform=dst_transform,
        dst_crs=rcm_data["src_crs"],
        resampling=Resampling.average,
        src_nodata=rcm_data["nodata_hh"],
        dst_nodata=np.nan
    )

    # resample HV
    reproject(
        source=rcm_data["hv"],
        destination=hv_200m,
        src_transform=rcm_data["src_transform"],
        src_crs=rcm_data["src_crs"],
        dst_transform=dst_transform,
        dst_crs=rcm_data["src_crs"],
        resampling=Resampling.average,
        src_nodata=rcm_data["nodata_hv"],
        dst_nodata=np.nan
    )

    # Create transformer for geocoding later
    transformer = Transformer.from_crs(rcm_data["src_crs"], "EPSG:4326", always_xy=True)
    
    # # create output folder inside product_dir
    # out_dir = product_dir / "200m_pixel_spacing"
    # out_dir.mkdir(exist_ok=True)
    # out_path = out_dir / (img_path.stem + "_200m.img")

    # # save unified .img
    # with rasterio.open(
    #     out_path,
    #     "w",
    #     driver="HFA",
    #     height=hh_200m.shape[0],
    #     width=hh_200m.shape[1],
    #     count=2,
    #     dtype=hh_200m.dtype,
    #     crs=src_crs,
    #     transform=dst_transform
    # ) as dst:
    #     dst.write(hh_200m, 1)
    #     dst.write(hv_200m, 2)
    #     dst.set_band_description(1, "HH")
    #     dst.set_band_description(2, "HV")

    return {
        "hh": hh_200m, 
        "hv": hv_200m,
        "src_transform": dst_transform,
        "src_crs": rcm_data["src_crs"],
        "src_bounds": rcm_data["src_bounds"],
        "folder_name": rcm_data["folder_name"],
        "dst_height": dst_height,
        "dst_width": dst_width,
        "transformer": transformer
    }

def load_rcm_base_images(rcm_data):
    
    rcm_200m_data = scale_hh_hv_to_200m(rcm_data, target_spacing_m=200)
    hh = rcm_200m_data["hh"]
    hv = rcm_200m_data["hv"]

    # Helpful geocoding info for transforming pixel->lat/lon
    geo_coord_helpers = {"dst_transform": rcm_200m_data["src_transform"],
                  "dst_crs": rcm_200m_data["src_crs"],
                  "transformer": rcm_200m_data["transformer"]}

    land_mask = build_land_masks(
        resource_path("landmask/StatCan_ocean.shp"),
        rcm_200m_data
    )["land_mask"]

    # Normalize HH band to uint8 for visualization
    nan_mask_hh = np.isnan(hh)
    min_ = hh[~nan_mask_hh].min(0)
    max_ = hh[~nan_mask_hh].max(0)
    hh_u8 = np.zeros_like(hh, dtype=np.uint8)

    if max_ > min_:
        hh_u8[~nan_mask_hh] = np.clip(
            255 * (hh[~nan_mask_hh] - min_) / (max_ - min_),
            0, 255
        ).astype(np.uint8)

    hh = hh_u8

    # Normalize HV band to uint8 for visualization
    nan_mask_hv = np.isnan(hv)
    min_ = hv[~nan_mask_hv].min(0)
    max_ = hv[~nan_mask_hv].max(0)
    hv_u8 = np.zeros_like(hv, dtype=np.uint8)

    if max_ > min_:
        hv_u8[~nan_mask_hv] = np.clip(
            255 * (hv[~nan_mask_hv] - min_) / (max_ - min_),
            0, 255
        ).astype(np.uint8)

    hv = hv_u8

    raw_img, img_base, hist, n_valid, nan_mask = setup_base_images(hh, hv, nan_mask_hh, nan_mask_hv)

    return raw_img, img_base, hist, n_valid, nan_mask, land_mask, rcm_200m_data, geo_coord_helpers

def run_pred_model(lbl_source, img, land_mask, model_path, device='cpu'):
    valid_mask = ~np.isnan(img["hh"])
    img_norm = Normalize_min_max(np.stack([img["hh"], img["hv"]], axis=-1),
                             valid_mask=valid_mask)
    device= 'cpu'   #state variable
    img_norm = torch.permute(torch.Tensor(img_norm[None, ...]).to(device), (0, 3, 1, 2)).float()

    model = load_model(model_path, device='cpu')

    colored_pred_map = forward_model(model, img_norm, nan_mask=valid_mask) # make sure nan_mask is passed

    colored_pred_map[land_mask == True] = [255, 255, 255]  # Set land areas to white
    colored_pred_map[valid_mask == False] = [255, 255, 255]  # Set NaN areas to white

    land_nan_mask = ~valid_mask | land_mask

    boundmask = generate_boundaries(rgb2gray(colored_pred_map))

    # Save colored_pred_map
    #Image.fromarray(colored_pred_map).save("model_prediction.png")

    return [(lbl_source, colored_pred_map, land_nan_mask, boundmask)]

def build_land_masks(shp_path: str, rcm_product: list[dict]) -> dict:
    """
      - The shapefile polygons represent OCEAN (ocean=1), so land is the inverse.
      - add boolean masks: True=land, False=other - to the dict  
    """

    # Read shapefile
    gdf_raw = gpd.read_file(shp_path)

    # Read SAR grid info from the rcm_product dict
    hh = rcm_product["hh"]
    transform = rcm_product["src_transform"]
    crs = rcm_product["src_crs"]
    bounds = rcm_product["src_bounds"]
    folder_name = rcm_product["folder_name"]
    shape = hh.shape

    # Reproject shapefile to this rcm_product CRS
    gdf = gdf_raw.to_crs(crs)

    # SAR bbox polygon
    sar_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Keep only shapefile features that intersect the SAR bbox + Clip the shapefile geometry to the SAR bbox 
    gdf = gdf[gdf.intersects(sar_bbox)].copy()
    gdf["geometry"] = gdf.intersection(sar_bbox)
    if len(gdf) == 0:
        print(f"warning: {folder_name}: shapefile does not intersect bbox")
        return None

    # Merge polygons
    geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])

    # Rasterize ocean polygons to mask
    ocean_mask = rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    ).astype(bool)

    # Convert ocean-mask to land-mask (land=True, ocean=False)
    land_mask = ~ocean_mask

    rcm_product['land_mask'] = land_mask

    return rcm_product
