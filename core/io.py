'''
Image input/output related functions
Handles loading of images and predictions and path to external resources

Last modified: Mar 2026
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
from rasterio.warp import reproject, Resampling, calculate_default_transform, transform_bounds, transform as crs_transform
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.env import Env
from rasterio.crs import CRS
from rasterio.transform import array_bounds, Affine, from_origin
from rasterio.control import GroundControlPoint
from rasterio.coords import BoundingBox
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

from pyproj import Transformer
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from model.model_helper import Normalize_min_max, load_model, forward_model_committee

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

    if resize and img_shape != pred.shape[:2]:
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
def load_existing_annotation(scene_name, shape=None):
    folder_name = "Custom_Annotation"
    folder_path = os.path.join(folder_name, scene_name)
    file_path = folder_path + "/custom_annotation.png"
    notes_file_path = folder_name + "/annotation_notes.json"
    notes = ""
    if os.path.exists(folder_path) and os.path.exists(file_path):
        #print(folder_path)
        annotation_file = os.path.join(folder_path, "custom_annotation.png")
        if shape is not None:
            custom_anno_variable = PredictionLoader(("Custom_Annotation", annotation_file), resize=True, img_shape=shape)
        else:
            custom_anno_variable = PredictionLoader(("Custom_Annotation", annotation_file), resize=False)
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

    def _is_identity_transform(t, tol=1e-9) -> bool:
            # rasterio Affine: (a,b,c,d,e,f)
            return (
                abs(t.a - 1.0) < tol and abs(t.b) < tol and abs(t.c) < tol and
                abs(t.d) < tol and abs(t.e + 1.0) < tol and abs(t.f) < tol
            ) or (
                # some files may use +1 for y as well
                abs(t.a - 1.0) < tol and abs(t.b) < tol and abs(t.c) < tol and
                abs(t.d) < tol and abs(t.e - 1.0) < tol and abs(t.f) < tol
            )

    def _detect_geometry(src_crs, src_transform, gcps_count: int) -> str:
        """
        Returns: "earth" or "sensor"
        """
        if src_crs is not None:
            # Usually earth geometry (geocoded) if CRS exists.
            # Still allow identity transform edge case: keep it "earth" if CRS exists.
            return "earth"

        # No CRS: most likely sensor geometry. If also identity transform, definitely sensor.
        # Presence of GCPs strengthens sensor case.
        return "sensor"

    def _parse_tie_points(xml_root, ns):
        """
        Parse RCM image tie points.
        Returns a dict with NumPy arrays:
            tie_lines   : (N,)
            tie_pixels  : (N,)
            tie_lats    : (N,)
            tie_lons    : (N,)
        """
        lines = []
        pixels = []
        lats = []
        lons = []
        tps = xml_root.findall(".//rcm:geolocationGrid//rcm:imageTiePoint", ns)
        for tp in tps:
            img = tp.find(".//rcm:imageCoordinate", ns)
            geo = tp.find(".//rcm:geodeticCoordinate", ns)
            if img is None or geo is None:
                continue
            line = img.find("rcm:line", ns)
            pixel = img.find("rcm:pixel", ns)
            lat = geo.find("rcm:latitude", ns)
            lon = geo.find("rcm:longitude", ns)
            if None in (line, pixel, lat, lon):
                continue
            lines.append(float(line.text))
            pixels.append(float(pixel.text))
            lats.append(float(lat.text))
            lons.append(float(lon.text))
        return {
            "tie_lines": np.array(lines, dtype=np.float64),
            "tie_pixels": np.array(pixels, dtype=np.float64),
            "tie_lats": np.array(lats, dtype=np.float64),
            "tie_lons": np.array(lons, dtype=np.float64),
        }

    def _infer_axis_mapping_from_xml(pixel_spacing):
        """
        For SAR products:
        sampledPixelSpacing -> sample/pixel spacing along x-axis (cols), typically range
        sampledLineSpacing  -> line spacing along y-axis (rows), typically azimuth
        """
        # Return a standard mapping dict you can use later.
        return {
            "cols_are": "range",   # sample direction
            "rows_are": "azimuth", # line direction
            "range_spacing_m": pixel_spacing.get("range_m"),
            "azimuth_spacing_m": pixel_spacing.get("azimuth_m"),
            "confidence": "high (SAR convention: samples=pixels=cols, lines=rows)"
        }

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

            gcps, gcp_crs = src.gcps
            gcps_count = len(gcps) if gcps is not None else 0
       
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

        # geocoded points (lat/lon only) - keep your original behavior
        geocoded_points = []
        geodetic_coords = xml_root.findall(".//rcm:geodeticCoordinate", ns)
        for coord in geodetic_coords:
            lat = coord.find("rcm:latitude", ns)
            lon = coord.find("rcm:longitude", ns)
            if lat is not None and lon is not None:
                geocoded_points.append({"latitude": float(lat.text), "longitude": float(lon.text)})

        # find tie-points
        tie_pts = _parse_tie_points(xml_root, ns)

        # Detect geometry type (earth vs sensor) based on CRS
        geometry = _detect_geometry(src_crs, src_transform, gcps_count)
 
        # Axis mapping assumption
        axis_mapping = _infer_axis_mapping_from_xml(pixel_spacing)

        out = {
            "folder_name": str(data_dir),
            "product_id": product_id,

            "hh": hh,
            "hv": hv,

            # original fields
            # "pixel_spacing": pixel_spacing,
            "geocoded_points": geocoded_points,
            "xml": xml_root,

            "src_transform": src_transform,
            "src_crs": src_crs, 
            "src_bounds": src_bounds,

            "nodata_hh": nodata_hh,
            "nodata_hv": nodata_hv,
            # new fields for routing + verification
            "geometry": geometry,                 # "earth" or "sensor"
            "transform_is_identity": _is_identity_transform(src_transform),
            "gcps_count": gcps_count,
            "gcp_crs": gcp_crs.to_string() if gcp_crs is not None else None,
            "axis_mapping": axis_mapping,
        }

        # Convenience: keys expected by your sensor downsampler (if you keep that API)
        out["range_pixel_spacing_m"] = pixel_spacing["range_m"]
        out["azimuth_pixel_spacing_m"] = pixel_spacing["azimuth_m"]

        out.update(tie_pts) # add tie points to the output dict (lines, pixels, lats, lons)

        return out
 
    except Exception as e:
        print(f"Skipping {data_dir}: {e}")
        return None
    
def scale_hh_hv_earth_geometry(rcm_data, target_spacing_m=200):
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

    # Scale tie-point image coordinates to match the new pixel spacing. 
    # This is a rough adjustment that assumes the tie points are on a regular grid 
    # and that the resampling is approximately uniform. For more complex cases, 
    # might want to re-derive tie points after resampling or use a more sophisticated approach.
    src_h, src_w = rcm_data["hh"].shape
    rs = dst_height / src_h   # row scale
    cs = dst_width  / src_w   # col scale

    return {
        "hh": hh_200m, 
        "hv": hv_200m,
        "src_transform": dst_transform,
        "src_crs": rcm_data["src_crs"],
        "src_bounds": rcm_data["src_bounds"],
        "folder_name": rcm_data["folder_name"],
        "geometry": rcm_data["geometry"],

        "dst_height": dst_height,
        "dst_width": dst_width,
        "transformer": transformer,

        "tie_lines":  rcm_data["tie_lines"] * rs,
        "tie_pixels": rcm_data["tie_pixels"] * cs,
        "tie_lats": rcm_data["tie_lats"],
        "tie_lons": rcm_data["tie_lons"],
    }

def scale_hh_hv_sensor_geometry(
    rcm_data,
    target_spacing_m=200.0,
    range_spacing_m_key="range_pixel_spacing_m",
    azimuth_spacing_m_key="azimuth_pixel_spacing_m",
):
    hh = rcm_data["hh"]
    hv = rcm_data["hv"]

    range_spacing = float(rcm_data[range_spacing_m_key])   # meters/pixel in cols (usually range)
    az_spacing    = float(rcm_data[azimuth_spacing_m_key]) # meters/pixel in rows (usually azimuth)

    if range_spacing <= 0 or az_spacing <= 0:
        raise ValueError(f"Invalid spacings: range={range_spacing}, azimuth={az_spacing}")

    src_h, src_w = hh.shape

    sx = target_spacing_m / range_spacing
    sy = target_spacing_m / az_spacing

    dst_w = max(1, int(np.round(src_w / sx)))
    dst_h = max(1, int(np.round(src_h / sy)))

    # Treat sensor grid as a planar metric grid (not Earth), but still define transforms in meters
    src_transform = Affine(range_spacing, 0, 0,
                           0, -az_spacing, 0)
    dst_transform = Affine(target_spacing_m, 0, 0,
                           0, -target_spacing_m, 0)

    # Dummy CRS to satisfy rasterio/gdal; since src_crs == dst_crs there is no real reprojection.
    dummy_crs = CRS.from_epsg(3857)

    hh_out = np.empty((dst_h, dst_w), dtype=np.float32)
    hv_out = np.empty((dst_h, dst_w), dtype=np.float32)

    reproject(
        source=hh,
        destination=hh_out,
        src_transform=src_transform,
        src_crs=dummy_crs,
        dst_transform=dst_transform,
        dst_crs=dummy_crs,
        resampling=Resampling.average,
        src_nodata=rcm_data.get("nodata_hh", 0.0),
        dst_nodata=np.nan,
    )

    reproject(
        source=hv,
        destination=hv_out,
        src_transform=src_transform,
        src_crs=dummy_crs,
        dst_transform=dst_transform,
        dst_crs=dummy_crs,
        resampling=Resampling.average,
        src_nodata=rcm_data.get("nodata_hv", 0.0),
        dst_nodata=np.nan,
    )

    # Scale tie-point image coordinates to match the new pixel spacing. 
    # This is a rough adjustment that assumes the tie points are on a regular grid 
    # and that the resampling is approximately uniform. For more complex cases, 
    # might want to re-derive tie points after resampling or use a more sophisticated approach.
    rs = dst_h / src_h   # row scale
    cs = dst_w / src_w   # col scale

    return {
        "hh": hh_out,
        "hv": hv_out,

        # Keep compatibility with your other function’s output keys:
        "src_transform": dst_transform,
        "src_crs": None,                 # still sensor-geometry (no real CRS)
        "src_bounds": None,              # not meaningful in Earth coords

        "src_height": src_h,
        "src_width": src_w,
        "dst_height": dst_h,
        "dst_width": dst_w,
        "transformer": None,

        "sensor_transform": dst_transform,
        range_spacing_m_key: target_spacing_m,
        azimuth_spacing_m_key: target_spacing_m,
        "folder_name": rcm_data.get("folder_name", None),

        "geometry": rcm_data["geometry"],  # preserve geometry type for downstream logic
        
        "tie_lines":  rcm_data["tie_lines"] * rs,
        "tie_pixels": rcm_data["tie_pixels"] * cs,
        "tie_lats": rcm_data["tie_lats"],
        "tie_lons": rcm_data["tie_lons"],
    }

def scale_hh_hv(rcm_data, target_spacing=200):
    # Keeping 200m scaling for prediction model input, but also return scaled data for visualization
    if rcm_data["geometry"] == "earth":
        rcm_200m_data = scale_hh_hv_earth_geometry(rcm_data, target_spacing_m=200)
        rcm_scaled_data = scale_hh_hv_earth_geometry(rcm_data, target_spacing_m=target_spacing)
    else:
        rcm_200m_data = scale_hh_hv_sensor_geometry(rcm_data, target_spacing_m=200, 
                                                             range_spacing_m_key="range_pixel_spacing_m", 
                                                             azimuth_spacing_m_key="azimuth_pixel_spacing_m")
        rcm_scaled_data = scale_hh_hv_sensor_geometry(rcm_data, target_spacing_m=target_spacing, 
                                                             range_spacing_m_key="range_pixel_spacing_m", 
                                                             azimuth_spacing_m_key="azimuth_pixel_spacing_m")
    return rcm_200m_data, rcm_scaled_data

def build_land_masks(rcm_200m_data):
    shp_path = resource_path("landmask/StatCan_ocean.shp")

    rcm_product = rcm_200m_data.copy()
    if rcm_product["geometry"] == "earth":
        land_mask = build_land_masks_earth_geometry(shp_path, rcm_product)
    else:
        land_mask = build_land_mask_sensor_geometry(
            rcm_product,
            shp_path,
            mask_res_m=200,      # 50–200 depending on memory
            simplify_m=0,        # no simplification
            threshold=0.5,
            chunk_rows=512
        )

    return land_mask

def normalize_and_prepare_images(rcm_200m_data, normalization_method="min-max"):
    hh = rcm_200m_data["hh"]
    hv = rcm_200m_data["hv"]

    # Helpful geocoding info for transforming pixel->lat/lon
    geo_coord_helpers = {"dst_transform": rcm_200m_data["src_transform"],
                  "dst_crs": rcm_200m_data["src_crs"],
                  "transformer": rcm_200m_data["transformer"]}

    nan_mask_hh = np.isnan(hh)
    nan_mask_hv = np.isnan(hv)

    if normalization_method == "min-max":
        # Min/Max normalization
        # Normalize HH band to uint8 for visualization
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
        min_ = hv[~nan_mask_hv].min(0)
        max_ = hv[~nan_mask_hv].max(0)
        hv_u8 = np.zeros_like(hv, dtype=np.uint8)

        if max_ > min_:
            hv_u8[~nan_mask_hv] = np.clip(
                255 * (hv[~nan_mask_hv] - min_) / (max_ - min_),
                0, 255
            ).astype(np.uint8)

        hv = hv_u8

    elif normalization_method == "mean-std":
        # Mean/Std normalization
        hh_mean = np.nanmean(hh)
        hh_std = np.nanstd(hh)
        hh_u8 = np.zeros_like(hh, dtype=np.uint8)

        hh_u8[~nan_mask_hh] = np.clip(
            255 * (hh[~nan_mask_hh] - hh_mean) / hh_std,
            0, 255
        ).astype(np.uint8)

        hh = hh_u8

        hv_mean = np.nanmean(hv)
        hv_std = np.nanstd(hv)
        hv_u8 = np.zeros_like(hv, dtype=np.uint8)

        hv_u8[~nan_mask_hv] = np.clip(
            255 * (hv[~nan_mask_hv] - hv_mean) / hv_std,
            0, 255
        ).astype(np.uint8)

        hv = hv_u8

    raw_img, img_base, hist, n_valid, nan_mask = setup_base_images(hh, hv, nan_mask_hh, nan_mask_hv)
    
    return raw_img, img_base, hist, n_valid, nan_mask, geo_coord_helpers


def run_pred_model(lbl_source, img, land_mask, model_path, target_width, target_height, 
                   target_spacing, model_spacing_m = 200, device="cpu"):
    hh = img["hh"]
    hv = img["hv"]
    valid_mask = np.isfinite(hh) & np.isfinite(hv)

    img_norm = Normalize_min_max(np.stack([hh, hv], axis=-1), valid_mask=valid_mask)

    img_norm_t = torch.from_numpy(img_norm[None]).permute(0, 3, 1, 2).to(device).float()

    colored_pred_map = forward_model_committee(
        model_path,
        img_norm_t,
        valid_mask=valid_mask,
        device=device,
    )

    # Scale colored_pred_map back to original image size if needed (e.g., if model runs on 200m but original is 100m)
    if target_spacing != model_spacing_m:
        new_size = (target_width, target_height)
        colored_pred_map = cv2.resize(colored_pred_map, new_size, interpolation=cv2.INTER_NEAREST)
        valid_mask = cv2.resize(valid_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST).astype(bool)

    colored_pred_map[land_mask] = [255, 255, 255]
    colored_pred_map[~valid_mask] = [255, 255, 255]

    land_nan_mask = (~valid_mask) | land_mask
    boundmask = generate_boundaries(rgb2gray(colored_pred_map))

    return [(lbl_source, colored_pred_map, land_nan_mask, boundmask)]


def build_land_masks_earth_geometry(shp_path: str, rcm_product: list[dict]) -> dict:
    """
    Build a land mask for an RCM product in earth geometry using a shapefile of ocean polygons.
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

    return land_mask

def build_land_mask_sensor_geometry(
    rcm_product: dict,
    shp_path: str,
    mask_res_m: float = 200,
    simplify_m: float = 0,
    threshold: float = 0.5,
    chunk_rows: int = 512,
):
    """
    Build a high-accuracy land mask for an RCM image in sensor geometry.
    Pipeline:
        1) Geocode SAR image to EPSG:3413 at chosen resolution.
        2) Rasterize ocean shapefile onto that projected grid.
        3) Resample ocean mask back to sensor grid using bilinear interpolation.
        4) Invert to obtain land mask.
    Parameters
    ----------
    rcm_product : dict
        Output from load_rcm_product (must contain hh + tie arrays).
    shp_path : str
        Path to ocean shapefile (polygons represent ocean).
    mask_res_m : float
        Intermediate projected raster resolution in meters.
        Smaller = sharper coastline but larger intermediate raster.
    simplify_m : float
        Optional coastline simplification tolerance (meters).
        Use 0 for maximum accuracy.
    threshold : float
        Threshold applied after bilinear resampling (default 0.5).
    chunk_rows : int
        Number of sensor rows processed per chunk to control memory usage.
    Returns
    -------
    land_mask : np.ndarray (bool)
        Boolean mask in sensor geometry (True = land, False = ocean).
    """
    def geocode_hh_to_3413_in_memory(rcm_product, pad_m=2000, target_res_m=200):
        """
        Robust: build dst grid from tie-point lon/lat bounds (projected to EPSG:3413),
        then reproject using GCPs (TPS-like mapping handled by GDAL internally).
        """
        hh = rcm_product["hh"].astype(np.float32, copy=False)
        H, W = hh.shape
        tie_r  = np.asarray(rcm_product["tie_lines"],  dtype=np.float64)
        tie_c  = np.asarray(rcm_product["tie_pixels"], dtype=np.float64)
        tie_la = np.asarray(rcm_product["tie_lats"],   dtype=np.float64)
        tie_lo = np.asarray(rcm_product["tie_lons"],   dtype=np.float64)
        # GCPs: X=lon, Y=lat
        gcps = [
            GroundControlPoint(row=float(r), col=float(c), x=float(lon), y=float(lat))
            for r, c, lat, lon in zip(tie_r, tie_c, tie_la, tie_lo)
        ]
        src_crs = CRS.from_epsg(4326)
        dst_crs = CRS.from_epsg(3413)
        # Use tie-point lon/lat bounds (more robust than asking GDAL to infer from GCPs)
        lon_min, lon_max = float(tie_lo.min()), float(tie_lo.max())
        lat_min, lat_max = float(tie_la.min()), float(tie_la.max())
        # Project bounds to EPSG:3413
        with Env(OGR_CT_FORCE_TRADITIONAL_GIS_ORDER="YES"):
            left, bottom, right, top = transform_bounds(
                src_crs, dst_crs, lon_min, lat_min, lon_max, lat_max, densify_pts=21
            )
        # Pad a bit to be safe
        left -= pad_m; right += pad_m
        bottom -= pad_m; top += pad_m
        # Target resolution (meters)
        xres = target_res_m
        yres = target_res_m
        dst_width  = int(np.ceil((right - left) / xres))
        dst_height = int(np.ceil((top - bottom) / yres))
        # North-up affine (y pixel size is negative)
        dst_transform = from_origin(left, top, xres, yres)
        dst = np.zeros((dst_height, dst_width), dtype=np.float32)
        with Env(OGR_CT_FORCE_TRADITIONAL_GIS_ORDER="YES"):
            reproject(
                source=hh,
                destination=dst,
                src_crs=src_crs,
                gcps=gcps,
                dst_crs=dst_crs,
                dst_transform=dst_transform,
                resampling=Resampling.bilinear,
            )
        return dst, dst_transform, dst_crs
    def ocean_mask_on_geocoded_grid_fast(ocean_shp_path, out_shape, dst_transform, dst_crs, simplify_m=0):
        # Read + reproject
        gdf = gpd.read_file(ocean_shp_path).to_crs(dst_crs)
        # Raster bounds in dst CRS
        H, W = out_shape
        left, top = (dst_transform.c, dst_transform.f)
        right = left + dst_transform.a * W
        bottom = top + dst_transform.e * H  # dst_transform.e is negative for north-up
        rbbox = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
        # Use spatial index to prefilter
        sidx = gdf.sindex
        hits = list(sidx.intersection(rbbox.bounds))
        gdf = gdf.iloc[hits].copy()
        # Precise clip to bounds
        gdf = gdf[gdf.intersects(rbbox)]
        if gdf.empty:
            return np.zeros(out_shape, dtype=bool)
        gdf["geometry"] = gdf.intersection(rbbox)
        # Optional: simplify coastline (meters, since CRS is EPSG:3413)
        if simplify_m and simplify_m > 0:
            gdf["geometry"] = gdf["geometry"].simplify(simplify_m, preserve_topology=True)
        # Dissolve/merge to reduce number of shapes
        geom = unary_union([g for g in gdf.geometry if g is not None and not g.is_empty])
        if geom.is_empty:
            return np.zeros(out_shape, dtype=bool)
        ocean = rasterize(
            [(geom, 1)],
            out_shape=out_shape,
            transform=dst_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        ).astype(bool)
        return ocean
    def ocean_mask_back_to_sensor_bilinear(
        ocean_mask_3413: np.ndarray,         # bool or uint8 (H3413, W3413)
        transform_3413,                      # Affine
        rcm_product: dict,
        thr: float = 0.5,
        chunk_rows: int = 256,
    ):
        """
        Build full-res ocean mask on sensor grid by:
        - lon/lat per sensor pixel (from regular tie grid)
        - lon/lat -> EPSG:3413
        - rowcol -> fractional pixel coords in ocean_mask_3413
        - manual bilinear sampling
        - threshold
        Returns ocean_sensor bool (H,W).
        """
        Hs, Ws = rcm_product["hh"].shape
        tie_r  = np.asarray(rcm_product["tie_lines"],  dtype=np.float64)
        tie_c  = np.asarray(rcm_product["tie_pixels"], dtype=np.float64)
        tie_la = np.asarray(rcm_product["tie_lats"],   dtype=np.float64)
        tie_lo = np.asarray(rcm_product["tie_lons"],   dtype=np.float64)
        # regular grid axes
        r_axis = np.unique(tie_r)
        c_axis = np.unique(tie_c)
        Ny, Nx = len(r_axis), len(c_axis)
        ir = np.searchsorted(r_axis, tie_r)
        ic = np.searchsorted(c_axis, tie_c)
        lon_grid = np.empty((Ny, Nx), dtype=np.float64); lon_grid[ir, ic] = tie_lo
        lat_grid = np.empty((Ny, Nx), dtype=np.float64); lat_grid[ir, ic] = tie_la
        # ocean mask as float in [0,1]
        src = ocean_mask_3413.astype(np.float32, copy=False)
        Hg, Wg = src.shape
        def interp_lonlat(rows, cols):
            rr = np.interp(rows, r_axis, np.arange(Ny))
            cc = np.interp(cols, c_axis, np.arange(Nx))
            r0 = np.floor(rr).astype(int); r1 = np.clip(r0 + 1, 0, Ny - 1)
            c0 = np.floor(cc).astype(int); c1 = np.clip(c0 + 1, 0, Nx - 1)
            dr = (rr - r0)[:, None]
            dc = (cc - c0)[None, :]
            lon00 = lon_grid[r0[:, None], c0[None, :]]
            lon01 = lon_grid[r0[:, None], c1[None, :]]
            lon10 = lon_grid[r1[:, None], c0[None, :]]
            lon11 = lon_grid[r1[:, None], c1[None, :]]
            lat00 = lat_grid[r0[:, None], c0[None, :]]
            lat01 = lat_grid[r0[:, None], c1[None, :]]
            lat10 = lat_grid[r1[:, None], c0[None, :]]
            lat11 = lat_grid[r1[:, None], c1[None, :]]
            lon0 = lon00 * (1 - dc) + lon01 * dc
            lon1 = lon10 * (1 - dc) + lon11 * dc
            lon  = lon0 * (1 - dr) + lon1 * dr
            lat0 = lat00 * (1 - dc) + lat01 * dc
            lat1 = lat10 * (1 - dc) + lat11 * dc
            lat  = lat0 * (1 - dr) + lat1 * dr
            return lon, lat
        ocean_sensor = np.zeros((Hs, Ws), dtype=bool)
        cols = np.arange(Ws, dtype=np.float64)
        for r_start in range(0, Hs, chunk_rows):
            r_end = min(Hs, r_start + chunk_rows)
            rows = np.arange(r_start, r_end, dtype=np.float64)
            # (R,C) lon/lat for this chunk
            lon, lat = interp_lonlat(rows, cols)
            # lon/lat -> EPSG:3413 coords
            xs, ys = crs_transform("EPSG:4326", "EPSG:3413",
                                lon.ravel().tolist(), lat.ravel().tolist())
            xs = np.asarray(xs); ys = np.asarray(ys)
            # fractional (row, col) in geocoded raster
            rr, cc = rasterio.transform.rowcol(transform_3413, xs, ys, op=float)
            rr = np.asarray(rr); cc = np.asarray(cc)
            # bilinear sampling
            r0 = np.floor(rr).astype(np.int64)
            c0 = np.floor(cc).astype(np.int64)
            r1 = r0 + 1
            c1 = c0 + 1
            # valid where all neighbors are inside
            valid = (r0 >= 0) & (c0 >= 0) & (r1 < Hg) & (c1 < Wg)
            out = np.zeros(rr.shape, dtype=np.float32)
            if np.any(valid):
                dr = (rr[valid] - r0[valid]).astype(np.float32)
                dc = (cc[valid] - c0[valid]).astype(np.float32)
                v00 = src[r0[valid], c0[valid]]
                v01 = src[r0[valid], c1[valid]]
                v10 = src[r1[valid], c0[valid]]
                v11 = src[r1[valid], c1[valid]]
                out[valid] = (
                    v00 * (1 - dr) * (1 - dc) +
                    v01 * (1 - dr) * dc +
                    v10 * dr * (1 - dc) +
                    v11 * dr * dc
                )
            ocean_sensor[r_start:r_end, :] = (out.reshape(r_end - r_start, Ws) > thr)
        return ocean_sensor
    # ------------------------------------------------------------------
    # STEP 1 — Geocode SAR band to projected CRS (EPSG:3413)
    # ------------------------------------------------------------------
    # This creates an intermediate raster in map geometry
    # at the requested resolution (mask_res_m).
    # The result is used only to rasterize ocean polygons cleanly.
    # ------------------------------------------------------------------
    hh_3413, tr_3413, crs_3413 = geocode_hh_to_3413_in_memory(
        rcm_product,
        target_res_m=mask_res_m
    )
    # ------------------------------------------------------------------
    # STEP 2 — Rasterize ocean shapefile on projected grid
    # ------------------------------------------------------------------
    # - Reproject shapefile to EPSG:3413
    # - Clip to raster bounds
    # - Optionally simplify coastline
    # - Rasterize into binary ocean mask
    #
    # Result: ocean_3413 (True = ocean)
    # ------------------------------------------------------------------
    ocean_3413 = ocean_mask_on_geocoded_grid_fast(
        shp_path,
        out_shape=hh_3413.shape,
        dst_transform=tr_3413,
        dst_crs=crs_3413,
        simplify_m=simplify_m,
    )
    # ------------------------------------------------------------------
    # STEP 3 — Resample ocean mask back to sensor geometry
    # ------------------------------------------------------------------
    # For each sensor pixel:
    #   - Interpolate lon/lat from tie-point grid
    #   - Project to EPSG:3413
    #   - Sample ocean_3413 using bilinear interpolation
    #   - Apply threshold to obtain boolean mask
    #
    # chunk_rows controls memory usage.
    # ------------------------------------------------------------------
    ocean_sensor = ocean_mask_back_to_sensor_bilinear(
        ocean_3413,
        tr_3413,
        rcm_product,
        thr=threshold,
        chunk_rows=chunk_rows,
    )
    # ------------------------------------------------------------------
    # STEP 4 — Convert ocean mask to land mask
    # ------------------------------------------------------------------
    # Shapefile represents ocean polygons (ocean=True),
    # so land is simply the inverse.
    # ------------------------------------------------------------------
    land_mask = ~ocean_sensor
    return land_mask
