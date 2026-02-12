# ArcticScope

ArcticScope is a Windows desktop application based in Python designed to help users review and correct model predictions on Synthetic Aperture Radar (SAR) imagery of sea ice.

The application allows you to:
- View SAR images in HH and HV polarization bands, with options for colour composite views (HH, HH, HV) and (HH, HV, HV)
- Adjust image brightness, contrast, and prediction overlay opacity
- Run a prediction model on the image
- Draw polygon annotations to highlight and label regions where the model prediction is incorrect


> **Key idea:** quickly review model predictions and generate high-quality correction labels for retraining and evaluation.

---

## Features

- **SAR visualization**
  - Toggle between **HH** and **HV** bands, including colour composite options **(HH, HH, HV)** and **(HH, HV, HV)**
  - **Brightness + contrast** controls (interactive)
  - **Zoom + pan**
  - **Latitude + longitude** on hover
  - **Minimap** to help orientation while zooming + panning

- **Model inference**
  - Runs a **prediction model** on the SAR image input
  - Displays a **prediction overlay** on top of the imagery
  - Adjustable overlay **opacity**

- **Annotation**
  - Draw polygons to mark regions the model mislabeled
  - **Local segmentation** option on HH or HV band to help form natural polygons
  - Labels include: **Ice, Water, Shoal, Ship, Iceberg, Unknown**
  - **Save/edit** annotation outputs (format described below)
  - Minimap shows edited areas

- **Windows executable**
  - App can be run as a standalone `.exe`
  - `.exe` is built using a provided `.bat` script in this repository

---

## User Manual

A user manual is included in this repo for a rundown on how to use the app.

`ArcticScope_User_Manual.pdf`

---

## Quick Start (Run from Source)

### Requirements
- Windows 10/11 (recommended)
- Python 3.10+
- Recommended screen resolution: 1920 × 1080 or higher
- Recommended memory: 32 GB or higher

### Setup
```bash
git clone https://github.com/jnoat92/RS-visualizer.git
cd <PATH_TO_REPO>

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Ensure that you have access to *magic_lib***

### Run
``` bash
python -m main
```

### Basic App Startup
1. Launch the app
2. Select a SAR scene (folder with .img and .xml files)
3. The app displays image and model prediction overlay

### Functionalities

- Choose Band / Colour Composite: **HH, HV, (HH, HH, HV), (HH, HV, HV)**
- Brightness/Contrast: adjust image intensity
- Overlay Opacity: adjust prediction visibility
- Annotate: draw polygons on the overlay to mark incorrect predictions

### Annotation Workflow

1. Run inference for a scene (set up to run automatically)
2. Visually inspect overlay vs image
3. Draw and label polygons over mislabeled regions, can also use included unsupervised segmentation to generate polygons based on the image
4. Save annotations for review/training

**For more information please refer to the user manual**

### Data Formats
#### Supported Inputs

SAR image formats: .img

Supporting files: .xml, .txt

### Annotation Output Format

Output directory: Custom_Annotation/<scene_id>

Files:

custom_annotation.png - coloured mask composed of polygons drawn by the user

annotation_notes.json - consists of scene_id, timestamp, and notes

annotated_area.npz - numpy array consisting of pixel coordinates of annotated areas (for minimap use)

Example (annotation_notes structure):
```json
{
  "<scene_id>": {
    "timestamp": "2026-02-10 16:41:05",
    "notes": "Minor annotations for water near landmask"
  },
  "<scene_id>": {
    "timestamp": "2026-02-11 10:28:07",
    "notes": "Misidentified ice"
  }
}
```

## Building the Windows Executable (.exe)

This repo includes a .bat script that builds a standalone Windows executable.

### Build Steps

Ensure dependencies are installed in your environment:

```bash
pip install -r requirements.txt
```

Run the build script:

```bash
RS-visualizer\create_executable.bat
```

#### Build Output

The generated executable will be placed in:

`ArcticScope-executable\ArcticScope`

## Notes

### Project Structure

```
├───app
│   ├───state.py
│   └───visualizer.py
├───core
│   ├───contrast_handler.py
│   ├───io.py
│   ├───overlay.py
│   ├───parallel_handler.py
│   ├───render.py
│   ├───segmentation.py
│   └───utils.py
├───icons
│   ├───polygon.png
│   └───rectangle.png
├───landmask
│   ├───StatCan_ocean.dbf
│   ├───StatCan_ocean.prj
│   ├───StatCan_ocean.shp
│   └───StatCan_ocean.shx
├───model
│   ├───DL
│   │   └───unet
│   │       ├───__init__.py
│   │       ├───unet_model.py
│   │       └───unet_parts.py
│   ├───model_helper.py
│   └───Unet_model_12_.pt
├───ui
│   ├───annotation.py
│   ├───evaluation.py
│   └───minimap.py
├───ArcticScope_User_Manual.pdf
├───create_executable.bat
├───main.py
├───README.md
└───requirements.txt
```

### Troubleshooting

Image Does Not Load
- Verify the file format is supported.
- Ensure all required files are present.

Overlay Not Visible
- Increase overlay opacity.
- Check if Ice/Water Labels is ON
- Confirm the model prediction completed successfully.

Application Feels Slow
- Zoom in on a smaller area
- Close other applications if system memory is limited.

Last modified: Feb 12, 2026
