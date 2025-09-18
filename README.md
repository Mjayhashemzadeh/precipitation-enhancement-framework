# Precipitation Enhancement Framework (Sample Repository)

This repository provides **sample data (2024 only)**, **Google Earth Engine scripts**, 
and **machine learning pipelines** supporting the paper:

> Hashemzadeh et al. (2025)  
> "A Comprehensive Framework for Enhancing Satellite-Based Precipitation Estimates Using Attention-Based Deep Learning"

---

## Contents

- **`data/`**  
  Sample 2024 cleaned datasets (Excel format), one file per feature:
  - `ground_precipitation_2024.xlsx` – Ground station observations
  - `era5_precip_2024_1km.xlsx` – ERA5-Land precipitation
  - `era5_sm_2024_1km.xlsx` – Soil moisture
  - `ndvi_2024_1km.xlsx` – MODIS NDVI
  - `ndwi_2024_1km.xlsx` – MODIS NDWI
  - `era5_Tmin_2024_1km.xlsx` – Minimum temperature
  - `modis_cloud_cover_2024_1km.xlsx` – Cloud Cover
  - `station_coordinates.xlsx` – Elevation & Coordinates

- **`gee_scripts/`**  
  Example script (`extract_precip_points.js`) to extract ERA5 and TerraClimate 
  precipitation values at station locations, resampled to 1 km.

- **`scripts/`**  
  - `XGBoost_Training_Pipeline.py` – Train XGBoost model  
  - `XGBoost_Parameter_tunning.py` – Hyperparameter tuning for XGBoost  
  - `TabNet_Training_Pipeline.py` – Train TabNet model  
  - `TabNet_Training_Temporal.py` – Demonstrates temporal holdout validation  
  - `TabNet_Raster_Prediction.py` – Apply TabNet to raster inputs (demo)

---

## Temporal Holdout Notes

- **Study setup:**  
  Train = 2014–2022, Test = 2023–2024  

- **Sample repo:**  
  Only 2024 data are provided. The script detects this and **falls back** to 
  splitting 2024 into two halves:  
  - Train = Jan–Jun 2024  
  - Test = Jul–Dec 2024  

This ensures the script runs correctly even with limited data.

---

## Raster Data

- **Raster Inputs (2024 only):**
  Included in `/data/raster_inputs/` are sample input rasters at 1 km resolution:
  - ERA5Land precipitation, runoff, soil moisture, tmin, cloud cover
  - MODIS NDVI, MODIS NDWI
  - SRTM DEM

  These rasters are provided only for 2024 to demonstrate data structure and reproducibility.

- **Raster Predictions:**
  While the paper describes predicted precipitation rasters for 2014–2024 (both TabNet and XGBoost),
  we include here only the **132 TabNet prediction rasters** (monthly, 2014–2024), packaged as a 
  single ZIP file (~25 MB). This provides a complete set of model outputs in a lightweight format.
  
  XGBoost rasters are not included due to size, but can be reproduced using the provided scripts.
  ---
  
## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
