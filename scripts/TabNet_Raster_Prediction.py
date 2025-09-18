import os
import numpy as np
import rasterio
import pickle
import pandas as pd
from datetime import datetime
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

class PrecipitationPredictionPipeline:
    def __init__(self, model_dir, raster_input_dir):
        self.model_dir = model_dir
        self.raster_input_dir = raster_input_dir
        self.output_dir = os.path.join(raster_input_dir, 'raster_predictions')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.bias_correction_params = None
        
        self.load_model_and_params()
    
    def load_model_and_params(self):
        try:
            model_path = os.path.join(self.model_dir, 'tabnet_model.zip')
            self.model = TabNetRegressor()
            self.model.load_model(model_path)
            print(f"TabNet model loaded successfully from {model_path}")
            
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded successfully")
            
            try:
                with open(os.path.join(self.model_dir, 'bias_correction_params.pkl'), 'rb') as f:
                    self.bias_correction_params = pickle.load(f)
                print(f"Bias correction parameters loaded successfully")
            except FileNotFoundError:
                print("Bias correction parameters not found, using default")
                self.bias_correction_params = {}
            
            try:
                self.feature_importance = pd.read_csv(os.path.join(self.model_dir, 'feature_importance.csv'))
                print(f"Feature importance loaded successfully")
                print(f"Features according to feature importance file: {self.feature_importance['Feature'].tolist()}")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print("Feature importance file not found or empty")
            
        except Exception as e:
            print(f"Error loading model or parameters: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def apply_bias_correction(self, raster_data):
        if not self.bias_correction_params:
            return raster_data
        
        if 'ground_quantiles' in self.bias_correction_params:
            ground_quantiles = self.bias_correction_params['ground_quantiles']
            remote_quantiles = self.bias_correction_params['remote_quantiles']
            
            corrected_raster = np.zeros_like(raster_data)
            mask = ~np.isnan(raster_data)
            
            values = raster_data[mask]
            corrected_values = np.zeros_like(values)
            
            for i, val in enumerate(values):
                if val <= remote_quantiles[0]:
                    corrected_values[i] = ground_quantiles[0]
                elif val >= remote_quantiles[-1]:
                    corrected_values[i] = ground_quantiles[-1]
                else:
                    idx = np.where(remote_quantiles <= val)[0][-1]
                    next_idx = idx + 1
                    weight = (val - remote_quantiles[idx]) / (remote_quantiles[next_idx] - remote_quantiles[idx])
                    corrected_values[i] = ground_quantiles[idx] + weight * (ground_quantiles[next_idx] - ground_quantiles[idx])
            
            corrected_raster[mask] = corrected_values
            return corrected_raster
            
        elif 'scaling_factor' in self.bias_correction_params:
            scaling_factor = self.bias_correction_params['scaling_factor']
            return raster_data * scaling_factor
        
        return raster_data
    
    def extract_lat_lon_rasters(self, reference_raster_path):
        with rasterio.open(reference_raster_path) as src:
            height, width = src.shape
            transform = src.transform
            
            rows, cols = np.indices((height, width))
            
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
            lon_raster = np.array(xs).reshape(height, width)
            lat_raster = np.array(ys).reshape(height, width)
            
            return lat_raster, lon_raster
    
    def load_multiband_data(self, year):
        precip_path = os.path.join(self.raster_input_dir, f"ERA5Land_precip_AllMonths_{year}.tif")
        sm_path = os.path.join(self.raster_input_dir, f"ERA5Land_sm_AllMonths_{year}.tif")
        ndvi_path = os.path.join(self.raster_input_dir, f"MODIS_NDVI_AllMonths_{year}.tif")
        ndwi_path = os.path.join(self.raster_input_dir, f"MODIS_NDWI_AllMonths_{year}.tif")
        runoff_path = os.path.join(self.raster_input_dir, f"ERA5Land_runoff_AllMonths_{year}.tif")
        cloudcover_path = os.path.join(self.raster_input_dir, f"MODIS_CloudCover_AllMonths_{year}.tif")
        tmin_path = os.path.join(self.raster_input_dir, f"ERA5Land_tmin_AllMonths_{year}.tif")
        dem_path = os.path.join(self.raster_input_dir, "SRTM_DEM_1km.tif")
        
        data_dict = {}
        
        with rasterio.open(precip_path) as src:
            data_dict['precip'] = src.read()
            profile = src.profile.copy()
        
        with rasterio.open(sm_path) as src:
            data_dict['sm'] = src.read()
        
        with rasterio.open(ndvi_path) as src:
            data_dict['ndvi'] = src.read()
        
        with rasterio.open(ndwi_path) as src:
            data_dict['ndwi'] = src.read()
            
        with rasterio.open(runoff_path) as src:
            data_dict['runoff'] = src.read()
            
        with rasterio.open(cloudcover_path) as src:
            data_dict['cloud_cover'] = src.read()
            
        with rasterio.open(tmin_path) as src:
            data_dict['t_min'] = src.read()
        
        with rasterio.open(dem_path) as src:
            data_dict['dem'] = src.read(1)
            
        lat_raster, lon_raster = self.extract_lat_lon_rasters(dem_path)
        data_dict['lat'] = lat_raster
        data_dict['lon'] = lon_raster
        data_dict['profile'] = profile
        
        return data_dict
    
    def predict_for_month(self, year, month, data_dict):
        month_str = f"{month:02d}"
        print(f"Processing data for {year}-{month_str}")
    
        try:
            band_index = month - 1
            
            precip_data = data_dict['precip'][band_index]
            sm_data = data_dict['sm'][band_index]
            ndvi_data = data_dict['ndvi'][band_index]
            ndwi_data = data_dict['ndwi'][band_index]
            runoff_data = data_dict['runoff'][band_index]
            cloudcover_data = data_dict['cloud_cover'][band_index]
            tmin_data = data_dict['t_min'][band_index]
            dem_data = data_dict['dem']
            lat_raster = data_dict['lat']
            lon_raster = data_dict['lon']
            profile = data_dict['profile']
        
            corrected_precip = self.apply_bias_correction(precip_data)
        
            sin_month = np.sin(2 * np.pi * month / 12)
            cos_month = np.cos(2 * np.pi * month / 12)
        
            height, width = precip_data.shape
            valid_mask = (~np.isnan(precip_data) & 
                         ~np.isnan(sm_data) & 
                         ~np.isnan(ndvi_data) & 
                         ~np.isnan(ndwi_data) & 
                         ~np.isnan(runoff_data) & 
                         ~np.isnan(cloudcover_data) & 
                         ~np.isnan(tmin_data) & 
                         ~np.isnan(dem_data) &
                         ~np.isnan(lat_raster) &
                         ~np.isnan(lon_raster))
        
            valid_indices = np.where(valid_mask)
        
            feature_order = ['precipitation_remote_corrected', 'ndvi', 'ndwi', 'sm', 'elevation', 'runoff', 
                           'cloud_cover', 't_min', 'latitude', 
                           'longitude', 'sin_month', 'cos_month']
            
            print(f"Creating features in the expected order: {feature_order}")
        
            feature_dict = {
                'precipitation_remote_corrected': corrected_precip[valid_indices],
                'ndvi': ndvi_data[valid_indices],
                'ndwi': ndwi_data[valid_indices],
                'runoff': runoff_data[valid_indices],
                'cloud_cover': cloudcover_data[valid_indices],
                't_min': tmin_data[valid_indices],
                'sm': sm_data[valid_indices],
                'elevation': dem_data[valid_indices],
                'latitude': lat_raster[valid_indices],
                'longitude': lon_raster[valid_indices],
                'sin_month': sin_month * np.ones_like(precip_data[valid_indices]),
                'cos_month': cos_month * np.ones_like(precip_data[valid_indices])
            }
        
            feature_df = pd.DataFrame({name: feature_dict[name] for name in feature_order})
        
            scaled_features = self.scaler.transform(feature_df)
        
            predictions = self.model.predict(scaled_features)
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            predictions = np.maximum(predictions, 0)
            
            output_raster = np.full_like(precip_data, np.nan)
            output_raster[valid_indices] = predictions
            
            output_path = os.path.join(self.output_dir, f"PrecipPrediction_{year}_{month_str}.tif")
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.float32,
                crs=profile['crs'],
                transform=profile['transform'],
            ) as dst:
                dst.write(output_raster.astype(rasterio.float32), 1)
            
            print(f"Successfully saved prediction to {output_path}")
            
            return output_path, output_raster
            
        except Exception as e:
            print(f"Error processing data for {year}-{month_str}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def process_all_months(self, year):
        print(f"Loading multiband data for year {year}")
        data_dict = self.load_multiband_data(year)
        
        results = {}
        monthly_predictions = []
        
        for month in range(1, 13):
            output_path, output_raster = self.predict_for_month(year, month, data_dict)
            results[month] = output_path
            if output_raster is not None:
                monthly_predictions.append(output_raster)
                
        return results

if __name__ == "__main__":
    model_dir = "TabNet"
    raster_dir = "rasters"
    
    pipeline = PrecipitationPredictionPipeline(model_dir, raster_dir)
    
    results = pipeline.process_all_months(2024)
    
    prediction_dir = os.path.join(raster_dir, 'raster_predictions')