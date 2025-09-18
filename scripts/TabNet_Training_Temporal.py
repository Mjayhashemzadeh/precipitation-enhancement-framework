import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch

class TemporalPrecipitationModelingPipeline:
    def __init__(self, config=None):
        self.config = config or {
            'file_mapping': {
                'ground': 'ground_precipitation_2024.xlsx',
                'remote': {
                    'CHIRPS': 'chrips-precip-monthly.xlsx',
                    'ERA5': 'era5_precip_2024_1km.xlsx',
                    'Terraclimate': 'terra-precip-monthly.xlsx'
                },
                'features': {
                    'NDVI': 'ndvi_2024_1km.xlsx',
                    'NDWI': 'ndwi_2024_1km.xlsx',
                    'RUNOFF': 'era5_runoff_2024_1km.xlsx',
                    'CC': 'modis_cloud_cover_2024_1km.xlsx',
                    'TMIN': 'era5_Tmin_2024_1km.xlsx',
                    'SM': 'era5_sm_2024_1km.xlsx',
                    'DEM': 'station_coordinates.xlsx'
                },
                'coords': 'station_coordinates.xlsx'
            },
            'output_dir': 'temporal_output',
            'target_col': 'precipitation_ground',
            'train_years': list(range(2014, 2023)),
            'test_years': [2023, 2024],
            'scaling': True,
            'include_seasonality': True,
            'apply_bias_correction': True,
            'bias_correction_method': 'quantile_mapping',
            'tabnet_params': {
                'n_d': 32,
                'n_a': 32,
                'n_steps': 3,
                'gamma': 1.3,
                'lambda_sparse': 1e-3,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': {'lr': 2e-2},
                'mask_type': 'sparsemax',
                'scheduler_params': {'step_size': 10, 'gamma': 0.9},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'seed': 42,
                'verbose': 1
            },
            'tabnet_fit_params': {
                'max_epochs': 200,
                'patience': 50,
                'batch_size': 1024,
                'virtual_batch_size': 128,
                'num_workers': 0,
                'drop_last': False
            }
        }
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.log_file = os.path.join(self.config['output_dir'], f"temporal_model_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        self.emoji_map = {
            '📊': '[DATA]',
            '✓': '[OK]',
            '⚠️': '[WARNING]',
            '❌': '[ERROR]',
            '🔍': '[PROCESSING]',
            '🚀': '[STARTING]',
            '✅': '[COMPLETE]',
            '⏰': '[TIME]'
        }
        
        self.scaler = None
        self.bias_correction_params = {}
        
    def log(self, message):
        print(message)
        safe_message = message
        for emoji, text in self.emoji_map.items():
            safe_message = safe_message.replace(emoji, text)
            
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{safe_message}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file: {str(e)}")
            
    def load_and_merge_data(self, var_name='precipitation', remote_source='ERA5'):
        self.log(f"📊 Loading data from {remote_source}...")
        
        try:
            ground = pd.read_excel(self.config['file_mapping']['ground'])
            remote = pd.read_excel(self.config['file_mapping']['remote'][remote_source])
            
            ground.columns = ground.columns.str.strip().str.lower()
            remote.columns = remote.columns.str.strip().str.lower()
    
            for df in [ground, remote]:
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df['year'] = df['date'].dt.year
                    df['month'] = df['date'].dt.month
                elif not all(col in df.columns for col in ['year', 'month']):
                    self.log("⚠️ Error: Missing temporal information (date or year/month)")
                    return None
    
            df = pd.merge(
                ground[['station', 'year', 'month', var_name]],
                remote[['station', 'year', 'month', var_name]],
                on=['station', 'year', 'month'],
                suffixes=('_ground', '_remote')
            )
            
            self.log(f"✓ Merged ground and remote data: {len(df)} records")
    
            for key, path in self.config['file_mapping']['features'].items():
                try:
                    feat_df = pd.read_excel(path)
                    feat_df.columns = feat_df.columns.str.strip().str.lower()
    
                    if 'date' in feat_df.columns:
                        feat_df['date'] = pd.to_datetime(feat_df['date'])
                        feat_df['year'] = feat_df['date'].dt.year
                        feat_df['month'] = feat_df['date'].dt.month
                    
                    if 'date' in feat_df.columns:
                        feat_df = feat_df.drop(columns=['date'])
    
                    if all(col in feat_df.columns for col in ['station', 'year', 'month']):
                        pre_merge_len = len(df)
                        df = df.merge(feat_df, on=['station', 'year', 'month'], how='left')
                        post_merge_len = len(df)
                        
                        if pre_merge_len != post_merge_len:
                            self.log(f"⚠️ Warning: Merge with {key} changed record count from {pre_merge_len} to {post_merge_len}")
                        else:
                            self.log(f"✓ Added {key} feature")
                    else:
                        self.log(f"⚠️ Skipping {key}: missing 'station', 'year', or 'month'")
                except Exception as e:
                    self.log(f"❌ Error loading {key}: {str(e)}")
    
            try:
                coords = pd.read_excel(self.config['file_mapping']['coords'])
                coords.columns = coords.columns.str.strip().str.lower()
                
                coord_cols = ['latitude', 'longitude', 'elevation', 'distance_to_coast']
                available_cols = [col for col in coord_cols if col in coords.columns]
                
                if available_cols:
                    coords_subset = coords[['station'] + available_cols]
                    df = df.merge(coords_subset, on='station', how='left')
                    self.log(f"✓ Added coordinate data: {', '.join(available_cols)}")
            except Exception as e:
                self.log(f"❌ Error loading coordinates: {str(e)}")
                
            if self.config.get('include_seasonality', False):
                df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
                df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
                self.log("✓ Added seasonality features")
            
            missing_data = df.isnull().sum()
            if missing_data.any():
                self.log("⚠️ Missing values detected:")
                for col, count in missing_data[missing_data > 0].items():
                    self.log(f"   - {col}: {count} missing values ({count/len(df)*100:.1f}%)")
            
            return df
            
        except Exception as e:
            self.log(f"❌ Data loading failed: {str(e)}")
            return None
    
    def compute_bias_correction_params(self, train_df, target_col='precipitation_ground', remote_col='precipitation_remote'):
        if not self.config.get('apply_bias_correction', False):
            return
        
        method = self.config.get('bias_correction_method', 'quantile_mapping')
        self.log(f"🔍 Computing bias correction parameters using {method} on training data only...")
        
        valid_mask = ~(train_df[target_col].isna() | train_df[remote_col].isna())
        train_df_clean = train_df[valid_mask]
        
        if method == 'quantile_mapping':
            ground_values = train_df_clean[target_col].values
            remote_values = train_df_clean[remote_col].values
            
            ground_quantiles = np.percentile(ground_values, np.arange(0, 101, 5))
            remote_quantiles = np.percentile(remote_values, np.arange(0, 101, 5))
            
            self.bias_correction_params = {
                'method': 'quantile_mapping',
                'ground_quantiles': ground_quantiles,
                'remote_quantiles': remote_quantiles
            }
            
        elif method == 'linear_scaling':
            ground_mean = train_df_clean[target_col].mean()
            remote_mean = train_df_clean[remote_col].mean()
            
            scaling_factor = ground_mean / remote_mean
            self.bias_correction_params = {
                'method': 'linear_scaling',
                'scaling_factor': scaling_factor
            }
        
        else:
            self.log(f"⚠️ Unknown bias correction method: {method}")
            self.bias_correction_params = {}
        
        self.log(f"✓ Bias correction parameters computed from {len(train_df_clean)} training samples")
    
    def apply_bias_correction(self, df, remote_col='precipitation_remote'):
        if not self.bias_correction_params or not self.config.get('apply_bias_correction', False):
            return df.copy()
        
        df_corrected = df.copy()
        method = self.bias_correction_params.get('method')
        
        if method == 'quantile_mapping':
            ground_quantiles = self.bias_correction_params['ground_quantiles']
            remote_quantiles = self.bias_correction_params['remote_quantiles']
            
            remote_values = df[remote_col].values
            corrected_values = np.zeros_like(remote_values)
            
            for i, val in enumerate(remote_values):
                if np.isnan(val):
                    corrected_values[i] = np.nan
                elif val <= remote_quantiles[0]:
                    corrected_values[i] = ground_quantiles[0]
                elif val >= remote_quantiles[-1]:
                    corrected_values[i] = ground_quantiles[-1]
                else:
                    idx = np.where(remote_quantiles <= val)[0][-1]
                    next_idx = idx + 1
                    weight = (val - remote_quantiles[idx]) / (remote_quantiles[next_idx] - remote_quantiles[idx])
                    corrected_values[i] = ground_quantiles[idx] + weight * (ground_quantiles[next_idx] - ground_quantiles[idx])
            
            df_corrected[f'{remote_col}_corrected'] = corrected_values
            
        elif method == 'linear_scaling':
            scaling_factor = self.bias_correction_params['scaling_factor']
            df_corrected[f'{remote_col}_corrected'] = df[remote_col] * scaling_factor
        
        return df_corrected

    # Updated temporal_split_data Function
    def temporal_split_data(self, df):
        train_years = self.config['train_years']
        test_years = self.config['test_years']
        
        # Primary split (study setup: 2014–2022 vs 2023–2024)
        train_df = df[df['year'].isin(train_years)].copy()
        test_df = df[df['year'].isin(test_years)].copy()
        
        # Fallback for sample dataset (only 2024 available)
        if train_df.empty:
            self.log("⚠️ No training data found for configured years. Using fallback split for sample dataset.")
            
            # Sort by year + month to simulate chronology
            df_sorted = df.sort_values(["year", "month"]).reset_index(drop=True)
            
            cutoff = len(df_sorted) // 2  # first half = train, second half = test
            train_df = df_sorted.iloc[:cutoff].copy()
            test_df = df_sorted.iloc[cutoff:].copy()
            
            train_years = [train_df["year"].min()]
            test_years = [test_df["year"].max()]
        
        self.log(f"📊 Training data: {len(train_df)} records ({min(train_years)}-{max(train_years)})")
        self.log(f"📊 Test data: {len(test_df)} records ({min(test_years)}-{max(test_years)})")
        
        train_stations = set(train_df["station"].unique())
        test_stations = set(test_df["station"].unique())
        common_stations = train_stations.intersection(test_stations)
        
        self.log(f"📊 Common stations in both train and test: {len(common_stations)}")
        
        return train_df, test_df

    
    def preprocess_features(self, df, is_training=False):
        if df is None or df.empty:
            self.log("❌ No data to preprocess")
            return None, None, None
        
        remote_col = 'precipitation_remote_corrected' if 'precipitation_remote_corrected' in df.columns else 'precipitation_remote'
        
        all_features = [remote_col, 'ndvi', 'ndwi', 'sm', 'elevation', 'runoff', 'cloud_cover', 't_min', 'latitude', 'longitude', 'month']
        
        if self.config.get('include_seasonality', False):
            seasonal_cols = ['sin_month', 'cos_month']
            if all(col in df.columns for col in seasonal_cols):
                all_features.extend(seasonal_cols)
                if 'month' in all_features:
                    all_features.remove('month')
        
        features = [f for f in all_features if f in df.columns]
        target_col = self.config['target_col']
        
        if is_training:
            self.log(f"🔍 Selected features: {', '.join(features)}")
        
        pre_dropna_len = len(df)
        df_clean = df.dropna(subset=features + [target_col])
        post_dropna_len = len(df_clean)
        
        if pre_dropna_len != post_dropna_len:
            self.log(f"⚠️ Removed {pre_dropna_len - post_dropna_len} rows with missing values")
        
        X = df_clean[features]
        y = df_clean[target_col]
        
        return X, y, df_clean['station'], df_clean
    
    def evaluate_model(self, y_true, y_pred):
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred)
        }
    
    def train_and_evaluate_temporal(self, train_df, test_df):
        self.compute_bias_correction_params(train_df)
        
        train_df_corrected = self.apply_bias_correction(train_df)
        test_df_corrected = self.apply_bias_correction(test_df)
        
        X_train, y_train, stations_train, _ = self.preprocess_features(train_df_corrected, is_training=True)
        if X_train is None:
            self.log("❌ No training data available")
            return None
        
        if self.config.get('scaling', False):
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            self.log("✓ Applied StandardScaler to training features")
        
        X_test, y_test, stations_test, test_df_clean = self.preprocess_features(test_df_corrected)
        if X_test is None:
            self.log("❌ No test data available")
            return None
        
        if self.scaler:
            X_test_scaled = self.scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            self.log("✓ Applied StandardScaler to test features")
        
        tabnet_params = self.config.get('tabnet_params', {})
        fit_params = self.config.get('tabnet_fit_params', {})
        
        self.log(f"🚀 Training TabNet model on {min(self.config['train_years'])}-{max(self.config['train_years'])} data...")
        
        model = TabNetRegressor(**tabnet_params)
        
        X_train_final = X_train.values.astype(np.float32)
        y_train_final = y_train.values.reshape(-1, 1).astype(np.float32)
        X_test_final = X_test.values.astype(np.float32)
        y_test_final = y_test.values.reshape(-1, 1).astype(np.float32)
        
        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_test_final, y_test_final)],
            **fit_params
        )
        
        y_pred_train = model.predict(X_train_final).flatten()
        train_metrics = self.evaluate_model(y_train, y_pred_train)
        
        self.log(f"📊 Temporal Training Results:")
        for metric, value in train_metrics.items():
            self.log(f"   - {metric}: {value:.4f}")
        
        self.log(f"🔍 Evaluating model on {min(self.config['test_years'])}-{max(self.config['test_years'])} data...")
        
        y_pred = model.predict(X_test_final).flatten()
        test_metrics = self.evaluate_model(y_test, y_pred)
        
        self.log(f"📊 Temporal Test Results:")
        for metric, value in test_metrics.items():
            self.log(f"   - {metric}: {value:.4f}")
        
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.log("📈 Feature Importance:")
        for _, row in feature_importance.iterrows():
            self.log(f"   - {row['Feature']}: {row['Importance']:.4f}")
        
        station_results = []
        for station in stations_test.unique():
            station_mask = stations_test == station
            if station_mask.sum() > 0:
                station_y_true = y_test[station_mask]
                station_y_pred = y_pred[station_mask]
                station_metrics = self.evaluate_model(station_y_true, station_y_pred)
                station_results.append({
                    'station': station,
                    'records': station_mask.sum(),
                    **station_metrics
                })
        
        station_df = pd.DataFrame(station_results)
        
        predictions_df = test_df_clean.copy()
        predictions_df = predictions_df.reset_index(drop=True)
        predictions_df['predicted'] = y_pred
        predictions_df['actual'] = y_test.values
        predictions_df['residual'] = predictions_df['actual'] - predictions_df['predicted']
        
        self.save_temporal_results(model, feature_importance, test_metrics, station_df, predictions_df)
        
        return model, feature_importance, test_metrics, station_df, predictions_df
    
    def save_temporal_results(self, model, feature_importance, test_metrics, station_df, predictions_df):
        output_dir = self.config['output_dir']
        
        model_path = os.path.join(output_dir, 'temporal_tabnet_model.zip')
        model.save_model(model_path)
        self.log(f"✓ Model saved to {model_path}")
        
        if hasattr(self, 'scaler') and self.scaler:
            scaler_path = os.path.join(output_dir, 'temporal_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.log(f"✓ Scaler saved to {scaler_path}")
        
        bias_correction_path = os.path.join(output_dir, 'temporal_bias_correction_params.pkl')
        with open(bias_correction_path, 'wb') as f:
            pickle.dump(self.bias_correction_params, f)
        self.log(f"✓ Bias correction parameters saved to {bias_correction_path}")
        
        feature_importance.to_csv(os.path.join(output_dir, 'temporal_feature_importance.csv'), index=False)
        station_df.to_csv(os.path.join(output_dir, 'temporal_station_performance.csv'), index=False)
        predictions_df.to_csv(os.path.join(output_dir, 'temporal_predictions.csv'), index=False)
        
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance.to_dict('records'),
            'train_years': self.config['train_years'],
            'test_years': self.config['test_years']
        }
        
        with open(os.path.join(output_dir, 'temporal_model_summary.pkl'), 'wb') as f:
            pickle.dump(summary, f)
        
        self.log(f"✓ All temporal results saved to {output_dir}")
    
    def run_temporal_pipeline(self, remote_source='ERA5'):
        self.log(f"🚀 Starting temporal precipitation modeling pipeline with {remote_source} data")
        self.log(f"⏰ Training: {min(self.config['train_years'])}-{max(self.config['train_years'])}")
        self.log(f"⏰ Testing: {min(self.config['test_years'])}-{max(self.config['test_years'])}")
        self.log(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        df = self.load_and_merge_data(remote_source=remote_source)
        if df is None:
            return None
        
        train_df, test_df = self.temporal_split_data(df)
        
        results = self.train_and_evaluate_temporal(train_df, test_df)
        if results is None:
            return None
        
        model, importance, test_metrics, station_df, predictions_df = results
        
        self.log("✅ Temporal pipeline completed successfully!")
        return model, test_metrics
    
    def load_model(self, model_path):
        model = TabNetRegressor()
        model.load_model(model_path)
        return model

if __name__ == "__main__":
    pipeline = TemporalPrecipitationModelingPipeline()

    model, metrics = pipeline.run_temporal_pipeline(remote_source='ERA5')


