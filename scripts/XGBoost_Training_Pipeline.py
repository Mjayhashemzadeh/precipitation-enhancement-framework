import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class PrecipitationModelingPipeline:
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
            'output_dir': 'output',
            'target_col': 'precipitation_ground',
            'scaling': True,
            'include_seasonality': True,
            'apply_bias_correction': True,
            'bias_correction_method': 'quantile_mapping',
            'xgb_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'random_state': 42
            }
        }
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        self.log_file = os.path.join(self.config['output_dir'], f"model_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        self.emoji_map = {
            '📊': '[DATA]',
            '✓': '[OK]',
            '⚠️': '[WARNING]',
            '❌': '[ERROR]',
            '🔍': '[PROCESSING]',
            '🚀': '[STARTING]',
            '🔄': '[VALIDATION]',
            '🏠': '[STATION]',
            '📈': '[FEATURES]',
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
    
    def apply_bias_correction(self, df, target_col='precipitation_ground', remote_col='precipitation_remote'):
        if not self.config.get('apply_bias_correction', False):
            return df
        
        method = self.config.get('bias_correction_method', 'quantile_mapping')
        self.log(f"🔍 Applying bias correction using {method}...")
        
        if method == 'quantile_mapping':
            ground_values = df[target_col].values
            remote_values = df[remote_col].values
            
            ground_quantiles = np.percentile(ground_values, np.arange(0, 101, 5))
            remote_quantiles = np.percentile(remote_values, np.arange(0, 101, 5))
            
            self.bias_correction_params = {
                'ground_quantiles': ground_quantiles,
                'remote_quantiles': remote_quantiles
            }
            
            corrected_values = np.zeros_like(remote_values)
            for i, val in enumerate(remote_values):
                if val <= remote_quantiles[0]:
                    corrected_values[i] = ground_quantiles[0]
                elif val >= remote_quantiles[-1]:
                    corrected_values[i] = ground_quantiles[-1]
                else:
                    idx = np.where(remote_quantiles <= val)[0][-1]
                    next_idx = idx + 1
                    weight = (val - remote_quantiles[idx]) / (remote_quantiles[next_idx] - remote_quantiles[idx])
                    corrected_values[i] = ground_quantiles[idx] + weight * (ground_quantiles[next_idx] - ground_quantiles[idx])
            
            df[f'{remote_col}_corrected'] = corrected_values
            self.log(f"✓ Bias correction applied, created column: {remote_col}_corrected")
        
        elif method == 'linear_scaling':
            ground_mean = df[target_col].mean()
            remote_mean = df[remote_col].mean()
            
            scaling_factor = ground_mean / remote_mean
            self.bias_correction_params = {'scaling_factor': scaling_factor}
            
            df[f'{remote_col}_corrected'] = df[remote_col] * scaling_factor
            self.log(f"✓ Linear scaling applied with factor {scaling_factor:.3f}")
        
        else:
            self.log(f"⚠️ Unknown bias correction method: {method}")
            return df
        
        return df
    
    def preprocess_data(self, df):
        if df is None or df.empty:
            self.log("❌ No data to preprocess")
            return None, None, None
        
        df = self.apply_bias_correction(df)
        
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
        
        self.log(f"🔍 Selected features: {', '.join(features)}")
        
        pre_dropna_len = len(df)
        df = df.dropna(subset=features + [target_col])
        post_dropna_len = len(df)
        
        if pre_dropna_len != post_dropna_len:
            self.log(f"⚠️ Removed {pre_dropna_len - post_dropna_len} rows with missing values")
        
        X = df[features]
        y = df[target_col]
        
        if self.config.get('scaling', False):
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=features, index=X.index)
            self.log("✓ Applied StandardScaler to features")
        
        return X, y, df['station']
    
    def evaluate_model(self, y_true, y_pred):
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred)
        }
    
    def train_and_evaluate(self, X, y, stations):
        if X is None or y is None:
            self.log("❌ No data for training")
            return None
        
        xgb_params = self.config.get('xgb_params', {})
        self.log(f"🚀 Training XGBoost with parameters: {xgb_params}")
        
        self.log("🔄 K-Fold Cross-Validation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        kfold_results = []
        
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            model = XGBRegressor(**xgb_params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X.iloc[test_idx])
            fold_metrics = self.evaluate_model(y.iloc[test_idx], y_pred)
            kfold_results.append(fold_metrics)
            self.log(f"   Fold {i+1}: RMSE={fold_metrics['RMSE']:.2f}, R²={fold_metrics['R²']:.2f}")
        
        kfold_df = pd.DataFrame(kfold_results)
        self.log(f"📊 K-Fold Mean Results:\n{kfold_df.mean().to_string()}")
        
        self.log("\n🏠 Leave-One-Station-Out Validation...")
        logo = LeaveOneGroupOut()
        logo_results = []
        station_results = []
        
        for train_idx, test_idx in logo.split(X, y, groups=stations):
            test_station = stations.iloc[test_idx].unique()[0]
            model = XGBRegressor(**xgb_params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = model.predict(X.iloc[test_idx])
            metrics = self.evaluate_model(y.iloc[test_idx], y_pred)
            
            logo_results.append(metrics)
            station_results.append({
                'station': test_station,
                'records': len(test_idx),
                **metrics
            })
            
            self.log(f"   Station {test_station}: RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.2f}")
        
        logo_df = pd.DataFrame(logo_results)
        station_df = pd.DataFrame(station_results)
        
        self.log(f"📊 Leave-One-Station-Out Mean Results:\n{logo_df.mean().to_string()}")
        
        station_df.to_csv(os.path.join(self.config['output_dir'], 'station_performance.csv'), index=False)
        
        self.log("\n🚀 Training Final Model on All Data...")
        final_model = XGBRegressor(**xgb_params)
        final_model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        self.log("📈 Feature Importance:")
        for _, row in feature_importance.iterrows():
            self.log(f"   - {row['Feature']}: {row['Importance']:.4f}")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature')
        plt.title('Feature Importance (Final Model)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'feature_importance.png'))
        plt.close()
        
        self.save_results(final_model, feature_importance, kfold_df, logo_df, station_df)
        
        return final_model, feature_importance
    
    def save_results(self, model, feature_importance, kfold_df, logo_df, station_df):
        output_dir = self.config['output_dir']
        
        model_path = os.path.join(output_dir, 'xgb_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.log(f"✓ Model saved to {model_path}")
        
        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.log(f"✓ Scaler saved to {scaler_path}")
        
        bias_correction_path = os.path.join(output_dir, 'bias_correction_params.pkl')
        with open(bias_correction_path, 'wb') as f:
            pickle.dump(self.bias_correction_params, f)
        self.log(f"✓ Bias correction parameters saved to {bias_correction_path}")
        
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        kfold_df.to_csv(os.path.join(output_dir, 'kfold_results.csv'), index=False)
        logo_df.to_csv(os.path.join(output_dir, 'logo_results.csv'), index=False)
        
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'kfold_mean': kfold_df.mean().to_dict(),
            'logo_mean': logo_df.mean().to_dict(),
            'feature_importance': feature_importance.to_dict('records')
        }
        
        with open(os.path.join(output_dir, 'model_summary.pkl'), 'wb') as f:
            pickle.dump(summary, f)
        
        self.log(f"✓ All results saved to {output_dir}")
    
    def run_pipeline(self, remote_source='ERA5'):
        self.log(f"🚀 Starting precipitation modeling pipeline with {remote_source} data")
        self.log(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        df = self.load_and_merge_data(remote_source=remote_source)
        if df is None:
            return None
        
        X, y, stations = self.preprocess_data(df)
        if X is None:
            return None
        
        model, importance = self.train_and_evaluate(X, y, stations)
        
        self.log("✅ Pipeline completed successfully!")
        return model

if __name__ == "__main__":
    pipeline = PrecipitationModelingPipeline()

    model = pipeline.run_pipeline(remote_source='ERA5')
