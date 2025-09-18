import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
from datetime import datetime

def tune_xgboost_parameters(X, y, stations=None, n_iter=100, cv=5, n_jobs=-1, output_dir=None):
    """
    Perform hyperparameter tuning for XGBoost using RandomizedSearchCV.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    stations : Series, optional
        Station identifiers for potential grouped CV
    n_iter : int
        Number of parameter settings sampled in RandomizedSearchCV
    cv : int or cross-validation generator
        Cross-validation strategy
    n_jobs : int
        Number of parallel jobs (use -1 for all processors)
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Results including best params, best model, and CV results
    """
    # Define parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1, 10],
        'reg_lambda': [0, 0.1, 1, 5, 10]
    }
    
    # Create base model
    xgb = XGBRegressor(random_state=42)
    
    # Configure cross-validation
    if stations is not None and len(stations.unique()) >= cv:
        from sklearn.model_selection import GroupKFold
        cv_strategy = GroupKFold(n_splits=cv)
        cv_generator = cv_strategy.split(X, y, groups=stations)
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_generator = cv_strategy.split(X, y)
    
    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=cv_generator,
        verbose=2,
        random_state=42,
        n_jobs=n_jobs,
        return_train_score=True
    )
    
    # Perform parameter search
    print(f"Starting RandomizedSearchCV with {n_iter} iterations...")
    start_time = datetime.now()
    random_search.fit(X, y)
    end_time = datetime.now()
    print(f"Parameter tuning completed in {end_time - start_time}")
    
    # Get best parameters and model
    best_params = random_search.best_params_
    best_score = np.sqrt(-random_search.best_score_)  # Convert to RMSE
    best_model = random_search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_score:.4f}")
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model
        with open(os.path.join(output_dir, 'tuned_xgb_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save best parameters
        with open(os.path.join(output_dir, 'best_params.pkl'), 'wb') as f:
            pickle.dump(best_params, f)
        
        # Save CV results as CSV
        cv_results = pd.DataFrame(random_search.cv_results_)
        cv_results.to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
        
        # Save summary report
        with open(os.path.join(output_dir, 'tuning_summary.txt'), 'w') as f:
            f.write(f"Parameter Tuning Summary\n")
            f.write(f"=====================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total iterations: {n_iter}\n")
            f.write(f"Cross-validation: {cv} folds\n")
            f.write(f"Duration: {end_time - start_time}\n\n")
            f.write(f"Best RMSE: {best_score:.4f}\n\n")
            f.write(f"Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"- {param}: {value}\n")
    
    # Return results
    return {
        'best_params': best_params,
        'best_model': best_model,
        'best_score': best_score,
        'cv_results': random_search.cv_results_
    }

def evaluate_tuned_model(model, X, y):
    """
    Evaluate the tuned model on the given data.
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained XGBoost model
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X)
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred),
        'R²': r2_score(y, y_pred)
    }
    
    print(f"Model Evaluation:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.4f}")
    
    return metrics

def integrate_with_pipeline(pipeline_instance):
    """
    Integrate parameter tuning into the existing precipitation modeling pipeline.
    
    Parameters:
    -----------
    pipeline_instance : PrecipitationModelingPipeline
        Instance of the PrecipitationModelingPipeline class
        
    Returns:
    --------
    dict
        Dictionary containing tuning results
    """
    # Load and preprocess data
    remote_source = pipeline_instance.config.get('remote_source', 'ERA5')
    df = pipeline_instance.load_and_merge_data(remote_source=remote_source)
    X, y, stations = pipeline_instance.preprocess_data(df)
    
    if X is None or y is None:
        pipeline_instance.log("❌ No data for tuning")
        return None
    
    # Configure tuning parameters
    n_iter = pipeline_instance.config.get('tuning_iterations', 100)
    cv = pipeline_instance.config.get('tuning_cv', 5)
    n_jobs = pipeline_instance.config.get('tuning_n_jobs', -1)
    
    # Create output directory for tuning results
    tuning_output_dir = os.path.join(pipeline_instance.config['output_dir'], 'parameter_tuning')
    
    # Log start of tuning
    pipeline_instance.log(f"🔍 Starting hyperparameter tuning with {n_iter} iterations...")
    
    # Perform tuning
    tuning_results = tune_xgboost_parameters(
        X=X,
        y=y,
        stations=stations,
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        output_dir=tuning_output_dir
    )
    
    # Log completion and best parameters
    pipeline_instance.log(f"✓ Parameter tuning completed")
    pipeline_instance.log(f"📊 Best RMSE: {tuning_results['best_score']:.4f}")
    pipeline_instance.log(f"📈 Best parameters: {tuning_results['best_params']}")
    
    # Update pipeline config with best parameters
    pipeline_instance.config['xgb_params'] = tuning_results['best_params']
    
    # Return results
    return tuning_results

# Example usage:
if __name__ == "__main__":
    from XGBoost_Training_Pipeline import PrecipitationModelingPipeline
    
    # Create pipeline with extended config for tuning
    config = {
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
        'tuning_iterations': 100,  # Number of parameter combinations to try
        'tuning_cv': 5,           # Number of CV folds
        'tuning_n_jobs': -1       # Use all available CPU cores
    }
    
    pipeline = PrecipitationModelingPipeline(config)
    
    # Run parameter tuning
    tuning_results = integrate_with_pipeline(pipeline)
    
    # Train final model with tuned parameters

    model = pipeline.run_pipeline(remote_source='ERA5')
