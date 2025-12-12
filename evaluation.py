"""
Model evaluation and comparison framework.

Main function: eval_baseline_comparison
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Support both package and direct source code execution
try:
    from .models.spatial import SpatialRegressor
    from .models.interpolation import InterpolationModel
    from .two_stage import TwoStageModel
    from .two_stage.model import find_bins_intervals, prepare_spatial_features, prepare_sampling_features_space
except ImportError:
    from models.spatial import SpatialRegressor
    from models.interpolation import InterpolationModel
    from two_stage import TwoStageModel
    from two_stage.model import find_bins_intervals, prepare_spatial_features, prepare_sampling_features_space


# Display names for models
MODEL_DISPLAY_NAMES = {
    'linear': 'Linear Regression',
    'svm': 'SVM (RBF)',
    'lightgbm': 'LightGBM',
    'gam_monotonic': 'Monotonic GAM',
    'gam_without_interaction': 'GAM (no interaction)',
    'interpolation': 'IDW Interpolation',
    'two_stage': 'TwoStageModel'
}


def eval_baseline_comparison(df, model_list, density_bins=7, split_method='spatial', 
                             train_by='grid', evaluate_by='grid', resolution=None,
                             metric='correlation', full_features='Spatial', 
                             spline=7, lam=0.5, clip=None, use_seeds=True,
                             verbose=True):
    """
    Baseline model comparison framework.
    
    Compare different methods for predicting spatial accuracy patterns.
    
    Parameters
    ----------
    df : DataFrame
        Data with columns: 'observed', 'predicted_*', 'longitude', 'latitude', 'density', 'sufficiency'
    model_list : list
        Models to compare: ['linear', 'svm', 'lightgbm', 'gam_monotonic', 'interpolation', 'two_stage']
    density_bins : int, default=7
        Number of density bins for aggregation
    split_method : str, default='spatial'
        'spatial': Split by spatial grid
        'sampling': Split by density bins
    train_by : str, default='grid'
        Training data aggregation: 'grid' or 'sampling'
    evaluate_by : str, default='grid'
        Test data aggregation: 'grid' or 'sampling'
    resolution : list, default=[10, 10]
        Spatial grid resolution [lon_bins, lat_bins]
    metric : str, default='correlation'
        Evaluation metric: 'correlation' or 'r2'
    full_features : str, default='Spatial'
        Feature set: 'Full', 'Spatial', or 'Density'
    spline : int, default=7
        GAM spline knots
    lam : float, default=0.5
        GAM regularization
    clip : list, default=[-0.5, 1.0]
        R² clipping range
    use_seeds : bool, default=True
        Use seeds for aggregation
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    results : dict
        {prediction_model: {baseline_model: score}}
    
    Example
    -------
    >>> model_list = ['linear', 'svm', 'gam_monotonic', 'interpolation', 'two_stage']
    >>> 
    >>> # Scenario 1: Unseen Spatial
    >>> report = eval_baseline_comparison(
    ...     df, model_list=model_list,
    ...     split_method='spatial', evaluate_by='grid',
    ...     metric='correlation'
    ... )
    >>> 
    >>> # Scenario 2: Unseen Sampling
    >>> report = eval_baseline_comparison(
    ...     df, model_list=model_list,
    ...     split_method='sampling', evaluate_by='sampling',
    ...     metric='correlation'
    ... )
    """
    if resolution is None:
        resolution = [10, 10]
    if clip is None:
        clip = [-0.5, 1.0]
    
    # Get prediction model names from columns
    pred_names = [c.replace('predicted_', '') for c in df.columns if c.startswith('predicted_')]
    
    # Feature configuration
    feature_configs = {
        'Full': ['longitude', 'latitude', 'sufficiency_log', 'density'],
        'Spatial': ['longitude', 'latitude'],
        'Density': ['sufficiency_log', 'density']
    }
    feature_cols = feature_configs[full_features]
    
    # Step 1: Prepare bins
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation: {split_method.upper()} split → {evaluate_by.upper()} evaluate")
        print(f"Features: {full_features} | Metric: {metric}")
        print(f"{'='*60}")
    
    bins_intervals = find_bins_intervals(df, density_bins=density_bins)
    
    # Step 2: Split data
    if split_method == 'spatial':
        df_train, df_test = _spatial_split(df, train_ratio=0.7)
    else:
        df_train, df_test = _sampling_split(df, bins_intervals, train_ratio=0.7)
    
    if verbose:
        print(f"Split: Train {len(df_train):,} | Test {len(df_test):,}")
    
    # Step 3: Aggregate training data
    train_dict = prepare_spatial_features(
        df_train, pred_names, split_by='grid',
        include_sampling_features=True, bins_intervals=bins_intervals, 
        resolution=resolution, clip=clip
    )
    
    # Step 4: Aggregate test data
    if evaluate_by == 'sampling':
        test_dict = prepare_sampling_features_space(
            bins_intervals, df_test, pred_names,
            split_by='grid', resolution=resolution, clip=clip
        )
    else:
        test_dict = prepare_spatial_features(
            df_test, pred_names, split_by='grid',
            include_sampling_features=True, bins_intervals=bins_intervals, 
            resolution=resolution, clip=clip
        )
    
    # Step 5: Train and evaluate
    results = {}
    iterator = tqdm(pred_names, desc="Evaluating") if verbose else pred_names
    
    for pred_name in iterator:
        train_data = train_dict.get(pred_name, pd.DataFrame())
        test_data = test_dict.get(pred_name, pd.DataFrame())
        
        if len(train_data) < 5 or len(test_data) < 3:
            continue
        
        # Ensure required columns exist
        for data in [train_data, test_data]:
            if 'sufficiency_log' not in data.columns:
                data['sufficiency_log'] = np.log10(df['sufficiency'].iloc[0])
            if 'density' not in data.columns and 'sparsity' in data.columns:
                data['density'] = data['sparsity']
        
        # Get available features
        available_cols = [c for c in feature_cols if c in train_data.columns]
        X_train = train_data[available_cols].values
        y_train = train_data['r2'].values
        X_test = test_data[available_cols].values
        y_test = test_data['r2'].values
        
        # Evaluate each model
        scores = {}
        for model_type in model_list:
            if model_type == 'two_stage':
                scores[model_type] = _eval_two_stage(
                    df_train, df_test, pred_name, bins_intervals,
                    resolution, spline, lam, clip, metric
                )
            elif model_type == 'interpolation':
                scores[model_type] = _eval_interpolation(train_data, test_data, metric)
            else:
                scores[model_type] = _eval_spatial_model(
                    model_type, X_train, X_test, y_train, y_test, 
                    metric, spline, lam
                )
        
        results[pred_name] = scores
    
    # Print results table
    if verbose:
        _print_results_table(results, model_list, metric)
    
    return results


def _spatial_split(df, train_ratio=0.7, seed=42):
    """Split data by spatial grid."""
    df = df.copy()
    df['lon_bin'] = pd.cut(df['longitude'], bins=10, labels=False)
    df['lat_bin'] = pd.cut(df['latitude'], bins=10, labels=False)
    df['split_id'] = df['lon_bin'].astype(str) + '_' + df['lat_bin'].astype(str)
    
    unique_ids = df['split_id'].dropna().unique()
    np.random.seed(seed)
    np.random.shuffle(unique_ids)
    n_train = int(len(unique_ids) * train_ratio)
    train_mask = df['split_id'].isin(set(unique_ids[:n_train]))
    
    return df[train_mask], df[~train_mask]


def _sampling_split(df, bins_intervals, train_ratio=0.7, seed=42):
    """Split data by density×sufficiency bins."""
    df = df.copy()
    density_edges, suff_map = bins_intervals
    
    df['density_bin'] = pd.cut(df['density'], bins=density_edges, labels=False)
    df['suff_bin'] = df['sufficiency'].map(suff_map)
    df['split_id'] = df['density_bin'].astype(str) + '_' + df['suff_bin'].astype(str)
    
    unique_ids = df['split_id'].dropna().unique()
    np.random.seed(seed)
    np.random.shuffle(unique_ids)
    n_train = int(len(unique_ids) * train_ratio)
    train_mask = df['split_id'].isin(set(unique_ids[:n_train]))
    
    return df[train_mask], df[~train_mask]


def _eval_spatial_model(model_type, X_train, X_test, y_train, y_test, metric, spline, lam):
    """Evaluate a spatial regression model."""
    model = SpatialRegressor(model_type=model_type, spline=spline, lam=lam)
    return model.fit_score(X_train, y_train, X_test, y_test, metric=metric)


def _eval_interpolation(train_data, test_data, metric):
    """Evaluate IDW interpolation."""
    model = InterpolationModel(method='idw', power=2)
    
    train_coords = train_data[['longitude', 'latitude']].values
    test_coords = test_data[['longitude', 'latitude']].values
    y_train = train_data['r2'].values
    y_test = test_data['r2'].values
    
    model.fit(train_coords, y_train)
    return model.score(test_coords, y_test, metric=metric)


def _eval_two_stage(df_train, df_test, model_name, bins_intervals, resolution, spline, lam, clip, metric):
    """Evaluate TwoStageModel."""
    ts = TwoStageModel(spline=spline, lam=lam, resolution=resolution, clip=clip, diagnose=False)
    ts.fit(df_train, model_name, bins_intervals, split_by='grid')
    
    # Aggregate test data
    test_agg = prepare_spatial_features(
        df_test, [model_name], split_by='grid',
        include_sampling_features=True, bins_intervals=bins_intervals,
        resolution=resolution, clip=clip
    ).get(model_name, pd.DataFrame())
    
    if len(test_agg) < 3:
        return np.nan
    
    y_true = test_agg['r2'].values
    density_col = 'density' if 'density' in test_agg else 'sparsity'
    
    y_pred = ts.predict(
        longitude=test_agg['longitude'].values,
        latitude=test_agg['latitude'].values,
        density=test_agg[density_col].values,
        sufficiency=10**test_agg['sufficiency_log'].values
    )
    
    if metric == 'correlation':
        return pearsonr(y_true, y_pred)[0]
    return r2_score(y_true, y_pred)


def _print_results_table(results, model_list, metric):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"Results ({metric.upper()})")
    print(f"{'='*80}")
    
    # Header with display names
    header = f"{'Prediction':<15}"
    for m in model_list:
        name = MODEL_DISPLAY_NAMES.get(m, m)[:12]
        header += f"{name:>14}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for pred_name, scores in results.items():
        row = f"{pred_name:<15}"
        for m in model_list:
            score = scores.get(m, np.nan)
            if np.isnan(score):
                row += f"{'N/A':>14}"
            else:
                row += f"{score:>14.4f}"
        print(row)
    
    print(f"{'='*80}")

