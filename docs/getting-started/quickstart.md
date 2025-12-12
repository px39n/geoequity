# Quick Start

This guide walks you through the complete workflow for assessing spatial equity in your ML models.

## Overview

```
1. Load Data → 2. Calculate Density → 3. Split Data → 4. Train Model → 5. Fit TwoStageModel → 6. Predict & Diagnose
```

## 1. Load Your Data

Your data should be a pandas DataFrame with:

| Column | Type | Description |
|--------|------|-------------|
| `longitude` | float | Spatial coordinate |
| `latitude` | float | Spatial coordinate |
| `time` | datetime | Timestamp |
| Target (e.g., `Ozone`) | float | Observed values |
| Features | float | Predictor variables |

```python
import pandas as pd

df = pd.read_pickle('your_data.pkl')
df['time'] = pd.to_datetime(df['time'])

# Convert to float32 for memory efficiency
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype('float32')
```

## 2. Calculate Density

Density measures data availability around each observation:

```python
from geoequity.data import calculate_density

df = calculate_density(df, radius=500)  # 500 km search radius
print(f"Density range: {df['density'].min():.2e} - {df['density'].max():.2e}")
```

## 3. Split Data (Site-wise)

Use site-wise splitting to avoid data leakage:

```python
from geoequity.data import split_test_train

train_sites, test_sites, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Site', seed=42
)
print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
```

## 4. Feature Engineering & Train Model

```python
from geoequity.data import simple_feature_engineering
from sklearn.linear_model import LinearRegression

# Feature engineering with spatial/temporal harmonics
FEATURE_COLS = ['TROPOMI_ozone', 'OMI_ozone', 'tco3', 'blh', 't2m', 'latitude', 'longitude']
TARGET_COL = 'Ozone'

X, y, feature_names = simple_feature_engineering(
    df, FEATURE_COLS, TARGET_COL,
    add_spatial_harmonics=True,
    add_temporal_harmonics=True,
    standardize=True
)

# Train your model
model = LinearRegression()
model.fit(X.loc[train_idx], y.loc[train_idx])

# Add predictions to dataframe
df['predicted_linear'] = model.predict(X)
df['observed'] = df[TARGET_COL]
```

## 5. Fit TwoStageModel

```python
from geoequity import TwoStageModel
from geoequity.two_stage.model import find_bins_intervals

# Create density bins
bins_intervals = find_bins_intervals(df, density_bins=7)

# Initialize and fit
ts_model = TwoStageModel(spline=7, lam=0.5, resolution=[30, 30])
ts_model.fit(
    df_train_raw=df.loc[test_idx],
    model_name='linear',
    bins_intervals=bins_intervals,
    split_by='grid'
)

print(f"Stage 1 R² (density effect): {ts_model.stage1_score:.4f}")
print(f"Stage 2 R² (+ spatial residual): {ts_model.stage2_score:.4f}")
```

## 6. Predict & Diagnose

```python
from geoequity.two_stage import predict_at_locations

# Get station locations
stations = df.drop_duplicates(subset=['longitude', 'latitude'])
sufficiency = df['sufficiency'].iloc[0]

# Predict accuracy at specific locations
r2_dense, _ = predict_at_locations(
    ts_model, 5.0, 50.0,  # Dense area coordinates
    stations['longitude'], stations['latitude'], sufficiency
)
r2_sparse, _ = predict_at_locations(
    ts_model, 30.0, 55.0,  # Sparse area coordinates
    stations['longitude'], stations['latitude'], sufficiency
)

print(f"Dense area (5°E, 50°N):  R² = {r2_dense[0]:.4f}")
print(f"Sparse area (30°E, 55°N): R² = {r2_sparse[0]:.4f}")
print(f"Equity Gap: {r2_dense[0] - r2_sparse[0]:.4f}")

# Generate diagnostic report
ts_model.diagnose(save_dir='diagnostics/', show=True)
```

## Next Steps

- Learn about [experiment modes](../guide/two-stage.md) (single vs multi-sufficiency)
- Explore [data splitting strategies](../guide/splitting.md)
- Compare [baseline methods](../guide/comparison.md)
- Create [publication-ready visualizations](../guide/visualization.md)
