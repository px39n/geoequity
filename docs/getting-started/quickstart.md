# Quick Start

## Overview

```
Data Preparation → TwoStageModel.fit() → Predict & Diagnose
```

---

## Part 1: Data Preparation

Before using GeoEquity, you need a DataFrame with validated ML predictions.

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `longitude` | float | Spatial coordinate |
| `latitude` | float | Spatial coordinate |
| `observed` | float | Ground truth values |
| `predicted_{model_name}` | float | Your model's predictions |
| `density` | float | Data density at location |
| `sufficiency` | int | Training sample size |

### Step 1: Load Data

```python
import pandas as pd

df = pd.read_pickle('your_data.pkl')
df['time'] = pd.to_datetime(df['time'])

# Memory optimization
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype('float32')
```

### Step 2: Calculate Density

```python
from geoequity.data import calculate_density

df = calculate_density(df, radius=500)  # 500 km search radius
```

### Step 3: Split Data

```python
from geoequity.data import split_test_train

_, _, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Site', seed=42
)
```

### Step 4: Train Your Model & Add Predictions

```python
from sklearn.linear_model import LinearRegression
from geoequity.data import simple_feature_engineering

# Feature engineering
FEATURE_COLS = ['feature1', 'feature2', 'latitude', 'longitude']
TARGET_COL = 'target'
X, y, _ = simple_feature_engineering(df, FEATURE_COLS, TARGET_COL)

# Train model
model = LinearRegression()
model.fit(X.loc[train_idx], y.loc[train_idx])

# Add required columns
df['predicted_linear'] = model.predict(X)
df['observed'] = df[TARGET_COL]
df['sufficiency'] = len(train_idx)
```

---

## Part 2: Spatial Equity Analysis

With your prepared data, analyze spatial accuracy patterns.

### Fit TwoStageModel

```python
from geoequity import TwoStageModel
from geoequity.two_stage.model import find_bins_intervals

bins_intervals = find_bins_intervals(df, density_bins=7)

ts = TwoStageModel(spline=7, lam=0.5, resolution=[30, 30])
ts.fit(
    df_train_raw=df.loc[test_idx],
    model_name='linear',
    bins_intervals=bins_intervals
)

print(f"Stage 1 R² (density): {ts.stage1_score:.4f}")
print(f"Stage 2 R² (spatial): {ts.stage2_score:.4f}")
```

### Predict Accuracy

```python
from geoequity.two_stage import predict_at_locations

stations = df.drop_duplicates(subset=['longitude', 'latitude'])

r2, density = predict_at_locations(
    ts, longitude=5.0, latitude=50.0,
    station_lons=stations['longitude'],
    station_lats=stations['latitude'],
    sufficiency=df['sufficiency'].iloc[0]
)
print(f"Predicted R² at (5°E, 50°N): {r2[0]:.4f}")
```

### Generate Diagnostics

```python
ts.diagnose(save_dir='diagnostics/', show=True)
```

Outputs:
- `stage1_gam.png` - Density → Accuracy relationship
- `stage2_svm.png` - Spatial residual patterns
- `diagnosis.txt` - Summary statistics

---

## Next Steps

- [Two-Stage Model details](../guide/two-stage.md)
- [Model Comparison](../guide/comparison.md)
- [Visualization options](../guide/visualization.md)
