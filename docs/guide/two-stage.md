# Two-Stage Model

The Two-Stage Model is GeoEquity's core method for predicting and understanding spatial variations in model accuracy.

## How It Works

### Stage 1: Density-Based Accuracy (GAM)

The first stage uses a **Monotonic Generalized Additive Model (GAM)** to capture the relationship between data density and model accuracy.

- **Input**: Density features (sparsity, sufficiency)
- **Model**: Monotonic GAM with shape constraints
- **Output**: Baseline accuracy prediction

### Stage 2: Spatial Residuals (SVM)

The second stage uses a **Support Vector Machine (SVM)** to capture spatial patterns in the residuals from Stage 1.

- **Input**: Spatial coordinates (longitude, latitude)
- **Model**: SVM with RBF kernel
- **Output**: Spatial adjustment to accuracy

## Experiment Modes

### Single Sufficiency Mode

Use full dataset as one sample size level:

```python
EXPERIMENT_MODE = 'single'

# Sample to 100k if dataset is larger
n_suff = min(100000, len(df))
df = df.sample(n=n_suff, random_state=42).copy()
df['sufficiency'] = 100000
```

**Use when**: Quick analysis, production models with fixed training size.

### Multi Sufficiency Mode

Study how accuracy changes with data volume:

```python
EXPERIMENT_MODE = 'multi'
SUFFICIENCY_LEVELS = [5000, 20000, 100000]

df_list = []
for suff in SUFFICIENCY_LEVELS:
    df_sampled = df.sample(n=suff, random_state=42).copy()
    df_sampled['sufficiency'] = suff
    df_list.append(df_sampled)
df = pd.concat(df_list, ignore_index=True)
```

**Use when**: Research on data efficiency, understanding sampling effects.

## Usage

```python
from geoequity import TwoStageModel
from geoequity.two_stage.model import find_bins_intervals

# Create density bins for aggregation
bins_intervals = find_bins_intervals(df, density_bins=7)

# Initialize model
ts_model = TwoStageModel(
    spline=7,           # Number of spline bases for GAM
    lam=0.5,            # GAM regularization
    resolution=[30, 30] # Spatial aggregation grid
)

# Fit the model on test data
ts_model.fit(
    df_train_raw=df.loc[test_idx],
    model_name='linear',           # Matches 'predicted_linear' column
    bins_intervals=bins_intervals,
    split_by='grid'
)

print(f"Stage 1 R²: {ts_model.stage1_score:.4f}")
print(f"Stage 2 R²: {ts_model.stage2_score:.4f}")
```

## Prediction

```python
from geoequity.two_stage import predict_at_locations, plot_predicted_accuracy_map

# Get station network info
stations = df.drop_duplicates(subset=['longitude', 'latitude'])
sufficiency = df['sufficiency'].iloc[0]

# Predict at specific coordinates
r2, density = predict_at_locations(
    ts_model,
    longitude=5.0,
    latitude=50.0,
    station_lons=stations['longitude'],
    station_lats=stations['latitude'],
    sufficiency=sufficiency
)

# Plot accuracy map
fig, axes = plot_predicted_accuracy_map(
    ts_model,
    stations['longitude'].values,
    stations['latitude'].values,
    sufficiency,
    lon_range=(-10, 35),
    lat_range=(35, 70),
    grid_size=30
)
```

## Diagnostic Outputs

```python
ts_model.diagnose(save_dir='diagnostics/', show=True)
```

Generates:

1. **stage1_gam.png**: GAM partial dependence plots showing density→accuracy relationship
2. **stage2_svm.png**: Spatial map of residual patterns
3. **diagnosis.txt**: Summary statistics

Example output:
```
Stage 1 (Monotonic GAM) - Density/Sampling Effect:
  - r = 0.9872, MAE = 0.0214

Stage 2 (SVM) - Spatial Residual:
  - r = 0.4986, MAE = 0.1830

Key Insights:
  1. Density effect captured in Stage 1 (r=0.987)
  2. Spatial pattern captured in Stage 2 (r=0.499)
  3. Prediction at unseen locations: r=0.643 (Total)
```

## Interpretation

| Pattern | Meaning |
|---------|---------|
| High Stage 1 R² | Density strongly predicts accuracy |
| High Stage 2 residuals | Location-specific factors beyond density |
| Clustered Stage 2 patterns | Geographic regions with systematic bias |
| Uniform Stage 2 | Density alone explains accuracy variations |

## Key Insight

Machine learning models trained on geospatial data exhibit **spatial inequity**: prediction accuracy varies systematically across space due to:

1. **Sampling Density Effect** (Stage 1): Denser observation networks → more training data → higher accuracy
2. **Spatial Residual Effect** (Stage 2): Location-specific data quality, terrain complexity, etc.

TwoStageModel quantifies this inequity, enabling fair assessment and identification of under-served regions.
