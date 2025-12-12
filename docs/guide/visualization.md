# Visualization

GeoEquity provides tools for creating publication-ready spatial accuracy visualizations.

## Visualization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `observation` | Per-station R² scatter | Show raw accuracy at observation locations |
| `interpolation` | IDW interpolation | Smooth accuracy surface |
| `average_NxM` | Grid-averaged R² | Aggregate accuracy in spatial bins |
| `spatial_model` | SVM regression | ML-based spatial pattern |
| `two_stage` | TwoStageModel prediction | Density-aware accuracy prediction |

## TwoStageModel Prediction Map

```python
from geoequity.two_stage.visualization import plot_predicted_accuracy_map

# Get station network
stations = df.drop_duplicates(subset=['longitude', 'latitude'])
sufficiency = df['sufficiency'].iloc[0]

# Plot density + predicted accuracy
fig, axes = plot_predicted_accuracy_map(
    ts_model,
    stations['longitude'].values,
    stations['latitude'].values,
    sufficiency,
    lon_range=(-10, 35),
    lat_range=(35, 70),
    grid_size=30
)
plt.suptitle('TwoStageModel: Density → Accuracy', y=1.02)
plt.show()
```

## Compare Multiple Modes

```python
from geoequity.visualization import plot_accuracy_comparison

fig, axes = plot_accuracy_comparison(
    df.loc[test_idx],
    model_name='linear',
    modes=['observation', 'interpolation', 'average_15x20', 'spatial_model', 'two_stage'],
    accuracy_range=(0, 1),
    lon_range=(-10, 35),
    lat_range=(35, 70),
    # Required for 'two_stage' mode:
    ts_model=ts_model,
    station_lons=stations['longitude'].values,
    station_lats=stations['latitude'].values
)
plt.suptitle('Model Accuracy - Visualization Modes', y=1.02)
plt.show()
```

## Single Mode Plot

```python
from geoequity.visualization import plot_accuracy_map

fig, ax = plot_accuracy_map(
    df.loc[test_idx],
    model_name='linear',
    mode='interpolation',
    accuracy_range=(0, 1),
    lon_range=(-10, 35),
    lat_range=(35, 70)
)
plt.show()
```

## Diagnostic Plots

```python
# Generate TwoStageModel diagnostics
ts_model.diagnose(save_dir='diagnostics/', show=True)
```

Outputs:
- **stage1_gam.png**: GAM partial dependence plots
- **stage2_svm.png**: Spatial residual map
- **diagnosis.txt**: Summary statistics

## Note on Negative R² Values

R² can be negative when the model performs worse than predicting the mean. This happens when:
- Sparse regions have insufficient data
- Model has poor generalization in certain areas

If negative R² values appear frequently, consider:

| Alternative Metric | Description |
|--------------------|-------------|
| **MSE/RMSE** | Always positive, easier to interpret |
| **MAE** | More robust to outliers |
| **Correlation (r)** | Ranges -1 to 1, captures linear relationship |

## Styling Tips

- Use `Spectral_r` or `RdYlGn` for accuracy (red=low, green=high)
- Set consistent `accuracy_range=(0, 1)` for R² values
- Add land boundaries with geopandas for context
- Use `grid_size=30` for smooth maps, lower for faster rendering
