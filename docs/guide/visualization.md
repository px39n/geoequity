# Visualization

GeoEquity provides tools for creating publication-ready spatial visualizations.

## Accuracy Maps

Visualize predicted accuracy across space:

```python
from geoequity.visualization import plot_accuracy_map

fig, ax = plot_accuracy_map(
    model,
    ds,                    # xarray Dataset with coordinates
    mask=land_mask,        # Optional land/sea mask
    cmap='Spectral_r',
    vmin=0, vmax=1,
    title='Model Accuracy'
)
```

## Diagnostic Plots

The TwoStageModel includes built-in diagnostic visualization:

```python
model.diagnose(save_dir='diagnostics/', show=True)
```

This generates:
- **stage1_gam.png**: Partial dependence plots for density features
- **stage2_svm.png**: Spatial map of residual patterns

## Custom Visualizations

### Accuracy Comparison

Compare multiple models or visualization modes:

```python
from geoequity.visualization import plot_accuracy_comparison

fig, axes = plot_accuracy_comparison(
    df,
    model_name='MyModel',
    modes=['observation', 'interpolation', 'spatial_model'],
    accuracy_range=(0, 1),
    lon_range=(-10, 35),
    lat_range=(35, 70)
)
```

### Density Distribution

Visualize data density across space:

```python
import matplotlib.pyplot as plt

plt.scatter(
    df['longitude'], df['latitude'],
    c=df['density'], cmap='viridis',
    s=5, alpha=0.5
)
plt.colorbar(label='Density')
```

## Styling Tips

- Use `Spectral_r` or `RdYlGn` for accuracy (red=low, green=high)
- Set consistent `vmin=0, vmax=1` for R² values
- Add land boundaries with geopandas for context

