# Visualization

## plot_accuracy_map

Plot predicted accuracy as a spatial map.

### Usage

```python
from geoequity.visualization import plot_accuracy_map

fig, ax = plot_accuracy_map(
    model,
    ds,
    mask=land_mask,
    cmap='Spectral_r',
    vmin=0, vmax=1
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| model | TwoStageModel | - | Fitted TwoStageModel |
| ds | xarray.Dataset | - | Dataset with lon/lat coordinates |
| mask | array | None | Boolean mask for valid regions |
| cmap | str | 'Spectral_r' | Matplotlib colormap |
| vmin | float | 0 | Colorbar minimum |
| vmax | float | 1 | Colorbar maximum |

---

## plot_accuracy_comparison

Compare accuracy across multiple visualization modes.

### Usage

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

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| df | DataFrame | - | Data with model predictions |
| model_name | str | - | Model column prefix |
| modes | list | - | Visualization modes to compare |
| accuracy_range | tuple | (0, 1) | Colorbar range |
| lon_range | tuple | None | Longitude bounds |
| lat_range | tuple | None | Latitude bounds |

