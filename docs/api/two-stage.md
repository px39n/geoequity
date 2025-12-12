# TwoStageModel

::: geoequity.two_stage.model.TwoStageModel
    options:
      show_source: true
      heading_level: 2

## Methods

### fit

```python
def fit(self, df, model_col, target_col='observed', 
        density_col='density', lon_col='longitude', lat_col='latitude'):
```

Train the two-stage model on test data.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| df | DataFrame | Test data with predictions and ground truth |
| model_col | str | Column name containing model predictions |
| target_col | str | Column name containing ground truth values |
| density_col | str | Column name containing density values |
| lon_col | str | Column name for longitude |
| lat_col | str | Column name for latitude |

### predict

```python
def predict(self, longitude, latitude, density):
```

Predict accuracy for given coordinates and density.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| longitude | float or array | Longitude(s) to predict |
| latitude | float or array | Latitude(s) to predict |
| density | float or array | Density value(s) |

**Returns:** Predicted R² score(s)

### diagnose

```python
def diagnose(self, save_dir='diagnostics/', show=True):
```

Generate diagnostic plots and summary report.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| save_dir | str | Directory to save outputs |
| show | bool | Whether to display plots |

**Outputs:**
- `stage1_gam.png`: GAM partial dependence plots
- `stage2_svm.png`: Spatial residual map
- `diagnosis.txt`: Summary statistics

