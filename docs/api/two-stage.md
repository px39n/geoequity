# TwoStageModel

The core model for predicting spatial variations in ML model accuracy.

```python
from geoequity import TwoStageModel

model = TwoStageModel(spline=7, lam=0.5, resolution=[30, 30])
```

## Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| spline | int | 7 | Number of spline bases for GAM |
| lam | float | 0.5 | GAM regularization parameter |
| resolution | list | [30, 30] | Grid resolution for SVM |

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

