# Two-Stage Model

The Two-Stage Model is GeoEquity's core method for predicting and understanding spatial variations in model accuracy.

## How It Works

### Stage 1: Density-Based Accuracy

The first stage uses a Generalized Additive Model (GAM) to capture the relationship between data density and model accuracy.

- **Input**: Density features (sparsity, sufficiency)
- **Model**: Monotonic GAM with shape constraints
- **Output**: Baseline accuracy prediction

### Stage 2: Spatial Residuals

The second stage uses a Support Vector Machine (SVM) to capture spatial patterns in the residuals from Stage 1.

- **Input**: Spatial coordinates (longitude, latitude)
- **Model**: SVM with RBF kernel
- **Output**: Spatial adjustment to accuracy

## Usage

```python
from geoequity import TwoStageModel

# Initialize with custom parameters
model = TwoStageModel(
    spline=7,           # Number of spline bases for GAM
    lam=0.5,            # GAM regularization
    resolution=[30, 30] # Grid resolution for SVM
)

# Fit the model
model.fit(
    df,                      # DataFrame with test data
    model_col='predicted',   # Column with model predictions
    target_col='observed'    # Column with ground truth (optional)
)

# Predict accuracy
accuracy = model.predict(
    longitude=5.0,
    latitude=50.0,
    density=0.001
)

# Generate diagnostics
model.diagnose(save_dir='output/', show=True)
```

## Diagnostic Outputs

The `diagnose()` method generates:

1. **stage1_gam.png**: GAM partial dependence plots showing density-accuracy relationship
2. **stage2_svm.png**: Spatial map of residual patterns
3. **diagnosis.txt**: Summary statistics and model parameters

## Interpretation

- **High Stage 1 residuals**: Spatial factors beyond density affect accuracy
- **Clustered Stage 2 patterns**: Geographic regions with systematic over/under-performance
- **Uniform Stage 2**: Density alone explains accuracy variations

