# geoequity

**Spatial Equity Assessment for Machine Learning Models**

`geoequity` provides tools to diagnose and visualize spatial performance disparities in geospatial machine learning models. It helps identify where models underperform due to data sparsity and provides methods to predict and visualize accuracy across space.

## Installation

```bash
pip install geoequity
```

Or install from source:

```bash
git clone https://github.com/px39n/geoequity.git
cd geoequity
pip install -e .
```

## Quick Start

```python
from geoequity import TwoStageModel
from geoequity.data import split_test_train, calculate_density

# 1. Load and prepare data
import pandas as pd
df = pd.read_pickle('observations.pkl')
df = calculate_density(df, radius=500)

# 2. Split data (site-wise to avoid data leakage)
train_sites, test_sites, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Site', seed=42
)

# 3. Train your ML model and add predictions to df
# df['predicted_mymodel'] = model.predict(...)

# 4. Fit TwoStageModel to analyze spatial accuracy
ts = TwoStageModel(resolution=[30, 30])
ts.fit(df.loc[test_idx], model_col='predicted_mymodel')

# 5. Predict accuracy for new locations
accuracy = ts.predict(longitude=5.0, latitude=50.0, density=0.001)

# 6. Diagnose model performance
ts.diagnose(save_dir='diagnostics/')
```

## Features

### TwoStageModel

Our core method for predicting model accuracy across space:

- **Stage 1**: Monotonic GAM on density features (sparsity, sufficiency)
- **Stage 2**: SVM on spatial features to capture residual patterns

```python
from geoequity import TwoStageModel

model = TwoStageModel(spline=7, lam=0.5, resolution=[30, 30])
model.fit(df, model_col='predicted')

# Get accuracy predictions
accuracy = model.predict(longitude, latitude, sparsity)

# Generate diagnostic report
model.diagnose(save_dir='output/')
# Creates:
#   - stage1_gam.png
#   - stage2_svm.png  
#   - diagnosis.txt
```

### Data Splitting

Support for geospatial cross-validation strategies:

```python
from geoequity.data import split_test_train

# Site-wise (same station always in same set)
train_sites, test_sites, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Site', seed=42
)

# Grid-wise (spatial blocks)
_, _, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Grid', grid_size=10, seed=42
)

# Spatiotemporal (no data leakage in space or time)
_, _, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Spatiotemporal', seed=42
)

# Cross-validation mode
train_list, test_list, train_idx_list, test_idx_list, df = split_test_train(
    df, cv=5, flag='Site', seed=42
)
```

### Density Calculation

Calculate data density for each location:

```python
from geoequity.data import calculate_density

df = calculate_density(df, radius=500)  # 500 km search radius
print(f"Density range: {df['density'].min():.2e} - {df['density'].max():.2e}")
```

### Visualization

```python
# Accuracy map
model.plot_accuracy_map(ds, mask=land_mask, cmap='Spectral_r')

# Predict accuracy
r2 = model.predict(longitude=5.0, latitude=50.0, density=0.001)
```

## Examples

See the `examples/` folder for Jupyter notebooks:

- **`two_stage.ipynb`**: Complete TwoStageModel workflow
- **`validation_compare.ipynb`**: Compare models across validation strategies
- **`spatial_equity_visualization.ipynb`**: Visualize spatial performance disparities

## API Reference

### `geoequity.TwoStageModel`

| Method | Description |
|--------|-------------|
| `fit(df, model_col, ...)` | Train the two-stage model |
| `predict(lon, lat, density, ...)` | Predict accuracy for locations |
| `diagnose(save_dir, show)` | Generate diagnostic plots and report |
| `plot_accuracy_map(ds, ...)` | Plot predicted accuracy as spatial map |

### `geoequity.data`

| Function | Description |
|----------|-------------|
| `split_test_train(df, flag, ...)` | Split data with various strategies |
| `calculate_density(df, radius, ...)` | Calculate data density |

## Requirements

- Python >= 3.8
- numpy >= 1.20
- pandas >= 1.3
- scikit-learn >= 1.0
- pygam >= 0.8.0
- matplotlib >= 3.4
- xarray >= 0.19 (optional)
- geopandas >= 0.10 (optional)

## Citation

If you use this package in your research, please cite:

```bibtex
@article{liang2025geoequity,
  title={Countering Local Overfitting for Equitable Spatiotemporal Modeling},
  author={Liang, Zhehao and Castruccio, Stefano and Crippa, Paola},
  journal={...},
  year={2025}
}
```

## License

MIT License
