# GeoEquity

**Spatial Equity Assessment for Machine Learning Models**

GeoEquity diagnoses and visualizes spatial performance disparities in geospatial ML models—identifying where models underperform and predicting accuracy across space.

## Features

- 🎯 **Two-Stage Accuracy Prediction**: Density effect (GAM) + Spatial residual (SVM)
- 📊 **Spatial Equity Diagnostics**: Identify under-served regions
- 🗺️ **Visualization Tools**: Publication-ready accuracy maps
- 📈 **Model Comparison**: Benchmark against baselines

## Quick Example

```python
from geoequity import TwoStageModel

# Fit on your validated predictions
ts = TwoStageModel()
ts.fit(df_test, model_name='mymodel')

# Predict accuracy anywhere
accuracy = ts.predict(longitude=5.0, latitude=50.0, density=0.001)

# Generate diagnostics
ts.diagnose(save_dir='diagnostics/')
```

### Required Data

**Input**: `df_test` should be a **test set DataFrame** (out-of-sample predictions) containing:

| Column | Required | Description |
|--------|----------|-------------|
| `longitude`, `latitude` | ✓ | Spatial coordinates |
| `observed` | ✓ | Ground truth values |
| `predicted_{model_name}` | ✓ | Model predictions (name must match `model_name` parameter) |
| `density` | ✓ | Spatial data density (user-defined function) |
| `sufficiency` | Optional* | Training sample size per prediction |

**\*Note on optional fields**:

- **`density`**: Any user-defined spatial density metric (kernel density, k-NN, IDW, etc.)
  ```python
  df['density'] = gaussian_kernel(coords, bandwidth=1.0)  # Example
  ```

- **`sufficiency`**: Optional if all samples share the same training size (auto-detected). Required for k-fold CV.
  ```python
  df['sufficiency'] = 1000000  # Single value
  # OR
  pd.concat([df1.assign(sufficiency=800000), df2.assign(sufficiency=1000000)])  # Multiple
  ```

## Installation

```bash
pip install geoequity
```

## Documentation

| Section | Description |
|---------|-------------|
| [Quick Start](getting-started/quickstart.md) | Data preparation + basic usage |
| [Two-Stage Model](guide/two-stage.md) | Core methodology |
| [Model Comparison](guide/comparison.md) | Benchmark methods |
| [Visualization](guide/visualization.md) | Accuracy maps |

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| [Standard Workflow](examples/Standard_workflow_geoequity.ipynb) | Complete pipeline |
| [Framework Comparison](examples/Framework_Comparision.ipynb) | Compare baselines |
| [Scientific Visualization](examples/Scientific_Visualization.ipynb) | Publication maps |

## Citation

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
