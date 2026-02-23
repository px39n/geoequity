# GeoEquity

**Spatial Equity Assessment for Machine Learning Models**

[![DOI](https://zenodo.org/badge/1114901804.svg)](https://doi.org/10.5281/zenodo.17915557)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GeoEquity diagnoses and visualizes spatial performance disparities in geospatial ML models—identifying where models underperform and predicting accuracy across space.

## Installation

```bash
pip install geoequity
```

## Quick Start

After training your ML model, use GeoEquity to analyze spatial accuracy patterns:

```python
from geoequity import TwoStageModel

# Your validated predictions (see Required Data below)
ts = TwoStageModel()
ts.fit(df_test, model_name='mymodel')

# Predict accuracy at any location
accuracy = ts.predict(longitude=5.0, latitude=50.0, density=0.001)

# Generate diagnostic report
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

## Documentation

📖 **Full documentation**: [https://px39n.github.io/geoequity/](https://px39n.github.io/geoequity/)

- [Quick Start Guide](https://px39n.github.io/geoequity/getting-started/quickstart/)
- [Two-Stage Model](https://px39n.github.io/geoequity/guide/two-stage/)
- [Example Notebooks](https://px39n.github.io/geoequity/examples/Standard_workflow_geoequity/)

## Data

Example datasets are available on Zenodo: [https://doi.org/10.5281/zenodo.17915696](https://doi.org/10.5281/zenodo.17915696)

Direct download: [geoequity.zip](https://zenodo.org/records/17915696/files/geoequity.zip?download=1)

## Citation

```bibtex
@article{liang2025geoequity,
  title={Local Overfitting Drives Spatially Uneven Performance in Observation-based Atmospheric Models},
  author={Liang, Zhehao and Castruccio, Stefano and Crippa, Paola},
  journal={...},
  year={2025}
}
```

## License

MIT License
