# GeoEquity

**Spatial Equity Assessment for Machine Learning Models**

GeoEquity provides tools to diagnose and visualize spatial performance disparities in geospatial machine learning models. It helps identify where models underperform due to data sparsity and provides methods to predict and visualize accuracy across space.

## Key Features

- 🎯 **Two-Stage Accuracy Prediction**: Predict model accuracy using density features (Stage 1) and spatial patterns (Stage 2)
- 📊 **Spatial Equity Diagnostics**: Identify regions where models underperform
- 🗺️ **Visualization Tools**: Create publication-ready accuracy maps
- ✂️ **Smart Data Splitting**: Site-wise, grid-wise, and spatiotemporal splitting strategies

## Quick Example

```python
from geoequity import TwoStageModel
from geoequity.data import split_test_train, calculate_density

# Load and prepare data
df = calculate_density(df, radius=500)
train_sites, test_sites, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Site', seed=42
)

# Fit two-stage model to analyze spatial accuracy
ts = TwoStageModel(resolution=[30, 30])
ts.fit(df.loc[test_idx], model_col='predicted_mymodel')

# Predict accuracy for any location
accuracy = ts.predict(longitude=5.0, latitude=50.0, density=0.001)

# Generate diagnostic report
ts.diagnose(save_dir='diagnostics/')
```

## Installation

```bash
pip install geoequity
```

## Documentation

- [Getting Started](getting-started/installation.md) - Installation and first steps
- [User Guide](guide/two-stage.md) - Detailed usage instructions
- [API Reference](api/two-stage.md) - Complete API documentation
- [Examples](examples/Standard_workflow_geoequity.ipynb) - Jupyter notebook tutorials

## Citation

If you use GeoEquity in your research, please cite:

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

