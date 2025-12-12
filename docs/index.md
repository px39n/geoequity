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

| Column | Description |
|--------|-------------|
| `longitude`, `latitude` | Coordinates |
| `observed` | Ground truth |
| `predicted_{model_name}` | Model predictions |
| `density` | Data density |
| `sufficiency` | Sample size |

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
