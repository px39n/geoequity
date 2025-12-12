# GeoEquity

**Spatial Equity Assessment for Machine Learning Models**

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

Your DataFrame should contain:

| Column | Description |
|--------|-------------|
| `longitude`, `latitude` | Coordinates |
| `observed` | Ground truth values |
| `predicted_{model_name}` | Your model's predictions |
| `density` | Data density at each location |
| `sufficiency` | Training sample size |

## Documentation

📖 **Full documentation**: [https://px39n.github.io/geoequity/](https://px39n.github.io/geoequity/)

- [Quick Start Guide](https://px39n.github.io/geoequity/getting-started/quickstart/)
- [Two-Stage Model](https://px39n.github.io/geoequity/guide/two-stage/)
- [Example Notebooks](https://px39n.github.io/geoequity/examples/Standard_workflow_geoequity/)

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
