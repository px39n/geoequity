# GeoEquity

**Spatial Equity Assessment for Machine Learning Models**

GeoEquity provides tools to diagnose and visualize spatial performance disparities in geospatial machine learning models. It helps identify where models underperform due to data sparsity and provides methods to predict and visualize accuracy across space.

## Key Features

- 🎯 **Two-Stage Accuracy Prediction**: Predict model accuracy using density features (Stage 1) and spatial patterns (Stage 2)
- 📊 **Spatial Equity Diagnostics**: Identify regions where models underperform
- 🗺️ **Visualization Tools**: Create publication-ready accuracy maps
- ✂️ **Smart Data Splitting**: Site-wise, grid-wise, and spatiotemporal splitting strategies
- 📈 **Model Comparison**: Compare baseline methods for accuracy prediction

## Quick Example

```python
from geoequity import TwoStageModel
from geoequity.data import split_test_train, calculate_density
from geoequity.two_stage.model import find_bins_intervals

# 1. Load and prepare data
df = calculate_density(df, radius=500)
_, _, train_idx, test_idx, df = split_test_train(df, split=0.2, flag='Site', seed=42)

# 2. Train your model & add predictions
df['predicted_mymodel'] = model.predict(X)
df['observed'] = df['target']

# 3. Fit TwoStageModel
bins_intervals = find_bins_intervals(df, density_bins=7)
ts = TwoStageModel(spline=7, lam=0.5, resolution=[30, 30])
ts.fit(df.loc[test_idx], model_name='mymodel', bins_intervals=bins_intervals)

# 4. Analyze results
print(f"Stage 1 R²: {ts.stage1_score:.4f}")  # Density effect
print(f"Stage 2 R²: {ts.stage2_score:.4f}")  # + Spatial residual

# 5. Generate diagnostics
ts.diagnose(save_dir='diagnostics/')
```

## Installation

```bash
pip install geoequity
```

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| [Standard Workflow](examples/Standard_workflow_geoequity.ipynb) | Complete TwoStageModel workflow |
| [Framework Comparison](examples/Framework_Comparision.ipynb) | Compare baseline methods |
| [Scientific Visualization](examples/Scientific_Visualization.ipynb) | Publication-ready maps |

## Documentation

- [Quick Start](getting-started/quickstart.md) - Get started in 5 minutes
- [Two-Stage Model](guide/two-stage.md) - Core methodology
- [Data Splitting](guide/splitting.md) - Avoid data leakage
- [Model Comparison](guide/comparison.md) - Benchmark methods
- [Visualization](guide/visualization.md) - Create accuracy maps

## Core Concepts

### Spatial Inequity

Machine learning models trained on geospatial data exhibit **spatial inequity**: prediction accuracy varies systematically across space due to:

| Source | Captured By | Description |
|--------|-------------|-------------|
| Sampling Density | Stage 1 (GAM) | Denser observation networks → more training data → higher accuracy |
| Spatial Residual | Stage 2 (SVM) | Location-specific data quality, terrain complexity, etc. |

### TwoStageModel

Decomposes accuracy prediction into interpretable components:

```
Total Accuracy = Density Effect (Stage 1) + Spatial Residual (Stage 2)
```

This enables:
- Fair assessment of model performance across regions
- Identification of under-served areas needing more observations
- Uncertainty quantification for downstream applications

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
