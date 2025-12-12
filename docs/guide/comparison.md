# Model Comparison

Compare different baseline methods for predicting spatial accuracy patterns.

## Evaluation Scenarios

| Scenario | Split Method | Question Answered |
|----------|--------------|-------------------|
| **Unseen Spatial** | spatial | How well predict accuracy at *new locations*? |
| **Unseen Sampling** | sampling | How well predict accuracy for *new density levels*? |

## Models Compared

| Category | Model | Description |
|----------|-------|-------------|
| **Traditional ML** | `linear` | Linear Regression |
| | `svm` | Support Vector Regression (RBF) |
| | `lightgbm` | Gradient Boosting |
| **GAM** | `gam_monotonic` | Monotonic GAM with interaction |
| **Interpolation** | `interpolation` | IDW (Inverse Distance Weighting) |
| **Two-Stage** | `two_stage` | GAM (density) + SVM (spatial residual) |

## Usage

```python
from geoequity.evaluation import eval_baseline_comparison

# Define models to compare
model_list = ['linear', 'lightgbm', 'svm', 'gam_monotonic', 'interpolation', 'two_stage']

# Scenario 1: Unseen Spatial
report_spatial = eval_baseline_comparison(
    df_analysis,
    model_list=model_list,
    density_bins=30,
    split_method='spatial',
    train_by='grid',
    evaluate_by='grid',
    metric='correlation',
    full_features='Spatial'
)

# Scenario 2: Unseen Sampling
report_sampling = eval_baseline_comparison(
    df_analysis,
    model_list=model_list,
    density_bins=30,
    split_method='sampling',
    train_by='grid',
    evaluate_by='sampling',
    metric='correlation',
    full_features='Spatial'
)
```

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(report_spatial, report_sampling, model_list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pred_name = list(report_spatial.keys())[0]
    spatial_scores = [report_spatial[pred_name].get(m, 0) for m in model_list]
    sampling_scores = [report_sampling[pred_name].get(m, 0) for m in model_list]
    
    x = np.arange(len(model_list))
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF6B6B']
    
    axes[0].bar(x, spatial_scores, color=colors)
    axes[0].set_title('Unseen Spatial')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_list, rotation=45)
    
    axes[1].bar(x, sampling_scores, color=colors)
    axes[1].set_title('Unseen Sampling')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_list, rotation=45)
    
    plt.tight_layout()
    plt.show()

plot_comparison(report_spatial, report_sampling, model_list)
```

## Key Insights

### Unseen Spatial (New Locations)
- **Interpolation methods (IDW)** leverage spatial autocorrelation
- **TwoStageModel** captures both global density effect and local patterns
- Traditional ML often struggles without spatial structure

### Unseen Sampling (New Density Levels)
- **GAM** excels at capturing density→accuracy relationship
- **TwoStageModel** combines density modeling with spatial residuals
- Linear/SVM fail to generalize to unseen density ranges

### TwoStageModel Advantage

Decomposes the problem into interpretable components:
- **Stage 1**: Global density effect (monotonic relationship)
- **Stage 2**: Location-specific residuals (spatial patterns)

This decomposition provides:
1. Better generalization to new conditions
2. Interpretable insights about accuracy drivers
3. Separate modeling of different sources of variation

