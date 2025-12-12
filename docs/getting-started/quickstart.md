# Quick Start

This guide walks you through the basic workflow for assessing spatial equity in your ML models.

## 1. Load Your Data

Your data should be a pandas DataFrame with:
- `longitude`, `latitude` columns
- A column with your model's predictions
- Ground truth values for calculating accuracy

```python
import pandas as pd
df = pd.read_pickle('your_data.pkl')
```

## 2. Calculate Density

Density measures data availability around each observation:

```python
from geoequity.data import calculate_density

df = calculate_density(df, radius=500)  # 500 km search radius
print(f"Density range: {df['density'].min():.2e} - {df['density'].max():.2e}")
```

## 3. Split Data

Use site-wise splitting to avoid data leakage:

```python
from geoequity.data import split_test_train

train_sites, test_sites, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Site', seed=42
)
```

## 4. Fit TwoStageModel

After training your ML model and adding predictions to `df`:

```python
from geoequity import TwoStageModel

ts = TwoStageModel(resolution=[30, 30])
ts.fit(df.loc[test_idx], model_col='predicted_mymodel')
```

## 5. Analyze Results

```python
# Predict accuracy at any location
accuracy = ts.predict(longitude=5.0, latitude=50.0, density=0.001)

# Generate diagnostic plots
ts.diagnose(save_dir='diagnostics/')
```

## Next Steps

- Learn more about the [Two-Stage Model](../guide/two-stage.md)
- Explore [data splitting strategies](../guide/splitting.md)
- See the [complete example notebook](../examples/Standard_workflow_geoequity.ipynb)

