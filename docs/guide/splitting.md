# Data Splitting

Proper data splitting is crucial for valid spatial model evaluation. GeoEquity provides several strategies to prevent data leakage.

## Splitting Strategies

### Site-wise Splitting

Ensures all observations from the same monitoring site stay together:

```python
from geoequity.data import split_test_train

train_sites, test_sites, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Site', seed=42
)
```

**Use when**: You have repeated measurements at fixed locations.

### Grid-wise Splitting

Divides the spatial domain into a grid and assigns entire grid cells to train/test:

```python
_, _, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Grid', grid_size=10, seed=42
)
```

**Use when**: You want to test spatial extrapolation ability.

### Spatiotemporal Splitting

Prevents leakage in both space and time:

```python
_, _, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Spatiotemporal', seed=42
)
```

**Use when**: You have time series at multiple locations.

## Cross-Validation

All strategies support k-fold cross-validation:

```python
train_list, test_list, train_idx_list, test_idx_list, df = split_test_train(
    df, cv=5, flag='Site', seed=42
)

for fold, (train_idx, test_idx) in enumerate(zip(train_idx_list, test_idx_list)):
    print(f"Fold {fold}: {len(train_idx)} train, {len(test_idx)} test")
```

## Why Spatial Splitting Matters

Standard random splitting can cause:

- **Spatial autocorrelation leakage**: Nearby points in train/test sets
- **Overly optimistic accuracy**: Model appears better than it is
- **Poor generalization**: Fails on truly unseen regions

Site-wise and grid-wise splitting prevent these issues by ensuring spatial separation between train and test sets.

