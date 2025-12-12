# Data Utilities

## split_test_train

Split data into train/test sets with spatial awareness.

### Usage

```python
from geoequity.data import split_test_train

# Single split
train_sites, test_sites, train_idx, test_idx, df = split_test_train(
    df, split=0.2, flag='Site', seed=42
)

# Cross-validation
train_list, test_list, train_idx_list, test_idx_list, df = split_test_train(
    df, cv=5, flag='Site', seed=42
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| df | DataFrame | - | Input data |
| split | float | 0.2 | Test set fraction |
| flag | str | 'Site' | Splitting strategy: 'Site', 'Grid', 'Spatiotemporal' |
| cv | int | None | Number of CV folds (overrides split) |
| seed | int | 42 | Random seed |
| grid_size | int | 10 | Grid cells per dimension (for Grid flag) |

---

## calculate_density

Calculate data density for each observation based on nearby points.

### Usage

```python
from geoequity.data import calculate_density

df = calculate_density(df, radius=500)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| df | DataFrame | - | Input data with lon/lat columns |
| radius | float | 500 | Search radius in kilometers |
| lon_col | str | 'longitude' | Longitude column name |
| lat_col | str | 'latitude' | Latitude column name |

### Returns

DataFrame with added `density` column.

