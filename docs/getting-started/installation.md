# Installation

## From PyPI

```bash
pip install geoequity
```

## From Source

```bash
git clone https://github.com/px39n/geoequity.git
cd geoequity
pip install -e .
```

## Optional Dependencies

For full functionality including GAM models and geospatial visualization:

```bash
pip install geoequity[full]
```

This installs:
- `pygam` - For Generalized Additive Models
- `xarray` - For gridded data handling
- `geopandas` - For geospatial data operations

## Requirements

- Python >= 3.8
- numpy >= 1.20
- pandas >= 1.3
- scikit-learn >= 1.0
- matplotlib >= 3.4
- tqdm >= 4.60

## Verify Installation

```python
import geoequity
print(geoequity.__version__)
```

