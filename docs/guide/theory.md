# Theory

## What We're Solving

You've trained an ML model and computed $R^2$ at each validation station. Now you want to know:

1. **Why** does accuracy vary across locations?
2. **What** accuracy can I expect at a new location?
3. **How** will accuracy change if I collect more data?

The challenge: you only have $R^2$ values at monitored stations—how do you estimate accuracy everywhere else?

---

## The Two-Stage Approach

We model accuracy as depending on two factors:

| Factor | What it captures |
|--------|-----------------|
| **Data availability** | Density of stations, training sample size |
| **Location** | Geographic patterns not explained by data availability |

```pseudocode
# Input: raw test data
df_test: 100,000 rows → (lon, lat, density, sufficiency, observed, predicted)

# Goal: predict R² at any (lon, lat, density, sufficiency)
```

---

## Stage 1: Data Availability → Accuracy

We fit a GAM (Generalized Additive Model) to predict $R^2$ from:

- **Station density** ($\rho$): How many nearby stations exist
- **Sample size** ($n$): How much training data was used

The model enforces a sensible constraint: **more data = better accuracy**. This is done through monotonic splines that can only increase with $\rho$ and $n$.

**Output**: A baseline accuracy estimate $\hat{R}^2_{\text{GAM}}$ that captures global trends.


```pseudocode
df_test 
  → Given: [sufficiency, lon, lat, observed, predicted, density]

  # Remove local spatial noise first.
  → Spatial Groupby (lon, lat)
  → [sufficiency, lon_avg, lat_avg, density_avg, R²_compute]

  # Extract sampling effect uniformly
  → Sampling Groupby (sufficiency, density)
  → [sufficiency, density_avg_bin, density_avg_avg, R²_avg] 

  # Then, GAM can learn the information of data availability in a uniform way

  → Build GAM(density, sufficiency) → R²

```

---

## Stage 2: Location-Specific Adjustments

After Stage 1, some locations  still have unexplained accuracy differences. These residuals:

$$r = R^2_{\text{observed}} - \hat{R}^2_{\text{GAM}}$$

are modeled using an SVM that learns spatial patterns—capturing things like:

- Regional data quality differences
- Terrain complexity effects
- Local climate variability

**Output**: A spatial correction $\hat{r}_{\text{SVM}}$ for each location.


  → Group by [lon_bin, lat_bin, suff_bin]
  → Get [(lon_bin, lat_bin), density, sufficiency, R²]
  → baseline_r2 = GAM(density, sufficiency)
  → residual = R² - baseline_r2
  → Build SVM(lon, lat) → residual


```pseudocode
df_test 
  → Given: [sufficiency, lon, lat, observed, predicted, density]

  # Remove local spatial noise.
  → Spatial Groupby (lon, lat)
  → [sufficiency, lon_avg, lat_avg, density_avg, R²_compute]

  # Use model in stage 1
  → residual_r2 = R² - GAM(sufficiency, density)

  # Then, SVM can learn the spatial pattern of residuals.
  → Build SVM(lon_bin, lat_bin) → residual_r2

```

---

## Final Prediction

Combine both stages:

$$\hat{R}^2 = \hat{R}^2_{\text{GAM}} + \hat{r}_{\text{SVM}}$$

| Component | Interprets | Extrapolates to |
|-----------|-----------|-----------------|
| Stage 1 | Density-accuracy relationship | New data conditions |
| Stage 2 | Spatial residuals | New locations |
| Combined | Full accuracy surface | Anywhere |

---

## Why This Works Better Than Interpolation

Traditional approach: interpolate $R^2$ based on distance to nearby stations.

**Problem**: This assumes accuracy varies smoothly with location, ignoring that sparse regions systematically underperform regardless of what's nearby.

**Our approach**: Accuracy depends on data availability first, then location. A sparse region will have low predicted accuracy even if surrounded by high-accuracy stations.

---

## Citation

```bibtex
@article{liang2025geoequity,
  title={Countering Local Overfitting for Equitable Spatiotemporal Modeling},
  author={Liang, Zhehao and Castruccio, Stefano and Crippa, Paola},
  journal={...},
  year={2025}
}
```
