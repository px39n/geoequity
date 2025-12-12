# Theory & Problem Formulation

## The Problem: Spatial Equity Assessment

Machine learning models trained on geospatial data exhibit **spatial inequity**: prediction accuracy varies systematically across space. This inequity is often **hidden** by conventional evaluation approaches.

### Why Spatial Inequity Matters

| Issue | Consequence |
|-------|-------------|
| Non-uniform sensor networks | Models prioritize data-dense regions |
| Global metrics mask disparities | Poor performance in sparse areas goes undetected |
| 3+ billion people live outside dense networks | Under-served populations face higher uncertainty |

### The Evaluation Gap

Most spatiotemporal studies report a **single global metric** (e.g., overall $R^2$):

| Approach | What it measures | Problem |
|----------|------------------|---------|
| **Global $R^2$** | Average across all samples | Masks spatial variation |
| **Time-wise CV** | Temporal generalization | Shares spatial neighbors |
| **Random CV** | General skill | Inflates via autocorrelation |

**GeoEquity's goal**: Reveal hidden spatial inequity by predicting model accuracy across space.

---

## Accuracy Surface: Problem Formulation

### The Challenge

Given a trained ML model, we want to assess its accuracy at every spatial location. However:

1. **Per-location $R^2$** only exists at monitored sites
2. **Unobserved locations** require accuracy estimation
3. **Spatial interpolation** fails to capture data-driven effects

### Location-wise Accuracy

At each monitored station $\ell$, we can compute:

$$R^2(\ell) = 1 - \frac{\sum_t (y_t(\ell) - \hat{y}_t(\ell))^2}{\sum_t (y_t(\ell) - \bar{y}(\ell))^2}$$

This reveals how model performance varies across space—but only at **observed locations**.

### Why Spatial Interpolation Fails

A common approach is to interpolate per-location $R^2$ based on geographic proximity. This assumes:

> "Model performance varies smoothly with location"

This assumption **ignores how data availability affects model behavior**. In reality, accuracy depends on:

1. **Station density** $\rho$ at each location
2. **Sample size** $n$ used for training  
3. **Location-specific factors** (terrain, data quality)

---

## TwoStageModel Solution

### Key Insight

Accuracy depends on **data sampling conditions**, not just spatial proximity.

### Stage 1: Density GAM

Model location-wise $R^2$ as a function of station density $\rho$ and sample size $n$:

$$R^2_{\mathrm{GAM}}(\rho, n) = s_1(\log_{10}(n)) + s_2(\rho) + s_3(\log_{10}(n), \rho) + \varepsilon$$

where:

- $s_1(\cdot), s_2(\cdot)$ are **monotonic increasing** spline functions
- $s_3(\cdot, \cdot)$ is a smooth interaction term

**Monotonicity constraints**:

$$\frac{\partial s_1}{\partial \log_{10}(n)} \geq 0, \quad \frac{\partial s_2}{\partial \rho} \geq 0$$

*More data and higher density → higher accuracy.*

### Stage 2: Spatial SVM

Capture location-specific residuals not explained by density:

$$r = R^2 - \hat{R}^2_{\mathrm{GAM}}$$

$$\hat{r} = h(n, \ell), \quad \ell = (\mathrm{lon}, \mathrm{lat})$$

### Final Prediction

$$\hat{R}^2 = \underbrace{g(\rho, n)}_{\text{Density effect}} + \underbrace{h(n, \ell)}_{\text{Spatial residual}}$$

---

## Local Station Density

For any location $\ell$, the **local weighted station density**:

$$\rho(\ell) = \frac{1}{\pi r^2} \sum_{j \in \mathcal{N}(\ell)} w_j$$

where:

- $w_j = \frac{r - \min(d_j, r)}{r}$ is the distance-based weight
- $d_j$ is the Haversine distance to station $j$
- $\mathcal{N}(\ell) = \{j \mid d_j \leq r\}$ is stations within radius $r$

---

## What TwoStageModel Enables

| Use Case | Input | Output |
|----------|-------|--------|
| **Accuracy at new location** | Coordinates + density | Predicted $R^2$ |
| **New data conditions** | Sample size + density | Expected $R^2$ |
| **Decision support** | Hypothetical deployment | Performance forecast |

### Predictive Performance

| Scenario | Correlation | vs. Interpolation |
|----------|-------------|-------------------|
| Unseen data conditions | ~0.95 | +89% improvement |
| Unseen spatial locations | ~0.60 | +46% improvement |

---

## Summary

| Component | Captures | Physical Meaning |
|-----------|----------|------------------|
| Stage 1 GAM | Density effect | More data → higher accuracy |
| Stage 2 SVM | Spatial residual | Location-specific factors |
| Combined | Full accuracy surface | Predict performance anywhere |

**GeoEquity** reveals spatial inequity by predicting accuracy based on **data availability**, not just location.

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
