# Theory

## The Problem: Hidden Spatial Inequity

Machine learning models trained on geospatial data exhibit **spatial inequity**—prediction accuracy varies systematically across space. This inequity is often masked by global evaluation metrics.

| Challenge | Impact |
|-----------|--------|
| Non-uniform sensor networks | Models prioritize data-dense regions |
| Global $R^2$ hides disparities | Sparse regions underperform undetected |
| Spatial interpolation of accuracy | Ignores data availability effects |

---

## Accuracy Surface: Problem Formulation

Given a validation dataset $\mathcal{D}_{\text{val}}$ collected under varying observational conditions and training experiments $\mathcal{D}_{\text{train}}$ under different sampling sizes, we aim to explain how $R^2$ changes as a function of:

- **Density** $\rho$ — station density at each location
- **Sample size** $n$ — training data volume  
- **Space** $\ell$ — geographic coordinates

The validation set is split into a fitting subset and a 20% held-out subset for evaluation.

---

## Two-Stage Modeling Strategy

We decompose the accuracy prediction into two interpretable stages:

- **Stage 1**: How $R^2$ depends on density and sample size
- **Stage 2**: Spatial residuals not explained by Stage 1

---

### Stage 1: Density GAM

For every location, model $R^2$ as a function of station density $\rho$ and sample size $n$ using a Generalized Additive Model:

$$R^2_{\text{GAM}}(\rho, n) = s_1(\log_{10}(n)) + s_2(\rho) + s_3(\log_{10}(n), \rho) + \varepsilon$$

where:

- $s_1(\cdot)$ and $s_2(\cdot)$ are **monotonic increasing** spline functions
- $s_3(\log_{10}(n), \rho)$ is a smooth interaction term (tensor-product spline), allowing the effect of sample size to vary with density

**Monotonicity constraints** encode the physical assumption that accuracy increases with both $n$ and $\rho$:

$$\frac{\partial s_1}{\partial \log_{10}(n)} \geq 0, \quad \frac{\partial s_2}{\partial \rho} \geq 0$$

The GAM is trained on data aggregated into sampling-number × density bins, producing the global baseline $\hat{R}^2_{\text{GAM}} = g(\rho, n)$.

---

### Stage 2: Spatial SVM

Compute residuals as spatially-aggregated data under different sampling sizes $n$:

$$r = R^2 - \hat{R}^2_{\text{GAM}}$$

Train a Support Vector Regression (SVR) model to predict these residuals:

$$\hat{r} = h(n, \ell)$$

where $\ell = (\text{lon}, \text{lat})$ captures location-specific deviations.

!!! note "Why $\rho$ is omitted in Stage 2"
    Density effects are already absorbed by Stage 1, so Stage 2 only captures spatial patterns beyond density.

---

### Final Prediction

The combined prediction:

$$\hat{R}^2 = g(\rho, n) + h(n, \ell) = \hat{R}^2_{\text{GAM}} + \hat{r}_{\text{SVM}}$$

This decomposition separates:

1. **Interpretable density-accuracy relationship** (Stage 1)
2. **Spatial residual patterns** (Stage 2)

Enabling both global trend estimation and local correction.

---

## Summary

| Stage | Input | Captures |
|-------|-------|----------|
| Stage 1 (GAM) | Density $\rho$, Sample size $n$ | Global density-accuracy trend |
| Stage 2 (SVM) | Location $\ell$, Sample size $n$ | Location-specific residuals |
| Combined | All | Full accuracy surface |

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
