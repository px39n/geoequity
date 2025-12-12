# Theory

## Problem Formulation

Given a validation dataset $\mathcal{D}_{\text{val}}$ collected under varying observational conditions and cross-validation or repeated experiments $\mathcal{D}_{\text{train}}$ under different sampling sizes, we aim to explain how $R^2$ changes as a function of **density**, **sample size**, and **space**.

The validation set $\mathcal{D}_{\text{val}}$ is further split into a fitting subset and a 20% held-out subset.

We adopt a **two-stage modeling strategy**:

- **Stage 1**: How $R^2$ is impacted by density and sample size
- **Stage 2**: Spatial dependence for the residuals of Stage 1

---

## Stage 1: Density GAM

For every location independently, $R^2$ is modeled as a function of station density $\rho$ and sample size $n$. We use a Generalized Additive Model:

$$R^2_{\text{GAM}}(\rho, n) = s_1(\log_{10}(n)) + s_2(\rho) + s_3(\log_{10}(n), \rho) + \varepsilon$$

where:

- $s_1(\cdot)$ and $s_2(\cdot)$ are **monotonic increasing** spline functions
- $s_3(\log_{10}(n), \rho)$ is a smooth interaction term implemented as a tensor-product spline, allowing the effect of sample size to vary with density

**Monotonicity constraints**:

$$\frac{\partial s_1}{\partial \log_{10}(n)} \geq 0, \quad \frac{\partial s_2}{\partial \rho} \geq 0$$

These encode the physical assumption that accuracy increases with both $n$ and $\rho$.

The GAM is trained on data aggregated into sampling-number × density bins, producing the global baseline $\hat{R}^2_{\text{GAM}} = g(\rho, n)$.

---

## Stage 2: Spatial SVM

Compute residuals $r = R^2 - \hat{R}^2_{\text{GAM}}$ as spatially-aggregated data under different sampling sizes $n$.

Train a Support Vector Regression (SVR) model to predict these residuals:

$$\hat{r} = h(n, \ell)$$

where $\ell = (\text{lon}, \text{lat})$ captures location-specific deviations.

!!! note
    $\rho$ is omitted in Stage 2 because density effects are already absorbed by Stage 1.

---

## Final Prediction

The combined prediction:

$$\hat{R}^2 = g(\rho, n) + h(n, \ell) = \hat{R}^2_{\text{GAM}} + \hat{r}_{\text{SVM}}$$

This decomposition separates the **interpretable density-accuracy relationship** (Stage 1) from **spatial residual patterns** (Stage 2), enabling both global trend estimation and local correction.

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
