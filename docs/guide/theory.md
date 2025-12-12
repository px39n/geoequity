# Theory & Problem Formulation

## Problem Statement

Consider a spatiotemporal process observed at $n$ monitoring stations $\ell_1, \ell_2, \ldots, \ell_n \in \mathcal{S} \subset \mathbb{R}^2$ over time periods $t \in \{1, 2, \ldots, T\}$. At each spatiotemporal location $(\ell, t)$, we observe the input-output pair $(y_t(\ell), \mathbf{x}_t(\ell))$, where:

- $y_t(\ell) \in \mathbb{R}$ is the target variable
- $\mathbf{x}_t(\ell) \in \mathbb{R}^d$ is a $d$-dimensional covariate vector

Our objective is to estimate a function $f$ parameterized by $\boldsymbol{\theta}$ such that:

$$y_t(\ell) = f(\mathbf{x}_t(\ell); \boldsymbol{\theta}) + \epsilon_t(\ell)$$

where $\epsilon_t(\ell)$ is Gaussian white noise with $\mathbb{E}[\epsilon_t(\ell)] = 0$ and $\text{Var}(\epsilon_t(\ell)) = \sigma^2$.

## The Challenge: Spatial Inequity

The fundamental inferential challenge arises from **spatiotemporal heterogeneity** in the data distribution. Since observations are non-uniformly distributed across the joint covariate-space-time domain, parameter estimates $\hat{\boldsymbol{\theta}}$ become biased towards data-dense regions, leading to poor generalization in sparse areas.

## Local Station Density

For any spatial location $\ell \in \mathcal{S}$, the **local weighted station density** is:

$$\rho(\ell) = \frac{1}{\pi r^2} \sum_{j \in \mathcal{N}(\ell)} w_{\ell j}$$

where:

- $w_j = \frac{r - \min(d_j, r)}{r}$ is the distance-based weight
- $d_j$ is the Haversine distance between location $\ell$ and station $j$
- $\mathcal{N}(\ell) = \{j \mid d_j \leq r\}$ is the set of stations within radius $r$
- $\pi r^2$ normalizes by the search area

### Haversine Distance

The distance $d_{ij}$ between two geographic points is:

$$d_{ij} = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_j - \phi_i}{2}\right) + \cos(\phi_i)\cos(\phi_j)\sin^2\left(\frac{\lambda_j - \lambda_i}{2}\right)}\right)$$

where $R = 6371$ km (Earth's radius), and $(\phi, \lambda)$ are latitude and longitude in radians.

## Types of Imbalance

### Spatial Imbalance

Non-uniform distribution of monitoring stations where $\rho(\ell)$ exhibits multi-order-of-magnitude variations across $\mathcal{S}$. Certain regions are overrepresented in training data relative to their spatial extent, while others remain critically undersampled.

### Distributional Imbalance

Spatial heterogeneity in the joint distribution of covariates and target variables. For stations $i$ and $j$ at different locations:

- $P(\mathbf{x}_{i,t}) \neq P(\mathbf{x}_{j,t})$ (different marginal distributions)
- $P(y_{i,t}|\mathbf{x}_{i,t}) \neq P(y_{j,t}|\mathbf{x}_{j,t})$ (different conditional distributions)

These distributions share the same functional form but exhibit location-dependent parameters.

---

## TwoStageModel: Accuracy Surface with Physical Constraints

### Motivation

Per-location $R^2$ provides empirical accuracy only at monitored sites. Accuracy at unobserved locations must be estimated. Spatial interpolation methods achieve only modest correlation because they cannot account for sample size and density effects.

### Stage 1: Density GAM

For every location, model $R^2$ as a function of station density $\rho$ and sample size $n$:

$$R^2_{\text{GAM}}(\rho, n) = s_1(\log_{10}(n)) + s_2(\rho) + s_3(\log_{10}(n), \rho) + \varepsilon$$

where:

- $s_1(\cdot)$ and $s_2(\cdot)$ are **monotonic increasing** spline functions
- $s_3(\log_{10}(n), \rho)$ is a smooth interaction term (tensor-product spline)

**Monotonicity constraints** encode the physical assumption that accuracy increases with both sample size and density:

$$\frac{\partial s_1}{\partial \log_{10}(n)} \geq 0, \quad \frac{\partial s_2}{\partial \rho} \geq 0$$

### Stage 2: Spatial SVM

Compute residuals from Stage 1:

$$r = R^2 - \hat{R}^2_{\text{GAM}}$$

Train a Support Vector Regression (SVR) to predict location-specific deviations:

$$\hat{r} = h(n, \ell)$$

where $\ell = (\text{lon}, \text{lat})$ captures spatial patterns not explained by density alone.

### Final Prediction

The combined prediction:

$$\hat{R}^2 = g(\rho, n) + h(n, \ell) = \hat{R}^2_{\text{GAM}} + \hat{r}_{\text{SVM}}$$

This decomposition separates:

1. **Interpretable density-accuracy relationship** (Stage 1)
2. **Spatial residual patterns** (Stage 2)

---

## Key Insights

| Component | Captures | Physical Meaning |
|-----------|----------|------------------|
| Stage 1 GAM | Global density effect | More data → higher accuracy |
| Stage 2 SVM | Local spatial patterns | Location-specific factors |
| Combined | Full accuracy surface | Predict performance anywhere |

### Predictive Performance

- **Unseen data conditions**: correlation ≈ 0.95
- **Unseen spatial locations**: correlation ≈ 0.60
- **Improvement over interpolation**: 89% (conditions), 46% (locations)

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

