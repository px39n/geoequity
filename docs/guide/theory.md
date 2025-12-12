# Theory & Problem Formulation

## Problem Statement

Consider a spatiotemporal process observed at $n$ monitoring stations $\ell_1, \ell_2, \ldots, \ell_n \in \mathcal{S} \subset \mathbb{R}^2$ over time periods $t \in \{1, 2, \ldots, T\}$. At each spatiotemporal location $(\ell, t)$, we observe the input-output pair $(y_t(\ell), \mathbf{x}_t(\ell))$, where:

- $y_t(\ell) \in \mathbb{R}$ is the target variable
- $\mathbf{x}_t(\ell) \in \mathbb{R}^d$ is a $d$-dimensional covariate vector

Our objective is to estimate a function $f$ parameterized by $\theta$ such that:

$$y_t(\ell) = f(\mathbf{x}_t(\ell); \theta) + \epsilon_t(\ell)$$

where $\epsilon_t(\ell)$ is Gaussian white noise with $\mathbb{E}[\epsilon_t(\ell)] = 0$ and $\mathrm{Var}(\epsilon_t(\ell)) = \sigma^2$.

## The Challenge: Spatial Inequity

The fundamental challenge arises from **spatiotemporal heterogeneity** in the data distribution. Since observations are non-uniformly distributed across the joint covariate-space-time domain, parameter estimates $\hat{\theta}$ become biased towards data-dense regions, leading to poor generalization in sparse areas.

---

## Global vs. Location-wise Accuracy

### The Problem with Global Metrics

Most spatiotemporal studies report a **single global metric** (e.g., overall $R^2$) under random cross-validation. This approach has critical limitations:

| Approach | What it measures | Problem |
|----------|------------------|---------|
| **Global $R^2$** | Average performance across all samples | Masks spatial variation in accuracy |
| **Time-wise CV** | Temporal generalization | Shares spatial neighbors between train/test |
| **Random CV** | General prediction skill | Inflates apparent skill via spatial autocorrelation |

### Location-wise Accuracy

A more informative approach measures accuracy **at each location separately**, then aggregates spatially:

$$R^2(\ell) = 1 - \frac{\sum_t (y_t(\ell) - \hat{y}_t(\ell))^2}{\sum_t (y_t(\ell) - \bar{y}(\ell))^2}$$

This per-location $R^2$ reveals how model performance varies across space. However, it only provides empirical accuracy at **monitored sites**—accuracy at unobserved locations must be estimated.

### Why Spatial Interpolation Fails

A common approach is to spatially interpolate per-location $R^2$ values based on geographic proximity. However, this assumes:

> "Model performance varies smoothly with location"

This assumption **ignores how data availability affects model behavior**. In reality, accuracy depends on:

1. **Station density** at each location
2. **Sample size** used for training
3. **Location-specific factors** (terrain, data quality, etc.)

**GeoEquity's solution**: Predict accuracy based on **data sampling conditions** (density, sample size), not just spatial proximity.

---

## Local Station Density

For any spatial location $\ell \in \mathcal{S}$, the **local weighted station density** is:

$$\rho(\ell) = \frac{1}{\pi r^2} \sum_{j \in \mathcal{N}(\ell)} w_j$$

where:

- $w_j = \frac{r - \min(d_j, r)}{r}$ is the distance-based weight
- $d_j$ is the Haversine distance between location $\ell$ and station $j$
- $\mathcal{N}(\ell) = \{j \mid d_j \leq r\}$ is the set of stations within radius $r$
- $\pi r^2$ normalizes by the search area

### Haversine Distance

The distance $d_{ij}$ between two geographic points:

$$d_{ij} = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_j - \phi_i}{2}\right) + \cos(\phi_i)\cos(\phi_j)\sin^2\left(\frac{\lambda_j - \lambda_i}{2}\right)}\right)$$

where $R = 6371$ km (Earth's radius), and $(\phi, \lambda)$ are latitude and longitude in radians.

---

## Types of Imbalance

### Spatial Imbalance

Non-uniform distribution of monitoring stations where $\rho(\ell)$ exhibits **multi-order-of-magnitude variations** across $\mathcal{S}$. Certain regions are overrepresented in training data, while others remain critically undersampled.

### Distributional Imbalance

Spatial heterogeneity in the joint distribution of covariates and target variables. For stations $i$ and $j$ at different locations:

- $P(\mathbf{x}_{i,t}) \neq P(\mathbf{x}_{j,t})$ (different marginal distributions)
- $P(y_{i,t}|\mathbf{x}_{i,t}) \neq P(y_{j,t}|\mathbf{x}_{j,t})$ (different conditional distributions)

---

## TwoStageModel: Accuracy Surface with Physical Constraints

### Motivation

Given per-location $R^2$ values at monitored sites, we want to:

1. **Explain** how accuracy depends on density and sample size
2. **Predict** accuracy at unobserved locations
3. **Forecast** model performance under different data conditions

### Stage 1: Density GAM

Model location-wise $R^2$ as a function of station density $\rho$ and sample size $n$:

$$R^2_{\mathrm{GAM}}(\rho, n) = s_1(\log_{10}(n)) + s_2(\rho) + s_3(\log_{10}(n), \rho) + \varepsilon$$

where:

- $s_1(\cdot)$ and $s_2(\cdot)$ are **monotonic increasing** spline functions
- $s_3(\cdot, \cdot)$ is a smooth interaction term (tensor-product spline)

**Monotonicity constraints** encode the physical assumption:

$$\frac{\partial s_1}{\partial \log_{10}(n)} \geq 0, \quad \frac{\partial s_2}{\partial \rho} \geq 0$$

*More data and higher density must yield higher accuracy.*

### Stage 2: Spatial SVM

Compute residuals from Stage 1:

$$r = R^2 - \hat{R}^2_{\mathrm{GAM}}$$

Train a Support Vector Regression (SVR) to predict location-specific deviations:

$$\hat{r} = h(n, \ell)$$

where $\ell = (\mathrm{lon}, \mathrm{lat})$ captures spatial patterns not explained by density alone.

### Final Prediction

$$\hat{R}^2 = \underbrace{g(\rho, n)}_{\text{Density effect}} + \underbrace{h(n, \ell)}_{\text{Spatial residual}}$$

---

## What TwoStageModel Predicts

| Scenario | Input | Output |
|----------|-------|--------|
| **Unseen location** | Coordinates + density | Predicted $R^2$ |
| **Unseen data condition** | New sample size + density | Predicted $R^2$ |
| **Decision support** | Hypothetical deployment | Expected performance |

### Predictive Performance

| Scenario | Correlation | vs. Interpolation |
|----------|-------------|-------------------|
| Unseen data conditions | ~0.95 | +89% improvement |
| Unseen spatial locations | ~0.60 | +46% improvement |

---

## Key Insight

Model performance inequity becomes **obscured** when using:

- Global metrics that average across all samples
- Interpolated accuracy based solely on geographic proximity

GeoEquity reveals this hidden inequity by modeling accuracy as a function of **data availability**, enabling:

- Fair assessment of model performance across regions
- Identification of under-served areas needing more observations
- Prediction of performance before deploying models

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
