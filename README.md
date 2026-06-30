# Equity Risk Modelling — Regime-Aware PCA, Heavy Tails, EWMA, and Kalman Factor Premia

Out-of-sample **equity covariance / density forecasting** for a panel of **24 US large-cap equities**.

The project started as a regime-aware PCA risk-modelling study and evolved into a broader empirical model-selection exercise: VIX/HMM regimes, eigenstructure states, shrinkage EWMA covariance, Gaussian vs Student-t scoring, VaR/ES tail validation, and an optional Kalman-filtered factor-premia layer.

---

## TL;DR — final run

The robust out-of-sample findings are:

1. **Heavy tails matter most.** A Student-t likelihood with \(\hat\nu \approx 5\) materially improves one-day density forecasts versus Gaussian scoring.
2. **Long-memory shrinkage EWMA is the best covariance model.** The fine grid selects memory mainly around \(\lambda = 0.99\)–\(0.9925\), with `ewma_0.9925` the best fixed-lambda model in the final run.
3. **Regime models are informative but not the robust winner.** The VIX/HMM calm-stress mixture beats the static global baseline on the fixed static OOS split, but does not survive as a clean walk-forward/live claim.
4. **Eigenstructure/adaptive-memory regimes are dominated after controls.** Apparent adaptive gains are explained by memory length rather than true eigenstructure timing.
5. **Kalman-filtered PCA factor premia do not add one-day density value.** The expected-return overlay is tested and kept as a negative/diagnostic Equity QR result: at this horizon, the signal is in covariance and tails, not daily mean forecasts.

The final model is therefore best read as a **risk / density-forecasting engine**, not an alpha model.

---

## Repository contents

```text
.
├── README.md
└── equity-risk-modelling-regime-aware-pca.ipynb
```

Raw market data are not included. The notebook can run with `yfinance` or be adapted to local CSV inputs.

---

## Universe and sample

- **Universe:** 24 US large-cap equities  
  `AAPL, MSFT, NVDA, ORCL, GOOGL, META, AMZN, HD, WMT, PG, KO, JPM, BRK-B, V, LLY, JNJ, UNH, XOM, CVX, CAT, RTX, LIN, NEE, AMT`
- **Regime proxy:** VIX (`^VIX`)
- **Low-rank factors:** \(k=3\)
- **Sample requested:** from 2010-01-01 onward
- **Effective common sample:** starts on **2012-05-21**, due to the intersection of histories across the stock panel and VIX
- **Test window:** **2022-01-01 → 2026-03-31**, consistently enforced in the final run

Absolute NLL levels are not comparable across all notebook sections because earlier Part A reports Gaussian NLL up to an additive constant, while later sections use full normalising constants and trailing-window forecasts. Rankings and deltas should be read **within** each section.

---

## Part A — VIX/HMM regime-aware PCA

The original study compares:

- global/static low-rank PCA covariance;
- HMM-informed piecewise PCA;
- a 2-state VIX/HMM calm-stress covariance mixture;
- a Vol-mix / Corr-fixed challenger.

### Static OOS performance

| Model | Mean NLL | Δ vs global |
|---|---:|---:|
| HMM Mixture (calm/stress) | **−175.978** | **−0.370** |
| Vol-mix Corr-fixed | −175.655 | −0.048 |
| Global (static) | −175.607 | 0.000 |
| Piecewise PCA (HMM thresholds) | −175.041 | +0.566 |

### Static block-bootstrap robustness

| Comparison | Bootstrap mean ΔNLL | P(beats global) |
|---|---:|---:|
| HMM Mixture − global | −0.363 | **1.00** |
| Vol-mix − global | −0.040 | 0.673 |
| Piecewise − global | +0.598 | 0.05 |

### Walk-forward recalibration

| Model | Mean NLL |
|---|---:|
| Global rolling | **−179.422** |
| HMM-mixture rolling | −179.398 |
| Vol-mix rolling | −179.329 |

`P(rolling HMM beats rolling global) = 0.133`.

**Reading:** the HMM static edge is real on the fixed split, but not robust as a walk-forward/live model-selection result.

---

## Part B — heavy tails, EWMA, and eigenstructure tests

The extended investigation tests progressively fairer benchmarks and mechanisms:

- Gaussian vs Student-t likelihood;
- fixed-lambda shrinkage EWMA covariance;
- eigenstructure regimes via absorption ratio and PCA-subspace rotation;
- adaptive-memory covariance models;
- Marchenko-Pastur / RMT diagnostics;
- confound controls against fixed-lambda EWMA.

### Fine-grid EWMA / Student-t result

The final fine grid tests:

```python
[0.90, 0.93, 0.95, 0.97, 0.98, 0.985,
 0.990, 0.9925, 0.995, 0.9975, 0.999]
```

Selected lambda frequencies in the final run:

| selected λ | Frequency |
|---:|---:|
| 0.9850 | 0.098 |
| 0.9900 | 0.333 |
| 0.9925 | 0.569 |

Student-t ranking:

| Model | NLL_t | Δ vs global |
|---|---:|---:|
| `ewma_0.9925` | **−70.7799** | **−0.5929** |
| `ewma_0.99` | −70.7628 | −0.5758 |
| `ewma_selected` | −70.7543 | −0.5673 |
| `ewma_0.995` | −70.7504 | −0.5634 |
| `global` | −70.1870 | 0.0000 |

**Reading:** the best memory is long but not static; \(0.99\)–\(0.9925\) is the stable region, while \(0.999\) drifts too close to the static limit.

---

## Market-risk capstone — VaR / ES backtest

The final risk model is also evaluated on equal-weight portfolio VaR/ES at 1% and 2.5%, comparing Gaussian vs Student-t.

Key 1% results:

| Model | Likelihood | Hit rate | Kupiec p_uc | ES hit ratio |
|---|---|---:|---:|---:|
| `ewma_selected` | Gaussian | 1.32% | 0.323 | 1.177 |
| `ewma_selected` | Student-t | **1.03%** | **0.912** | **0.948** |
| `ewma_0.9925` | Gaussian | 1.32% | 0.323 | 1.196 |
| `ewma_0.9925` | Student-t | **1.13%** | **0.681** | **0.940** |

**Reading:** Student-t materially improves 1% tail calibration and ES severity. The 2.5% conditional-coverage diagnostics are less clean, so the conservative claim is tail-risk improvement rather than perfect tail validation.

---

## Section 21 — Kalman-filtered PCA factor premia

The optional Equity QR extension tests whether a low-dimensional expected-return layer adds value beyond the final covariance/tail-risk engine.

Method:

- extract the top \(k=3\) PCA factor returns inside each trailing calibration window;
- estimate latent factor expected returns with a diagonal Kalman filter;
- select \(\phi\), process-noise scale \(q\), and mean-shrink on an internal validation slice;
- map filtered factor premia back to asset-level expected returns;
- compare Kalman mean overlay versus the same covariance model without the overlay.

Final result:

| Base covariance | NLL_t Kalman | NLL_t base | Δ Kalman − base | P(Kalman beats base) |
|---|---:|---:|---:|---:|
| `ewma_0.9925` | −70.77884 | **−70.77989** | +0.00105 | 0.220 |
| `ewma_selected` | −70.75321 | **−70.75430** | +0.00109 | 0.213 |

**Reading:** the Kalman mean layer does **not** add one-day density-forecast value. This is a useful negative result for Equity QR: on this panel and horizon, expected-return dynamics are too weak/noisy to improve OOS density scoring beyond the covariance/tail engine.

---

## Methodology

### Data
Daily adjusted-close returns for 24 US large-cap equities and daily VIX proxy.

### Likelihoods
Gaussian and multivariate Student-t likelihoods are evaluated with full normalising constants in the final walk-forward sections. The Student-t scale matrix is set so that the covariance target is comparable to the Gaussian covariance target.

### Covariance models
- Global/static shrinkage covariance.
- Low-rank PCA covariance models.
- VIX/HMM calm-stress covariance mixtures.
- Eigenstructure-state covariance variants.
- Shrinkage EWMA covariance with validation-selected and fixed lambda grids.

### Evaluation
- Mean OOS NLL.
- Moving-block bootstrap of daily NLL differences.
- Walk-forward monthly refits on trailing 5-year windows.
- VaR/ES coverage and exceedance-severity diagnostics.
- Kalman mean overlay tested against identical covariance/tail baselines.

---

## Limitations

- Research notebook, not a production library.
- Single 24-stock US large-cap panel, survivorship-selected using current constituents.
- One-day horizon only.
- Single post-2022 test path.
- No raw data snapshot included.
- Regime/eigenstructure conclusions are specific to this equity panel; multi-asset universes may behave differently.
- Kalman factor premia are tested only as one-day density mean overlays, not as standalone alpha or allocation signals.

---

## Resume bullets

### Market Risk Quant / Risk Strats

```latex
\item \href{https://github.com/svr-L/equity-risk-modelling-regime-aware-pca}{Built a 24-stock US equity risk-forecasting framework benchmarking PCA/HMM regimes, eigenstructure states, and shrinkage EWMA covariance models under Gaussian vs Student-$t$ density scoring; found that heavy tails ($\hat{\nu}\approx5$) and long-memory EWMA ($\lambda\approx0.9925$) dominated regime variants, with moving-block bootstrap and VaR/ES backtests supporting the final risk model} \href{https://github.com/svr-L/equity-risk-modelling-regime-aware-pca}{\textbf{(repo)}};
```

### Equity QR / portfolio research

```latex
\item \href{https://github.com/svr-L/equity-risk-modelling-regime-aware-pca}{Developed an empirical US equity risk-model selection framework for 24 large-cap stocks, comparing PCA factor risk models, VIX/HMM regimes, eigenstructure states, shrinkage EWMA covariance forecasts, Student-$t$ scoring, and Kalman-filtered factor premia; showed that tail modelling and long-memory covariance updates drive OOS risk performance more robustly than regime or one-day mean-forecast variants} \href{https://github.com/svr-L/equity-risk-modelling-regime-aware-pca}{\textbf{(repo)}};
```

### Short general quant version

```latex
\item \href{https://github.com/svr-L/equity-risk-modelling-regime-aware-pca}{Built a 24-stock US equity risk-forecasting framework comparing PCA, VIX/HMM regimes, eigenstructure states, EWMA covariance, Student-$t$ scoring, and Kalman factor premia; heavy tails and long-memory covariance updates dominated regime and one-day mean-forecast variants, with bootstrap and VaR/ES tests supporting the final model} \href{https://github.com/svr-L/equity-risk-modelling-regime-aware-pca}{\textbf{(repo)}};
```

---

## Roadmap

- Re-run on multi-asset universes where correlation sign/shape regimes should matter more.
- Test multi-day horizons, where regime persistence and factor premia may be more relevant.
- Add point-in-time / survivorship-free universes.
- Connect the final risk engine to portfolio construction: volatility targeting, min-variance, risk parity, and mean-variance allocation with turnover and transaction-cost diagnostics.
- Refactor notebook logic into reusable Python modules.
