# When Regimes Do Not Help
### Heavy Tails, Covariance Memory, and Short-Horizon Equity Risk Forecasting

**Author:** Saverio Lauriola
**Repository:** `equity-risk-modelling-regime-aware-pca`
**Status:** Research note — companion to the reproducible notebook and to the working paper (`working_paper.pdf`)

---

## Abstract

This note studies one-day-ahead density and tail-risk forecasting for a panel of US large-cap equities and asks which modelling ingredients survive strict out-of-sample validation. **Three deliver robust gains, each established against a strong benchmark:** (i) a multivariate Student-$t$ likelihood with estimated tail index $\hat\nu \approx 5$, which dominates every covariance refinement by roughly an order of magnitude and corrects expected-shortfall calibration; (ii) a long-memory shrinkage EWMA covariance update with $\lambda \approx 0.9925$, the best covariance model in a fine grid; and (iii) a reactive portfolio-variance layer (fast EWMA or GARCH) that restores Value-at-Risk conditional coverage where static specifications fail.

Against these, a battery of regime-aware models — VIX/HMM covariance mixtures, eigenstructure-defined regimes, adaptive spectral forgetting, and a Kalman-filtered factor-premia overlay — does **not** improve the final out-of-sample result. A direct bootstrap control is decisive: the apparent edge of adaptive eigenstructure forgetting is fully explained by covariance-memory length, not by regime adaptation. A Marchenko–Pastur diagnostic explains why — on this panel a short estimation window resolves essentially one signal mode, leaving correlation-rotation detectors little to act on.

The contribution is a **model-validation result**: the robust short-horizon signal lies in tail shape and covariance memory, not in regime conditioning or expected-return dynamics — and a candidate model should be judged against the simplest benchmark that could explain its edge.

---

## What survives strict out-of-sample validation

| Ingredient | Role | Survives OOS? | Decisive evidence |
|---|---|:---:|---|
| **Student-$t$ likelihood** ($\hat\nu\approx 5$) | marginal / tail shape, ES | **Yes** (dominant) | $\Delta\mathrm{NLL}\approx -2.3$ to $-2.5$ vs Gaussian; ES ratio $\approx 0.95$ |
| **Long-memory shrinkage EWMA** ($\lambda\approx 0.9925$) | one-day covariance | **Yes** (best cov.) | $\Delta\mathrm{NLL}\approx -0.593$ vs global; interior optimum in fine grid |
| **Reactive portfolio variance** (EWMA-vol / GARCH) | VaR conditional coverage | **Yes** | restores Christoffersen cc at 2.5% ($p_{cc}: 0.003 \to 0.94$) |
| VIX/HMM covariance mixture | regime covariance | No (not live) | static $P(\text{win})=1.00$ but walk-forward $P\approx 0.13$ |
| Eigenstructure regimes / adaptive forgetting | regime covariance | No (confound) | $P(\text{beats fixed }\lambda{=}0.99)=0.000$; edge is memory length |
| Kalman factor-premia overlay | expected-return layer | No (at 1 day) | $\Delta\mathrm{NLL}\approx +0.001$, $P(\text{beat})=0.215$ |

*Negative $\Delta\mathrm{NLL}$ = improvement over the stated baseline. Absolute NLL levels are comparable only **within** a section (same constants and window), never across sections.*

**Effect hierarchy** — the organising principle of the note:

$$\underbrace{\text{tail shape}}_{\approx\,2.3\text{–}2.5\ \mathrm{NLL}} \;\gg\; \underbrace{\text{covariance memory}}_{\approx\,0.6\ \mathrm{NLL}} \;\gg\; \underbrace{\text{regime conditioning}}_{\approx\,0\ \text{after controls}} \;\approx\; \underbrace{\text{expected-return dynamics}}_{\approx\,0\ \text{at 1 day}}$$

---

## 1. Research question

The project asks whether increasingly sophisticated regime-aware equity risk models improve short-horizon out-of-sample density forecasts *once they are compared against fair covariance-memory and tail-shape benchmarks*.

The initial idea was intuitive: market stress should change the covariance structure of equities, and a static covariance matrix should miss that time variation. The first version compared a global low-rank PCA covariance model against regime-aware alternatives driven by a VIX-based state variable. Later extensions made the test stricter by adding heavy-tailed likelihoods, long-memory EWMA benchmarks, eigenstructure-defined regimes, adaptive-memory covariance models, a Kalman-filtered expected-return overlay, and VaR/ES backtests with exceedance-clustering diagnostics. The final question is therefore not whether a regime model can beat a static baseline, but whether regime conditioning remains useful **after controlling for the two effects risk models most often need: tail thickness and covariance memory.**

## 2. Data and forecasting design

The panel contains daily adjusted-close returns for 24 US large-cap equities, with VIX as the external volatility/regime proxy. Data are requested from 2010 onward, but the common sample begins in 2012 (intersection of available histories); the main test window begins in 2022.

The design is deliberately **causal**: models are estimated on training or trailing windows only; state variables and filtered probabilities are lagged before scoring; walk-forward sections refit periodically using only past data; the primary score is out-of-sample negative log-likelihood (NLL); a moving-block bootstrap assesses loss-difference robustness. The target is the one-day predictive distribution of the return vector, so the effort goes into the covariance matrix and the likelihood shape, not expected-return forecasting.

> **Reproducibility.** All results come from the companion notebook on a fixed data vintage; windows, lags, bootstrap block lengths and the $\nu$ grid are specified there. Absolute NLL levels differ between the up-to-constant scoring of §5.4 (window-level, $\approx -175$) and the full-constant five-year-window scoring of the covariance grid ($\approx -70$); only within-section differences are interpreted.

## 3. Model classes

- **Static low-rank PCA (baseline).** $k=3$ factors with shrinkage for numerical stability — captures the dominant common equity-risk structure without a fully unconstrained estimate.
- **VIX/HMM covariance mixtures.** A past-looking VIX state (log-VIX, EWMA-standardised) drives a two-state Gaussian HMM; filtered calm/stress probabilities $p_{t-1}$ mix calm and stress covariances, lagged one day:

$$\Sigma_{t\mid t-1} = (1-p_{t-1})\,\Sigma_{\text{calm}} + p_{t-1}\,\Sigma_{\text{stress}}.$$

  This is not an endogenous latent-state return model — it is an externally driven, interpretable stress-conditioned covariance. Simpler variants (HMM-informed piecewise PCA, vol-mix/correlation-fixed, hard low/mid/high splits) separate probabilistic smoothing from covariance scaling from sample splitting.
- **Eigenstructure-defined regimes.** State derived from the correlation eigenstructure: absorption ratio (top-$k$ share of correlation variance), PCA-subspace rotation between short/long windows, eigenrank buckets, adaptive forgetting driven by spectral rotation. Tests whether regimes defined *from the covariance object* beat external VIX-based regimes.
- **EWMA covariance benchmark.** The key benchmark:

$$\Sigma_t = \lambda\,\Sigma_{t-1} + (1-\lambda)\,r_{t-1} r_{t-1}^{\top},$$

  shrunk toward a stable trailing-window anchor; a wide/fine grid tests $\lambda \in [0.90, 0.999]$.
- **Student-$t$ likelihood.** Scale matrix adjusted so the covariance target matches the Gaussian one, $\mathrm{Cov}(r_t)=\Sigma_t$ (scale $= \Sigma_t(\nu-2)/\nu$). Tail index $\nu$ estimated by profile likelihood on each trailing window; empirical median $\hat\nu \approx 5$.
- **Kalman-filtered factor premia.** A Kalman filter on the first $k=3$ PCA factor returns; filtered premia mapped back to asset-level means and scored against the same covariance/tail model — tests whether an expected-return layer adds one-day value.
- **Reactive portfolio-variance layer.** Density forecasts converted to equal-weight portfolio VaR/ES; exceedance clustering attacked with reactive conditional variance (fast EWMA-vol, GARCH(1,1)-style).

## 4. Evaluation metrics

Primary metric: mean out-of-sample NLL, $\mathrm{NLL}_t = -\log f(r_t \mid \mu_t, \Sigma_t)$ — lower means higher predictive density on the realised vector. Comparisons use the loss difference $\Delta\mathrm{NLL} = \mathrm{NLL}_{\text{model}} - \mathrm{NLL}_{\text{baseline}}$ (negative is better). The risk capstone uses one-day VaR and ES at 1% and 2.5%, via hit rates, Kupiec unconditional coverage, Christoffersen independence and conditional coverage, and the ES realised-loss-to-forecast ratio on exceedance days.

## 5. Results

### 5.1 Heavy tails are the dominant ingredient

Replacing the Gaussian with a Student-$t$ likelihood gives the largest and most stable gain. The median tail index is $\hat\nu \approx 5$, and the Student-$t$ likelihood improves NLL by roughly **2.3–2.5 points** on the main models — an order of magnitude larger than most covariance differences. For one-day US large-cap density forecasting, **tail shape matters more than regime sophistication.**

### 5.2 Long-memory EWMA is the best covariance model

In a fixed-$\lambda$ grid the long-memory EWMA beats the static global covariance and the regime/eigenstructure variants; a finer grid confirms this is not a boundary artefact, with an **interior optimum near $\lambda = 0.9925$**.

| Model | Student-$t$ NLL | $\Delta$ vs global |
|---|---:|---:|
| EWMA $\lambda = 0.9925$ | −70.780 | **−0.593** |
| EWMA $\lambda = 0.9900$ | −70.763 | −0.576 |
| Validation-selected EWMA | −70.754 | −0.567 |
| Global | −70.187 | 0.000 |

The covariance conclusion is not "use a reactive regime switch" but **"use a stable, long-memory update with mild recency weighting."**

### 5.3 Reactive variance restores VaR conditional coverage

With the final EWMA covariance, Student-$t$ tails improve 1% tail calibration and ES severity: the 1% Student-$t$ VaR hit rate is near target and the ES loss-to-forecast ratio is near one (≈ 0.95, against 1.1–1.2 under Gaussian tails). At **2.5%** a sharper picture emerges — exceedances **cluster**, and both Gaussian and Student-$t$ *static* specifications fail Christoffersen conditional coverage. This is a **variance-dynamics problem, not a tail-shape problem.** A reactive conditional-variance layer fixes it under either likelihood:

| Portfolio-variance model (2.5%, Student-$t$) | Christoffersen $p_{cc}$ |
|---|:---:|
| Static 5y window | 0.003 *(fails)* |
| GARCH(1,1) | 0.14 |
| Reactive EWMA-vol ($\lambda = 0.94$) | 0.46 |
| Reactive EWMA-vol ($\lambda = 0.97$) | 0.94 |

Tail shape and reactive variance are **two orthogonal interventions**: the former controls tail magnitude (ES), the latter controls clustering (conditional coverage). *(Power is limited by exceedance count, so the robust signal is static-vs-reactive, not the ranking among reactive models.)*

### 5.4 The regime models, as controlled nulls

**The VIX/HMM mixture works statically but not live.** In static scoring the HMM calm/stress mixture improves materially ($\Delta\mathrm{NLL} = -0.370$; bootstrap $P(\text{win}) = 1.00$). But under walk-forward recalibration the probability that the rolling HMM beats the rolling global benchmark is only $\approx 0.13$. The static edge is evidence of useful regime structure on a fixed split, **not** a deployable real-time advantage.

**Eigenstructure adaptation is a memory-length confound.** The adaptive eigenstructure model beats the global baseline and looks attractive — until a direct bootstrap control against the best fixed-$\lambda$ EWMA dissolves it:

$$P(\text{adaptive eigen-}\lambda \text{ beats fixed }\lambda{=}0.99)=0.000, \quad P(\text{beats }\lambda{=}0.97)=1.000, \quad P(\text{beats global})=1.000.$$

It beats only weaker, faster-memory benchmarks, because it spends most of its time near a slow setting; against a plain long-memory EWMA it **loses**. The driver is covariance-memory length, not regime adaptation. **This single control is the methodological core of the note.**

**Why eigenstructure regimes are weak here.** A Marchenko–Pastur diagnostic shows that a short detector window contains essentially **one** reliable signal mode — the market mode. In a panel dominated by a stable market factor, RMT-cleaned rotation detectors are blind to weaker sector/style rotations, so eigenstructure regime conditioning has little to act on. Not a general claim — this *particular* one-day US large-cap panel lacks enough robust covariance-shape rotation.

**Kalman expected returns add no one-day density value.** $\Delta\mathrm{NLL} \approx +0.001$ with $P(\text{Kalman beats base}) \approx 0.21$ on the final EWMA covariances. At one day, expected-return dynamics are too weak relative to covariance and tail effects. This does not rule out factor-premia modelling for multi-day allocation — it shows the density gains here are not coming from a mean forecast.

## 6. Interpretation

The main lesson is **negative in a useful way.** Several regime models are plausible *ex ante*, and one produces a strong static improvement; but once the tests are made stricter — walk-forward refitting, fair EWMA benchmarks, Student-$t$ likelihoods, direct confound controls — the robust contribution comes from three simpler, **orthogonal** components: a heavy-tailed likelihood (tail magnitude, ES), a long-memory shrinkage EWMA covariance (one-day density), and a reactive portfolio-variance layer (VaR conditional coverage). These compose into the final model; the regime and expected-return channels are the well-documented nulls that justify *not* adding them.

This is a **model-validation result.** It argues against over-interpreting static regime wins and in favour of asking whether a candidate survives the simplest strong benchmark that could explain its edge. For equity QR it is informative — one-day density quality here is driven more by risk dynamics than expected-return dynamics. For market-risk and risk-strategy use it is directly relevant: the final model improves density scoring and tail calibration while avoiding fragile regime overfitting.

## 7. Limitations and next steps

An empirical research note, not a general theorem. The universe is a survivorship-selected panel of current US large-cap equities; the horizon is one day; the test path starts in 2022; the result concerns density and tail-risk scoring, not portfolio alpha. Regime conditioning may matter more in multi-asset universes, at multi-day horizons, or under allocation objectives.

The natural next steps sharpen exactly the boundary this note draws:

- **Cross-asset panel** (equities, bonds, commodities, FX) — introduces a covariance change this equity panel lacks (a sign-flipping stock–bond correlation) and tests the prediction that regime conditioning pays when the *between-regime covariance change is large*. A unifying statement: realised regime value scales as **(size of covariance change) × (quality of the regime classifier)**.
- **Multi-horizon** (HAR-style realised-variance forecasting at 5/10/22 days) — tests that the covariance signal *strengthens* with horizon, in contrast to the one-day null for expected returns.

Together these turn the present single-panel result into a generalisable account of *when* regime conditioning helps — the intended journal-grade synthesis, of which this note is the seed.

## 8. Conclusion

Not that regimes never help — but that, in this one-day US large-cap equity density-forecasting setting, **regime complexity does not survive the strongest simple controls.** The robust model is a Student-$t$ likelihood combined with a long-memory shrinkage EWMA covariance, with a reactive portfolio-variance layer for VaR conditional coverage. VIX/HMM mixtures, eigenstructure regimes, adaptive forgetting, and Kalman-filtered one-day factor premia are useful tests, but they do not improve the final out-of-sample result once tail shape and covariance memory are properly modelled. Less a "regime model wins" story, more a model-validation one: **the useful result is not the most complex model, but the process of identifying the simpler mechanism that actually survives out of sample.**

---

## References

1. Acerbi, C. and Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance*, 26(7), 1487–1503.
2. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
3. Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
4. Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
5. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
6. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
7. Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
8. Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
9. Marčenko, V. A. and Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Mathematics of the USSR-Sbornik*, 1(4), 457–483.
10. McNeil, A. J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques and Tools*. Princeton University Press, revised edition.
11. Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611–1644.
12. J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th edition.
