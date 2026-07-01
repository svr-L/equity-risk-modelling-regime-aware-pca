# Equity Risk Modelling — Regime-Aware PCA, Heavy Tails, and Eigenstructure Regimes

Out-of-sample **covariance / density forecasting** for a panel of **24 US large-cap equities**.

The project began as a study of whether **regime-aware covariance forecasts** beat a static low-rank
PCA baseline, and then extended into a systematic test of *what actually drives out-of-sample density
forecast quality* on this panel — including a heavy-tailed likelihood, well-conditioned conditional
covariance benchmarks, and regimes defined directly from the covariance eigenstructure (no HMM).

---

## TL;DR — headline finding

After testing regime conditioning in many forms and against progressively fairer benchmarks, the only
**robust out-of-sample gains** come from:

1. a **Student-t likelihood** (estimated tail index \(\nu \approx 5\)) — by far the dominant effect; and
2. a **long-memory shrinkage EWMA covariance** (\(\lambda \approx 0.9925\), the interior optimum of a fine
   grid), which beats the static global covariance by \(\approx -0.593\) mean NLL.

The Student-t gain is confirmed **economically** (Part C): in a VaR/ES backtest the Gaussian model **fails**
its coverage tests (too many 1% tail breaches; Kupiec and Christoffersen reject; ES underestimates tail loss),
while the Student-t model is **calibrated** and passes — the difference between a VaR model a risk function
can sign off on and one it cannot.

**Every regime / eigenstructure variant is dominated** once the benchmarks are well-conditioned and the
likelihood models the tails:

- the original VIX/HMM calm–stress mixture shows a **static-OOS** edge (\(\Delta\)NLL \(=-0.370\),
  block-bootstrap \(P(\text{win})=1.00\)) that **does not survive walk-forward recalibration**;
- a piecewise PCA split, absorption-ratio regime buckets, and a rotation-driven adaptive-memory model are
  all dominated;
- the adaptive-memory model *appeared* to win, but a **direct bootstrap control proves its edge is just a
  longer covariance memory** (it loses to a fixed \(\lambda=0.99\) EWMA with \(P=0.000\)), and the adaptive
  forgetting actually *hurts* on this stable panel.

This is reported as a feature, not a disappointment: the project demonstrates the judgment to separate an
in-sample edge from a genuine out-of-sample one, to build fair benchmarks, and to isolate a confound rather
than claim a fragile positive.

---

## Project structure: three parts

- **Part A (original study, V2):** VIX-driven regime-aware low-rank PCA risk model — global baseline,
  HMM-informed piecewise PCA, a 2-state Gaussian HMM calm/stress covariance mixture, and a Vol-mix /
  Corr-fixed challenger; scored by Gaussian OOS NLL with moving-block bootstrap and walk-forward refit.
- **Part B (extended investigation):** heavy-tailed likelihood, fair EWMA benchmarks, eigenstructure-defined
  regimes (no HMM), Random-Matrix-Theory cleaning of the rotation signal, and a confound control.
- **Part C (economic capstone):** VaR / ES backtest of the final model, Gaussian vs Student-t, with Kupiec
  and Christoffersen coverage tests and an ES calibration check.

The notebook contains all three parts in order (Sections 0–11 = Part A; 12–17 = Part B; 18–20 = Part C).

---

## Universe & sample

- **Universe:** 24 US large-cap equities
  (`AAPL, MSFT, NVDA, ORCL, GOOGL, META, AMZN, HD, WMT, PG, KO, JPM, BRK-B, V, LLY, JNJ, UNH, XOM, CVX,
  CAT, RTX, LIN, NEE, AMT`).
- **Regime proxy:** VIX (`^VIX`).
- **Low-rank factors:** \(k = 3\); covariance with Ledoit–Wolf shrinkage.
- **Sample:** data requested from 2010-01-01; the effective common sample (intersection of all histories +
  IV proxy) starts **2012-05-21** (bound by META's 2012 IPO). Train 2010–2021, test 2022 onward.
  Latest run: `ret_train` 2421×24, `ret_test` 1064×24.

> **Note on absolute NLL levels.** Part A reports Gaussian NLL *up to an additive constant* (\(\approx-175\)).
> Part B uses the *full* normalising constants (so Gaussian and Student-t are comparable) and a 5-year
> trailing window (\(\approx-70\)). Absolute levels are therefore **not comparable across the two parts** —
> only rankings/deltas *within* a section are meaningful.

---

## Part A — original regime-aware PCA study (V2)

### Static out-of-sample performance (Gaussian NLL, up to constant)

| Model | Mean NLL | Δ vs global |
|---|---:|---:|
| HMM Mixture (calm/stress) | **−175.978** | **−0.370** |
| Vol-mix Corr-fixed | −175.655 | −0.048 |
| Global (static) | −175.607 | 0.000 |
| Piecewise PCA (HMM thresholds) | −175.041 | +0.566 |

Model **selection** uses train/validation only (the validation-selected live candidate is Vol-mix
Corr-fixed); the test set is never used for selection. The HMM Mixture is the strongest *realized static*
challenger but is reported as diagnostic-only.

### Block-bootstrap robustness (static test set)

| Comparison | Bootstrap mean ΔNLL | P(beats global) |
|---|---:|---:|
| HMM Mixture − global | −0.363 | **1.00** |
| Vol-mix − global | −0.040 | 0.673 |
| Piecewise − global | +0.598 | 0.05 |

### Walk-forward recalibration (the honest test)

| Model (rolling) | Mean NLL |
|---|---:|
| Global rolling | **−179.422** |
| HMM-mixture rolling | −179.398 |
| Vol-mix rolling | −179.329 |

`P(rolling HMM beats rolling global) = 0.13`; `P(rolling Vol-mix beats global) = 0.05`.

**Reading Part A:** the static-OOS regime edge is strong and bootstrap-robust *on the fixed split*, but it
**evaporates under refitting** — the rolling global benchmark is at least as good. This gap is the central
result: the static "win" is an in-period fit advantage, not a forward-deployable one. Part B explains it.

---

## Part B — extended investigation

### What was tested, why, outcome, interpretation

\(d_t\) = OOS mean NLL vs `global` under **Student-t**, walk-forward; negative = better.

| Idea | Why | Outcome | Interpretation |
|---|---|---|---|
| HMM mixture (VIX) — *Part A* | regime conditioning | static −0.370, P=1.00; **rolling** P(win)=0.13 | in-sample edge, dies under refit |
| Piecewise PCA (HMM thresholds) | hard stress split | +0.57, P=0.05 | dominated (hard splits unstable) |
| Rotation → HMM input | eigen signal in the HMM | +1.24 | worse than global |
| Absorption → HMM input | eigen signal in the HMM | +0.61 | worse than global |
| **Student-t vs Gaussian** | Gaussian mis-prices tails | **t−gauss ≈ −2.5**, ν̂≈5, robust on all models | **the real, dominant gain** (≈10× any covariance effect) |
| EWMA vol / corr-fixed | continuous "vol-mix" | +0.20 | freezing the correlation discards the signal |
| EWMA full / shrunk, λ=0.97 | conditional covariance | −0.18 | beats global; edge **compresses under t** (−0.48 → −0.15) |
| **EWMA fixed λ (shrunk), fine grid** | longer memory | **−0.593 at λ≈0.9925 (best covariance model)** | mild recency on a stable panel; interior optimum |
| Absorption buckets — **Model A** | eigenrank regimes, no HMM | +0.10 / +0.88, P≈0, unstable | dominated + unstable |
| Adaptive-λ (rotation) — **Model B** | eigenrank-driven forgetting | −0.36, P(vs global)=1.00 | **looked like a win → confound** |
| RMT / MP cleaning of detector | downweight noisy eigenvectors | 30d window ≈ 1 signal mode | would blind the detector here → not adopted |
| **Confound control** | mechanism vs memory length | **P(eig_lambda beats fixed-0.99) = 0.000** | Model B's edge was *only* longer memory; adaptation *hurts* |

### The unified walk-forward grid (real panel, full-constant NLL)

| Model | NLL_gauss | NLL_t | t − gauss | d_t vs global | P(beats global, t) |
|---|---:|---:|---:|---:|---:|
| `ewma_0.99` (shrunk) | −68.417 | **−70.609** | −2.19 | **−0.598** | 1.00 |
| `eig_lambda` (Model B) | −68.053 | −70.374 | −2.32 | −0.363 | 1.00 |
| `ewma_0.97` (shrunk) | −67.735 | −70.196 | −2.46 | −0.185 | ~0.99 |
| `global` (static) | −67.516 | −70.011 | −2.50 | 0.000 | — |
| `ewma_0.95` (shrunk) | −66.817 | −69.746 | −2.93 | +0.264 | low |
| `eigrank_buckets` (Model A) | — | — | — | +0.88 | ~0.00 |

`nu_hat` median = **5**. Every model improves by ~2.2–2.9 NLL when scored under Student-t instead of Gaussian.

**Confound control (the decisive experiment):**

- `P(eig_lambda beats best fixed-λ = ewma_0.99)` = **0.000**
- `P(eig_lambda beats ewma_0.97)` = 1.000
- `P(eig_lambda beats global)` = 1.000

i.e. Model B beats the *0.97* benchmark only because it sits mostly at 0.99; against a plain fixed-0.99 EWMA
it loses outright, and its adaptive dips to faster memory actively raise NLL.

### MP / RMT diagnostic (why RMT cleaning does not help here)

| Detector window | MP edge \(\lambda_+\) | median #signal modes |
|---|---:|---:|
| short, 30d | 3.59 | **1** (market only) |
| long, 150d | 1.96 | 2 |

On 30-day windows only the market mode is reliably above the noise floor, so RMT/MP-edge weighting tracks
essentially the market mode and is **blind to the sub-dominant sector/style rotation** Model B would need —
hence `plain` is the correct detector and RMT is not adopted.

### Synthetic mechanism validation (Section 14, runs offline)

The mechanisms are validated on data with known ground truth, to show that the negative real-data result is
a *panel that lacks the relevant structure*, not a broken method:

- **Student-t:** recovers \(\hat\nu \approx 5\) on \(\nu=5\) data and cuts NLL by ~3 vs Gaussian; on Gaussian
  data \(\hat\nu\to 40\) and \(t\approx\) Gaussian (no false benefit).
- **Adaptive-λ:** keeps long memory when calm and forgets fast at a genuine abrupt correlation break,
  beating *both* fixed-slow and fixed-fast there.
- **RMT detector:** improves signal-to-noise only when few modes are signal (noise-dominated regime); it
  *hurts* when the rotating modes are strong.

---

## Part C — VaR / ES economic capstone

NLL says the Student-t fits better; this part shows the gain is *economically* real, at the level a risk
function cares about. The final-model covariance is projected to a portfolio loss distribution (equal-weight
and minimum-variance) and Value-at-Risk / Expected Shortfall are backtested with formal coverage tests.

### Synthetic validation (Section 18, runs offline)

Equal-weight portfolio, trailing-window covariance + estimated \(\nu\); Gaussian vs Student-t VaR.

| Data | α | breach % G / T (nominal) | Kupiec p G / T | Christoffersen cc p G / T | ES ratio G / T |
|---|---|---|---|---|---|
| fat-tailed (ν=5) | 1% | 1.69 / 1.20 (1%) | **0.000** / 0.267 | **0.001** / 0.336 | 1.11 / 0.95 |
| fat-tailed (ν=5) | 5% | 4.12 / 4.62 (5%) | **0.018** / 0.308 | **0.048** / 0.431 | 1.14 / 1.02 |
| Gaussian (control) | 1% | 0.86 / 0.80 (1%) | 0.416 / 0.235 | 0.564 / 0.401 | 1.02 / 1.00 |
| Gaussian (control) | 5% | 4.31 / 4.31 (5%) | 0.064 / 0.064 | 0.166 / 0.166 | 0.99 / 0.98 |

(Kupiec / Christoffersen p > 0.05 = coverage not rejected; ES ratio ≈ 1 = calibrated, > 1 = tail loss
underestimated. Bold = rejected.)

**Reading Part C.** On fat-tailed data the Gaussian VaR is **rejected** by both coverage tests at 1% (it
produces ~1.7× the nominal breaches) and its ES underestimates the realised tail loss; the Student-t VaR is
**calibrated** and passes. On Gaussian data both pass — no false advantage. Section 19 applies the same
backtest to the final model on the real panel (runs when the notebook is executed with data). The economic
statement: the Student-t is the difference between a VaR model that fails its regulatory-style coverage tests
and one that passes them.

---

## Final model

**Student-t likelihood (\(\nu \approx 5\)) + long-memory shrinkage EWMA covariance (\(\lambda \approx 0.9925\)),
shrunk toward a stable anchor.** No regime switching, no eigenstructure conditioning. The gain comes from
heavy tails and a gently recency-weighted covariance — not from regimes.

*Finalisation note:* the coarse sweep {0.95, 0.97, 0.99} put the optimum at the \(\lambda=0.99\) boundary; a
finer grid resolves it to an **interior optimum at \(\lambda \approx 0.9925\)** (\(\Delta\)NLL \(\approx -0.593\)
vs global), short of the \(\lambda \to 1\) static limit.

---

## Why "regimes don't help here" is not "the method is broken"

The mechanisms work in isolation (Section 14). The 24 US large-cap panel is dominated by a single market
factor whose **correlation *shape* is stable through time**; market stress raises the correlation *level*
(absorption ratio), it does not *rotate* the eigenvectors, and the MP diagnostic confirms only the market
mode is reliably above the noise floor in short windows. A regime model that conditions covariance *shape*
on an eigenstructure state has little to act on **on this panel**.

Where regime-awareness would plausibly pay off, and the natural next projects:

- **Multi-asset universes**, where correlations change sign (e.g. stock–bond correlation flipped from
  negative to positive in 2022) — a regime of *correlation*, which this single-factor equity panel does not
  exhibit.
- A **return / allocation objective** rather than covariance density — the literature finds regime-switching
  matters far more for asset allocation than for covariance.
- **Multi-day horizons**, where regime persistence is visible (it is nearly invisible at 1 day) and for
  scenario generation / stress testing, where "we are in a stress regime with \(p=0.8\)" is interpretable and
  actionable even when it does not improve a point density forecast.

---

## Methodology

### Data
- Daily adjusted-close returns for 24 US large-caps; daily VIX proxy. Explicit train / validation / test
  split. Data via `yfinance` or local CSV.

### State variable (Part A)
- Past-looking VIX state \(z_t\): log transform → smoothing → EWMA standardisation.

### Likelihoods (Part B)
- **Gaussian** and **multivariate Student-t**, both with *full* normalising constants. The \(t\) scale matrix
  is set to \(\Sigma\,(\nu-2)/\nu\) so that \(\mathrm{Cov}(x)=\Sigma\) — the two share the same covariance
  target and differ only in tail shape, making the comparison fair. \(\nu\) is estimated by profile ML on
  each trailing window.

### Covariance models
- **Global / static:** Ledoit–Wolf shrinkage covariance (and low-rank PCA, \(k=3\)).
- **Fixed-λ EWMA (shrunk):** \(\Sigma_t = \lambda\Sigma_{t-1} + (1-\lambda)r_{t-1}r_{t-1}^\top\), then shrunk
  toward the trailing-window covariance for conditioning; swept over \(\lambda \in \{0.95, 0.97, 0.99\}\).
- **HMM calm/stress mixture (Part A):** \(\Sigma_{t|t-1} = (1-p_{t-1})\Sigma_{\text{calm}} + p_{t-1}\Sigma_{\text{stress}}\),
  with \(p_{t-1}\) the lagged filtered HMM stress probability.

### Eigenstructure signals & regime models (Part B, no HMM)
- **Absorption ratio:** top-\(k\) share of correlation variance (Kritzman et al. 2011) — the intensity axis.
- **PCA-subspace rotation:** \(1 - \) aligned-subspace similarity between a *recent* (30d) and *long-run*
  (150d) top-\(k\) subspace, via projector cosine. Sign- and degeneracy-robust; responsive by construction.
- **Model A — eigenrank buckets:** low/mid/high buckets on absorption (tertiles recomputed per window), one
  shrunk covariance per bucket.
- **Model B — adaptive-λ:** \(\lambda_t = \lambda_{\text{slow}} - (\lambda_{\text{slow}}-\lambda_{\text{fast}})
  \,\sigma(\beta(s_{t-1}-c))\) driven by the rotation signal \(s\) — long memory when calm, fast when the
  structure rotates. \((\lambda_{\text{slow}},\lambda_{\text{fast}},c)\) selected on a validation slice.

### Random Matrix Theory
- Marchenko–Pastur upper edge \(\lambda_+ = (1+\sqrt{N/T})^2\) used to (i) weight eigenvectors by their
  distance above the noise band (Laloux–Bouchaud noise-dressing) in the rotation detector, and (ii) as a
  structural-rank diagnostic (number of eigenvalues above the edge).

### Evaluation
- **Primary:** mean OOS NLL (Gaussian and Student-t).
- **Walk-forward:** monthly refit on a trailing 5-year window over the test period; all signals causal/lagged.
- **Uncertainty:** moving-block bootstrap (block 20) of per-day NLL differences.
- **Confound control:** direct block bootstrap of `eig_lambda` vs the best fixed-λ EWMA.
- **Secondary (Part A):** PCA reconstruction \(R^2\) (global 0.469, piecewise 0.466 — descriptive only).

---

## Notebook structure

```
Part A — regime-aware PCA (V2)
  0  Setup            5  HMM on z_t             9  Static OOS evaluation
  1  Configuration    6  Fit models             6.5 Smoothing p_t (sensitivity)
  2  Data loading     7  Static OOS             8  Visual diagnostics
  3  Core functions   8/9 HMM + filtering       9  Block bootstrap (ΔNLL)
  4  State variable                            10  Rolling recalibration (walk-forward)
                                               11  Final notes
Part B — extended investigation
 12  Overview                      15  Unified walk-forward comparison (real panel)
 13  Extended machinery            16  Marchenko–Pastur / RMT diagnostic
 14  Mechanism validation (synthetic)   17  Extended findings & final model
Part C — economic capstone
 18  VaR/ES machinery & validation (synthetic)   20  VaR/ES findings
 19  VaR/ES on the final model (real panel)
```

---

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `equity-risk-modelling-regime-aware-pca.ipynb` and run top to bottom. Section 14 (synthetic validation)
runs without market data; Sections 15–16 use the loaded panel. To A/B the RMT detector, set
`ROT_DETECTOR = "rmt_weighted"` in Section 15.

### Reproducibility
- **Option A — quick start:** run with `yfinance` enabled.
- **Option B — fixed snapshot:** replace the data-loading block with CSV inputs for the price panel and the
  IV proxy. Raw market data are not included in the repo.

---

## Limitations

- A research notebook, not a production library.
- Single universe (24 US large-caps, survivorship-selected — today's large caps), single test path
  (2022→), 1-day horizon, density (NLL) objective. The negative regime result is specific to this setting.
- Absolute NLL levels are not comparable across Part A (up-to-constant) and Part B (full constants).
- Market data pulled externally unless the CSV loader is used; exact numbers vary with download timing.
- The EWMA optimum \(\lambda \approx 0.9925\) is an interior point of the fine grid; on a new sample it should
  still be re-selected on validation rather than fixed.
- The VaR/ES backtest (Part C) demonstrates the Student-t coverage advantage on synthetic data and is wired
  to run on the real panel; the real-panel run uses an unconstrained minimum-variance portfolio (weights can
  be negative) alongside equal-weight.

---

## Roadmap

- Validation-select the final \(\lambda\) (and \(\nu\)) over wider grids. (The VaR / ES coverage backtest —
  Kupiec + Christoffersen — is now included as Part C.)
- Re-run the regime / eigenstructure models on settings where they should pay off: **multi-asset** universes
  (sign-changing correlations), **return/allocation** objectives, and **multi-day** horizons.
- Survivorship-free, point-in-time universes; cross-market validation (UK / EU / JP).
- Refactor notebook logic into reusable modules.
