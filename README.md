# Equity Risk Modelling with Regime-Aware PCA

A research repository on **out-of-sample covariance / density forecasting** for a panel of **24 US large-cap equities**.

## Project summary

This project studies whether **regime-aware covariance forecasts** can improve on a static low-rank PCA baseline when market stress is proxied by an implied-volatility state variable.

The workflow compares:

- a **global / static PCA risk model**
- an **HMM-informed piecewise PCA baseline** built from low / mid / high stress buckets
- a **2-state Gaussian HMM mixture** that blends calm and stress covariance forecasts using lagged filtered probabilities
- an auxiliary **Vol-mix / Corr-fixed** challenger

The main objective is **risk modelling**, not return prediction: the key score is **out-of-sample Gaussian negative log-likelihood (NLL)**.

## Current specification

- **Universe:** 24 US large-cap equities
- **Regime proxy:** **VIX**-based state variable
- **State model:** 2-state Gaussian HMM on the scalar stress indicator \(z_t\)
- **Risk model:** low-rank PCA / shrinkage covariance
- **Primary metric:** mean OOS Gaussian NLL
- **Secondary metric:** PCA reconstruction \(R^2\)

## Sample coverage

The notebook requests data from **2010-01-01** onward, but the effective common return sample is shorter because the analysis uses the intersection of available histories across all names plus the IV proxy.

For the current executed **V2** run:

- **Effective common-sample returns coverage:** **2012-05-21 → 2026-04-17**
- **Training set:** **2421 × 24**
- **Test set:** **1064 × 24**

## Main findings from the current executed V2 notebook

### 1) Validation-selected live candidate vs realized static OOS winner

These are intentionally reported separately:

- **Validation-selected live candidate:** **Vol-mix Corr-fixed**
- **Best realized static OOS model (diagnostic only):** **HMM Mixture**

This is a methodological feature, not a bug: the notebook does **not** use the test set for model selection.

### 2) Static out-of-sample performance

| Model | Mean NLL | Δ vs global |
|---|---:|---:|
| HMM Mixture (calm/stress) | **-175.978** | **-0.370** |
| Vol-mix Corr-fixed (calm/stress) | -175.655 | -0.048 |
| Global (static) | -175.607 | 0.000 |
| Piecewise PCA (HMM-informed thresholds) | -175.041 | +0.566 |

Interpretation:

- the **HMM mixture** is the strongest **static OOS** challenger in the current run;
- the **Vol-mix / Corr-fixed** variant improves only modestly on the global baseline;
- the **HMM-informed piecewise baseline** remains clearly weaker.

### 3) PCA reconstruction

- **Global PCA \(R^2_{total,test}\):** **0.469**
- **Piecewise PCA \(R^2_{total,test}\):** **0.466**

These are descriptive only; the main ranking comes from density forecasting (NLL), not reconstruction fit.

### 4) Block-bootstrap robustness on the static OOS test set

The notebook uses moving-block bootstrap to preserve serial dependence in the test sample.

#### HMM Mixture vs global
- **Bootstrap mean ΔNLL:** **-0.3628**
- **P(HMM beats global):** **1.00**

#### Vol-mix Corr-fixed vs global
- **Bootstrap mean ΔNLL:** **-0.0400**
- **P(Vol-mix beats global):** **0.673**

#### Piecewise PCA vs global
- **Bootstrap mean ΔNLL:** **+0.5975**
- **P(Piecewise beats global):** **0.050**

Interpretation:

- the **static HMM result is strong and robust** in the current broad-US V2 run;
- the **piecewise baseline is dominated**;
- the **Vol-mix** challenger is weaker than HMM but still modestly competitive in static OOS.

### 5) Rolling recalibration (walk-forward)

The notebook also includes a live-style rolling refit exercise.

For the current executed V2 run:

| Model | Mean NLL |
|---|---:|
| Global rolling | **-179.422** |
| HMM-mixture rolling | -179.398 |
| Vol-mix Corr-fixed rolling | -179.329 |

Additional rolling diagnostics:

- **Δ Mean NLL (rolling HMM − rolling global):** **+0.0236**
- **P(rolling HMM beats rolling global):** **0.133**
- **Δ Mean NLL (rolling Vol-mix − rolling global):** **+0.0925**
- **P(rolling Vol-mix beats rolling global):** **0.050**

Interpretation:

- the current notebook supports a strong **static OOS** regime-aware result;
- the **rolling global benchmark remains slightly stronger** in the live-style exercise;
- the project is therefore better described as a **serious research prototype** than as a finalized production-ready live selection pipeline.

## Why this project is interesting

This is not just a “PCA notebook”.

It addresses a real modelling problem that appears often in applied quant work:

- static low-rank structure is interpretable but can miss time variation;
- hard regime splits are simple but often unstable;
- probabilistic state models offer a cleaner way to make covariance forecasts adapt through time.

That makes the project relevant for:

- **market risk quant** roles,
- **risk strats / quantitative risk analytics**,
- **quant research** roles with a risk-modelling angle,
- **systematic / portfolio research** profiles that value regime-aware risk forecasts.

## Methodology

### 1) Data
- daily adjusted-close data for 24 US large-cap equities
- daily implied-volatility proxy (VIX)
- explicit train / validation / test workflow

### 2) State variable
The notebook builds a past-looking state variable from implied volatility using:
- log transform,
- smoothing,
- EWMA standardisation.

### 3) Baselines
- **Global baseline:** covariance estimated on the full training set with shrinkage and low-rank PCA
- **Piecewise PCA:** low / mid / high stress buckets using **HMM-informed thresholds**

### 4) Main challenger
A **2-state Gaussian HMM** is fit on the training state series.  
The lagged filtered stress probability is then used to blend calm and stress covariance forecasts:

\[
\Sigma_{t|t-1} = (1-p_{t-1})\Sigma_{\text{calm}} + p_{t-1}\Sigma_{\text{stress}}
\]

### 5) Evaluation
- **Primary:** mean OOS Gaussian NLL
- **Secondary:** PCA reconstruction \(R^2\)
- **Uncertainty:** moving-block bootstrap on daily test observations
- **Extension:** rolling / walk-forward recalibration

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
└── equity-risk-modelling-regime-aware-pca.ipynb
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open:

```text
equity-risk-modelling-regime-aware-pca.ipynb
```

## Reproducibility

### Option A — quick start
Run the notebook with `yfinance` enabled.

### Option B — fixed data snapshot
Replace the data-loading block with CSV inputs for:
- the asset price panel,
- the IV proxy series.

The repository does **not** include raw market data.

## Limitations

- The project is **not** a production library; it is a research notebook.
- Market data are pulled externally unless the loader is replaced with local CSV files.
- The strongest evidence in the current V2 run is **static OOS + block-bootstrap**, not live rolling dominance.
- The effective sample starts in **2012-05-21**, not the nominal request date of 2010-01-01.
- Exact numeric results can vary with data-download timing and reruns.

## Roadmap

Natural next extensions:
- stronger rolling model selection,
- broader or historically reconstituted universes,
- additional conditional covariance benchmarks,
- cross-market validation (UK / EU / JP),
- separation of notebook logic into reusable Python modules.
