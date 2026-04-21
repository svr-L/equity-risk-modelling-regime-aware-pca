# Equity Risk Modelling with Regime-Aware PCA

A research notebook on **out-of-sample covariance / density forecasting** for a panel of **24 US large-cap equities**.

## Project summary

This repository studies whether a **regime-aware covariance challenger** can improve on a static low-rank PCA risk model when market stress is proxied by a **past-looking VIX-derived state variable**.

The workflow compares:

- a **global / static PCA benchmark** with shrinkage covariance;
- an **HMM-informed piecewise PCA baseline** built from low / mid / high stress buckets;
- a **2-state Gaussian HMM calm/stress mixture** that blends covariance forecasts through time using filtered stress probabilities;
- an auxiliary **vol-mix / correlation-fixed** challenger.

The main objective is **risk modelling**, not return prediction. The key score is **out-of-sample Gaussian negative log-likelihood (NLL)**.

## Current notebook snapshot

The current executed notebook (`LMCS_Saverio_Lauriola_Equity_Risk_Modelling_GitReady.ipynb`) uses:

- **24 US large-cap equities**
- **VIX (`^VIX`)** as the regime proxy
- requested history from **2010 onward**
- an effective common-sample returns history that begins later because the analysis uses the **intersection of available histories across the whole panel**
- a train / test split ending training on **2021-12-31** and starting test on **2022-01-01**

### Static out-of-sample results

On the observed test set, the main static OOS ranking is:

- **HMM Mixture (calm/stress):** mean NLL ≈ **-175.978**
- **Vol-mix Corr-fixed:** mean NLL ≈ **-175.655**
- **Global (static):** mean NLL ≈ **-175.607**
- **Piecewise PCA (HMM-informed thresholds):** mean NLL ≈ **-175.041**

Relative to the global benchmark:

- **HMM Mixture − Global:** Δ mean NLL ≈ **-0.370**
- **Vol-mix − Global:** Δ mean NLL ≈ **-0.048**
- **Piecewise − Global:** Δ mean NLL ≈ **+0.566**

Secondary descriptive metric:

- **Global PCA reconstruction:** `R²_total,test ≈ 0.469`
- **Piecewise PCA reconstruction:** `R²_total,test ≈ 0.466`

### Bootstrap evidence

The notebook includes a moving-block bootstrap over the test period.

For the HMM-mixture challenger:

- **mean ΔNLL (mixture − global):** ≈ **-0.363**
- **block-bootstrap `P(win)` vs global:** **1.00**

For the vol-mix / corr-fixed challenger:

- **mean ΔNLL (vol-mix − global):** ≈ **-0.040**

For the piecewise PCA challenger:

- **mean ΔNLL (piecewise − global):** ≈ **+0.577**

### Rolling recalibration

The notebook also includes a walk-forward rolling section with periodic refits.

This should be read separately from the static OOS evidence:

- the **static** section asks whether a regime-aware structure contains useful information;
- the **rolling** section asks how much of that edge survives under repeated re-estimation.

In the current executed run, the rolling global benchmark remains slightly stronger than the rolling regime-aware challengers, so the strongest evidence in the repository is still the **static OOS + bootstrap** result.

## Why this project is interesting

This is not just a PCA notebook. It addresses a real modelling issue that appears often in applied quant work:

- static low-rank structure is interpretable but can miss time variation;
- hard regime splits are simple but can create unstable middle buckets;
- probabilistic state models offer a cleaner way to make covariance forecasts adapt over time.

That makes the project relevant for:

- **market risk quant** roles;
- **risk strats / quantitative risk research** roles;
- **quantitative research** profiles with a risk-modelling angle;
- systematic / portfolio-research roles that care about **time-varying covariance forecasts**.

## Methodology

### 1) Data

- Universe: 24 liquid US large-cap equities
- Regime proxy: CBOE VIX (`^VIX`)
- Returns: daily simple returns
- Split: explicit train / test windows

### 2) State variable

The notebook builds a strictly past-looking state variable from implied volatility using:

- a log transform;
- smoothing;
- EWMA standardisation.

### 3) Benchmarks and challengers

- **Global model:** shrinkage covariance + low-rank PCA on the full training sample
- **Piecewise PCA:** low / mid / high regimes defined by **HMM-informed** thresholds on the state variable
- **HMM-mixture:** 2-state Gaussian HMM fitted on the training state series; filtered stress probabilities are then used to mix calm and stress covariance forecasts through time
- **Vol-mix corr-fixed:** covariance scaling on volatility while keeping correlation fixed

### 4) Evaluation

- Primary: **mean OOS Gaussian NLL**
- Secondary: **PCA reconstruction `R²`**
- Robustness: **moving-block bootstrap**
- Deployment proxy: **rolling / walk-forward recalibration**

## Key interpretation notes

A few modelling choices are worth reading carefully:

- The notebook deliberately separates **validation-based selection** from **ex-post OOS ranking**.  
  The configuration flagged as selected on validation is the live candidate chosen inside TRAIN/validation; the best realized static OOS model is reported separately for transparency only.
- The piecewise baseline is **not** based on hard-coded quantiles.  
  Its thresholds are selected from HMM information to produce a stronger comparator.
- The repository is best read as a **research prototype**, not as a production library or as a final empirical paper.

## Repository contents

```text
.
├── README.md
├── equity-risk-modelling-regime-aware-pca.ipynb
├── requirements.txt            
└── LICENSE                     
```

## How to run

### Option A — quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open:

```text
equity-risk-modelling-regime-aware-pca.ipynb
```

### Option B — fixed local data

Replace the download block with local CSV inputs for:

- the equity price panel;
- the VIX series.

The repository does **not** include raw market data.

## Limitations

- The current universe is a **research panel**, not a historically reconstituted index.
- The strongest result is **static OOS**; the rolling section is more mixed.
- The benchmark set is meaningful for a research notebook, but can be broadened further for a more paper-like comparison.
- Exact numbers can vary slightly with data-download timing and reruns.

## Suggested next steps

- strengthen rolling / nested validation and model-family selection;
- test alternative universes and broader cross-market panels;
- compare against additional conditional covariance benchmarks;
- separate reusable functions into modules if the repository grows.
