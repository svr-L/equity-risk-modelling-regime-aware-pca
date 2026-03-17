# US Tech Risk Modelling with Regime-Aware PCA / HMM Mixtures

A research notebook on **out-of-sample covariance / risk forecasting** for a basket of large-cap US technology equities.

## Project summary

This repository studies whether a **regime-aware challenger** can improve on a static low-rank covariance model when regimes are inferred from an implied-volatility proxy.

The workflow compares:

- a **global / static covariance model**
- a **piecewise percentile baseline** built on an IV-derived state variable
- a **2-state Gaussian HMM challenger** that produces filtered calm/stress probabilities and mixes covariance forecasts through time
- an auxiliary **vol-mix / correlation-fixed** challenger for robustness comparison

The main goal is **risk modelling**, not return prediction: the key metric is **out-of-sample Gaussian negative log-likelihood (NLL)**.

## Why this project is interesting

This is not just a “PCA notebook”.

It tackles a real modelling issue that appears often in applied quant work:
- static low-rank structure is interpretable but can miss time variation,
- hard regime splits are simple but can create unstable or weakly identified middle regimes,
- probabilistic state models offer a cleaner way to make covariance forecasts adapt over time.

That makes the project relevant for:
- market risk quant roles,
- quantitative research roles with a risk-modelling angle,
- systematic / portfolio-research profiles that value regime-aware risk forecasts.

## Current notebook findings

The notebook contains **two different layers of evidence**, and they should be read separately.

### 1) Single-run point estimates on the observed test set
Based on the executed outputs embedded in the notebook:

- **Global PCA reconstruction**: $R^2_{total,test} pprox 0.662$
- **Piecewise PCA reconstruction**: $R^2_{total,test} pprox 0.645$
- **Global mean OOS Gaussian NLL**: about **-68.547**
- **Piecewise percentile mean OOS Gaussian NLL**: about **-67.980**
- **Static HMM-mixture mean OOS Gaussian NLL**: about **-68.459**

Interpretation of the point estimates:
- the **piecewise percentile split underperforms**,
- the **static HMM challenger is conceptually stronger**, but in the embedded single-run snapshot it does **not** beat the global baseline on mean test NLL.

### 2) Block-bootstrap evidence on performance robustness
Section 9 of the notebook reports dependence-preserving **block-bootstrap distributions** for performance deltas relative to the global baseline.

For the HMM-mixture challenger:
- **mean $\Delta$NLL (mixture minus global)**: about **-0.209**
- **$P(	ext{HMM-mixture beats global})$**: about **0.943**

For the vol-mix / corr-fixed challenger:
- **mean $\Delta$NLL (vol-mix minus global)**: about **-0.061**
- **$P(	ext{vol-mix beats global})$**: about **0.603**

For the piecewise PCA challenger:
- **mean $\Delta$NLL (piecewise minus global)**: about **+0.551**
- **$P(	ext{piecewise beats global})$**: about **0.010**

Interpretation of the bootstrap evidence:
- **piecewise PCA is clearly the weakest challenger and is effectively rejected**,
- **the HMM mixture is the strongest challenger in the current notebook**,
- the fairest wording is that the HMM shows **bootstrap-based OOS gains** rather than a uniformly dominant single-run test-set gain.

That is still strong and fully publishable on GitHub.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   └── README.md
└── us_tech_risk_modelling_regime_aware_pca_hmm.ipynb
```

## Methodology

### 1) Data
- Universe: large-cap US tech equities
- Regime proxy: Nasdaq volatility proxy (`^VXN` in the current notebook)
- Sample split: explicit train / test windows

### 2) State variable
The notebook builds a past-looking state variable from implied volatility:
- log transform,
- smoothing,
- EWMA standardisation.

### 3) Baselines
- **Global model:** covariance estimated on the training set with shrinkage
- **Piecewise PCA:** low / mid / high regimes defined by percentile thresholds of the state variable

### 4) Challenger
A **2-state Gaussian HMM** is fit on the training state series.  
The filtered stress probability is then used in a no-look-ahead way:

$$
\Sigma_{t|t-1} = (1-p_{t-1})\Sigma_{calm} + p_{t-1}\Sigma_{stress}
$$

### 5) Evaluation
- Primary: **mean OOS Gaussian NLL**
- Secondary: **PCA reconstruction $R^2$**
- Uncertainty: **block bootstrap** over daily test observations
- Extension scaffold: **rolling / walk-forward recalibration**

## Reproducibility

### Option A — quick start
Run the notebook with `yfinance` enabled.

### Option B — fixed data snapshot
Replace the data-loading block with CSV inputs for:
- the asset price panel,
- the IV proxy series.

The repository does **not** include raw market data. The `data/README.md` file documents the expected file format.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scriptsctivate
pip install -r requirements.txt
jupyter notebook
```

Then open:

```text
us_tech_risk_modelling_regime_aware_pca_hmm.ipynb
```

## Suggested positioning for GitHub / CV

A fair public positioning is:

> Regime-aware covariance forecasting for US tech equities using low-rank PCA baselines, IV-driven state conditioning, and a Gaussian HMM calm/stress mixture challenger evaluated out of sample with Gaussian NLL and block-bootstrap diagnostics.

A fair CV-style wording is:

> Implemented a low-rank PCA risk model for US tech equities with shrinkage covariance and IV-driven state conditioning ($R^2_{total,test}pprox0.66$). Benchmarked piecewise PCA against a two-state Gaussian HMM mixture challenger; the HMM delivered **block-bootstrap OOS gains** versus the global baseline ($\Delta$mean NLL $pprox -0.209$, $P(	ext{win})pprox0.943$), while piecewise PCA underperformed clearly.

## Limitations

- The project is **not** a production library; it is a research notebook.
- Market data are pulled externally unless the loader is replaced with local CSV files.
- The strongest current HMM evidence is **bootstrap-based**, while the full rolling / walk-forward benchmark remains an explicit next step.
- Exact numeric results can vary with data-download timing and reruns.

## Roadmap

Natural next extensions:
- full walk-forward reporting with periodic HMM refits,
- broader equity universes,
- cross-market validation (UK / EU / JP),
- more formal validation-driven hyper-parameter selection,
- separation of notebook logic into reusable Python modules.

## Publication verdict

**Yes — publishable now**, provided it is presented as a **research notebook / modelling study** rather than a production library or a finalized paper.
