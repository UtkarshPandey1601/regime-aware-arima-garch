# Regime-Aware ARIMA-GARCH for Stock Return, Volatility and Risk Forecasting

**An Empirical Study on Indian Equities**

B.Tech Major Project | Group A12 | 2026–27  
Department of Computer Science and Engineering, Delhi Technological University

## Overview

This project investigates whether incorporating volatility regimes into ARIMA-GARCH models improves stock market volatility forecasting and Value-at-Risk (VaR) estimation.

The framework combines ARMA mean modelling, GARCH-family volatility models and Hidden Markov Models (HMM) to study six Indian equity market series.

## Research Novelty & Contributions

The primary contribution is an empirical investigation of **regime-aware forecasting under a leakage-free evaluation framework**.

- **Look-ahead bias analysis:** Quantifies the apparent forecasting improvement obtained from smoothed HMM probabilities compared with real-time filtered probabilities.
- **Regime-aware forecasting:** Evaluates hard and soft regime-specific forecasts against pooled GARCH models.
- **Comprehensive model comparison:** Tests 12 GARCH-family specifications combining four volatility models and three innovation distributions.
- **Statistical validation:** Uses walk-forward evaluation, Diebold-Mariano tests, Model Confidence Set analysis and VaR backtesting.

Rather than assuming regime information always improves predictions, the project evaluates its actual out-of-sample contribution.

## Dataset

Daily adjusted closing prices obtained from Yahoo Finance:

| Series | Sector |
|---|---|
| Nifty 50 | Market Index |
| ICICI Bank | Banking |
| TCS | IT |
| Reliance Industries | Energy |
| Maruti Suzuki | Automobile |
| Hindustan Unilever | FMCG |

**Period:** January 2010 – October 2026  
**Final aligned sample:** 4,112 observations per series.

## Methodology

1. Data preprocessing and statistical diagnostics.
2. ARMA mean-model selection using BIC.
3. Evaluation of GARCH, EGARCH, GJR-GARCH and APARCH under Normal, Student-t and Skewed-t distributions.
4. HMM-based volatility regime identification.
5. Filtered-probability-based regime-aware walk-forward forecasting.
6. Volatility evaluation using QLIKE, MAE, RMSE and statistical tests.
7. VaR estimation and backtesting at 1% and 5% levels.

## Key Findings

- GARCH(1,1) improves QLIKE over ARMA-only forecasts across all six series.
- Asymmetric GARCH specifications do not demonstrate consistent statistically significant forecasting gains.
- Heavy-tailed distributions improve aspects of VaR estimation.
- Filtered HMM regime information does not establish a reliable overall forecasting improvement.
- Smoothed HMM probabilities produce artificially better results, highlighting the importance of preventing look-ahead bias.

## Repository Structure

```text
├── data/          # Raw and processed datasets
├── src/           # Modelling and evaluation scripts
├── tests/         # Unit tests
├── results/       # Forecasts, tables and figures
├── config.yaml
├── requirements.txt
├── run_all.py
└── README.md
```

## Installation & Execution

```bash
git clone https://github.com/UtkarshPandey1601/regime-aware-arima-garch.git
cd regime-aware-arima-garch

python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.lock.txt
```

Run the pipeline:

```bash
python run_all.py --jobs 4
```

Run tests:

```bash
python -m pytest -q tests
```

*Note: A clean-clone, end-to-end execution is still pending verification.*

## Tech Stack

Python · NumPy · Pandas · SciPy · statsmodels · arch · hmmlearn · Matplotlib · pytest

---

**Repository:** [GitHub – regime-aware-arima-garch](https://github.com/UtkarshPandey1601/regime-aware-arima-garch)
