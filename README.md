# Regime-Aware ARIMA-GARCH for Indian Equities

Walk-forward comparison of ARMA-GARCH-family models, HMM regime layers and VaR backtests
on the Nifty 50 and five NSE stocks (Jan 2010 - Oct 2026, Yahoo Finance, frozen as tag `data-v1`).

## Setup
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt

## Run
python run_all.py --jobs 4          # all stages; resume with --from <stage>
python -m pytest -q tests           # unit tests and leakage test

## Layout
data/raw (frozen)  ->  data/processed/final (4,112 common returns)  ->  src/*.py  ->  results/ (CSVs, figures)

## Design rules fixed in advance
K = 2 regimes; primary spec GARCH-normal, secondary GJR-normal; regime model falls back to the pooled model
if a regime has < 250 in-sample days; HMM refit every 20 days using past data only; filtered probabilities only
(smoothed ones are used solely to measure look-ahead bias); Variant B (Nifty regimes) declared primary.

## Main findings (test period 3 Mar 2020 - 30 Oct 2026)
<fill in from results/ once final>

## Limitations
One test window dominated by 2020; squared-return proxy is noisy; six series; single unofficial data source.