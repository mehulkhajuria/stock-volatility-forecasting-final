# Data

This project uses daily OHLCV data downloaded from Yahoo Finance through
`yfinance`.

The raw financial data are public data and are not original to this project.
The original contribution is the reproducible pipeline that constructs a
leakage-safe volatility forecasting dataset, engineers lagged and technical
features, evaluates multiple model families, and generates figures and summary
results.

Raw downloads are cached in `data/raw/` as CSV files. Processed modeling data
are written to `data/processed/`.
