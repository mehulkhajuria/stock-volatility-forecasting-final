# Video Presentation Script

Target length: 12-13 minutes, leaving a buffer under the 15-minute limit.

## 0:00-0:45 Introduction and Research Question

On screen: title slide or first page of `paper/final_project.pdf`.

Say: My project is titled "Predicting Short-Term Stock Market Volatility Using Time Series and Machine Learning Models." I am forecasting future 5-day realized volatility for SPY using public Yahoo Finance data and a reproducible Python pipeline. The research question is whether historical price, volume, technical, and market-context features can forecast short-term volatility without leaking future information.

Transition: Move from the paper title to the README.

## 0:45-1:45 Motivation

On screen: README abstract and motivation.

Say: Volatility matters for risk management, options pricing, and position sizing. Predicting price direction is extremely hard, but volatility has persistence and clustering, so it is a good target for a data science comparison. The project is not financial advice; it is a controlled model-comparison exercise.

## 1:45-2:45 Dataset and Preprocessing

On screen: `data/README.md`, `results/data_inventory.csv`, or README dataset table.

Say: I used daily OHLCV data from Yahoo Finance through yfinance. The target symbol is SPY, with AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, and META used as context. The run covers 2014-01-01 through 2026-04-24. The split is chronological: train through 2021, validation in 2022, and final test from 2023 onward.

## 2:45-4:00 Feature Engineering and Target Construction

On screen: `src/features.py`.

Say: The target is future 5-day realized volatility. At date t, the target uses returns from t+1 through t+5, while features use only data through t. I engineered log returns, rolling realized volatility over 5, 10, and 20 days, 30 days of return and volatility lags, volume ratios, RSI, Bollinger features, MACD, and cross-symbol volatility context.

Transition: Point to the comment in `features.py` documenting the leakage rule.

## 4:00-5:30 Models Compared

On screen: README model table or `src/train_baselines.py`, `src/train_ml.py`, `src/train_lstm.py`.

Say: I compared naive persistence, a rolling mean baseline, GARCH(1,1), Random Forest, XGBoost gradient boosting, and an MLP neural fallback. The code attempts TensorFlow and PyTorch LSTM first, but this Python 3.13 environment did not have those packages available, so the reported neural result is a deterministic MLP fallback. I document that limitation instead of claiming a fake LSTM result.

## 5:30-7:30 Validation Protocol and Leakage Prevention

On screen: `src/run_all.py` and `src/features.py`.

Say: I never randomly shuffle the financial time series. The final test period is held out by date. Models train on train plus validation before final test evaluation. The target is shifted forward, and lagged features are shifted backward, so a row never sees the future returns used in its target.

## 7:30-9:30 Results and Metrics

On screen: `results/metrics.csv` and README results table.

Say: Gradient Boosting had the lowest RMSE at 0.0839. Random Forest was second at 0.0875. The rolling mean baseline was close at 0.0899 RMSE and had the best MAE and best defined directional accuracy. Naive persistence predicts no change by construction, so its directional score is NA.

## 9:30-11:00 Figures and Interpretation

On screen: open `figures/figure_contact_sheet.png`, then individual figures if time allows.

Say: The volatility time series shows regime shifts and spikes. The correlation heatmap shows that volatility across large tech stocks and SPY is strongly related. The predicted-versus-actual plot shows that the best model follows broad regimes but smooths sudden spikes. Feature importance shows current and lagged volatility dominate, which is financially sensible.

## 11:00-12:00 Limitations

On screen: paper limitations section.

Say: The biggest limitations are daily data only, no option-implied volatility, no news or macro events, a one-shot GARCH fit, and no true LSTM result in this environment. The Diebold-Mariano tests are approximate. The work is a reproducible academic project, not a trading system.

## 12:00-13:00 GitHub, Reproducibility, and Conclusion

On screen: GitHub repo or local README.

Say: The project can be reproduced with one command: `python -m src.run_all`. The repo includes source code, generated metrics, figures, the final paper, and this video script. The main conclusion is that leakage-safe validation and strong baselines are essential; XGBoost performed best by RMSE, but the rolling baseline was very competitive.

## 13:00-15:00 Buffer

Use this only for demo delays, opening files, or restating the conclusion.

## Shorter Backup Version

If running long, skip detailed code walkthroughs and show only:

1. README dataset/model/results tables.
2. `src/features.py` leakage comment.
3. `results/metrics.csv`.
4. `figures/figure_contact_sheet.png`.
5. Final paper conclusion.

## Recording Checklist

The course wording says Zoom, so Zoom is safest unless the instructor allows another recorder. If using Windows Game Bar, verify that the course permits it.

- Open the public GitHub repo or local README.
- Open `paper/final_project.pdf`.
- Open the `results/` folder.
- Open the `figures/` folder or `figure_contact_sheet.png`.
- Run or show `python -m src.run_all`.
- Keep the recording under 15 minutes.
- Make sure the final video link is public or viewable by the instructor.
