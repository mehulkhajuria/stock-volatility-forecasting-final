from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import TARGET_WINDOW


def naive_persistence(test: pd.DataFrame) -> tuple[pd.Series, float, str]:
    start = time.perf_counter()
    pred = test[f"realized_vol_{TARGET_WINDOW}"].copy()
    return pred.rename("Naive Persistence"), time.perf_counter() - start, "Uses current 5-day realized volatility."


def rolling_mean(test: pd.DataFrame, window: int = 20) -> tuple[pd.Series, float, str]:
    start = time.perf_counter()
    cols = [f"vol_lag_{i}" for i in range(1, window + 1)]
    pred = test[cols].mean(axis=1)
    return pred.rename("Rolling Mean"), time.perf_counter() - start, f"Mean of previous {window} volatility lags."


def garch_forecast(train_validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, float, str]:
    start = time.perf_counter()
    try:
        from arch import arch_model

        returns_pct = train_validation["log_return"].dropna() * 100
        model = arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        fit = model.fit(disp="off", show_warning=False)
        forecasts = fit.forecast(horizon=len(test), reindex=False)
        variance = forecasts.variance.iloc[-1].to_numpy()
        pred = pd.Series(np.sqrt(variance) / 100 * np.sqrt(252), index=test.index, name="GARCH(1,1)")
        note = "arch GARCH(1,1), fit once on train+validation returns."
    except Exception as exc:
        pred = test[f"realized_vol_{TARGET_WINDOW}"].copy()
        pred.name = "GARCH(1,1)"
        note = f"GARCH unavailable; persistence fallback used. Error: {exc}"
    return pred, time.perf_counter() - start, note
