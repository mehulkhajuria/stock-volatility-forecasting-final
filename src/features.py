from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    MAX_LAG,
    PROCESSED_DIR,
    ROLLING_WINDOWS,
    TARGET_WINDOW,
    VOLUME_WINDOWS,
    ensure_directories,
)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _add_symbol_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    price_col = "Adj Close" if "Adj Close" in out.columns else "Close"
    out["adj_close"] = out[price_col].astype(float)
    out["log_return"] = np.log(out["adj_close"] / out["adj_close"].shift(1))

    for window in ROLLING_WINDOWS:
        out[f"realized_vol_{window}"] = out["log_return"].rolling(window).std() * np.sqrt(252)

    # Leakage rule: the target at date t is realized volatility over returns
    # t+1 through t+TARGET_WINDOW. The feature row can only see data through t.
    out["target_future_vol"] = out[f"realized_vol_{TARGET_WINDOW}"].shift(-TARGET_WINDOW)

    for lag in range(1, MAX_LAG + 1):
        out[f"return_lag_{lag}"] = out["log_return"].shift(lag)
        out[f"vol_lag_{lag}"] = out[f"realized_vol_{TARGET_WINDOW}"].shift(lag)

    for window in VOLUME_WINDOWS:
        out[f"volume_ma_{window}"] = out["Volume"].rolling(window).mean()
        out[f"volume_ratio_{window}"] = out["Volume"] / out[f"volume_ma_{window}"]

    out["rsi_14"] = _rsi(out["adj_close"])
    ma20 = out["adj_close"].rolling(20).mean()
    sd20 = out["adj_close"].rolling(20).std()
    out["bb_width"] = (4 * sd20) / ma20
    out["bb_percent_b"] = (out["adj_close"] - (ma20 - 2 * sd20)) / (4 * sd20)

    ema12 = out["adj_close"].ewm(span=12, adjust=False).mean()
    ema26 = out["adj_close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["symbol"] = symbol
    return out


def build_feature_frames(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {symbol: _add_symbol_features(df, symbol) for symbol, df in raw_frames.items()}


def build_spy_model_frame(feature_frames: dict[str, pd.DataFrame], target_symbol: str = "SPY") -> pd.DataFrame:
    ensure_directories()
    spy = feature_frames[target_symbol].copy()
    context_cols = []
    for symbol, df in feature_frames.items():
        if symbol == target_symbol:
            continue
        col = f"{symbol}_vol5"
        spy[col] = df[f"realized_vol_{TARGET_WINDOW}"].reindex(spy.index)
        context_cols.append(col)

    base_cols = [
        "adj_close",
        "log_return",
        "realized_vol_5",
        "realized_vol_10",
        "realized_vol_20",
        "target_future_vol",
        "volume_ma_5",
        "volume_ratio_5",
        "volume_ma_20",
        "volume_ratio_20",
        "rsi_14",
        "bb_width",
        "bb_percent_b",
        "macd",
        "macd_signal",
        "macd_hist",
    ]
    lag_cols = [f"return_lag_{i}" for i in range(1, MAX_LAG + 1)] + [
        f"vol_lag_{i}" for i in range(1, MAX_LAG + 1)
    ]
    keep_cols = base_cols + lag_cols + context_cols
    model_df = spy[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model_df.index.name = "Date"
    model_df.to_csv(PROCESSED_DIR / "spy_features.csv")
    return model_df


def split_by_date(df: pd.DataFrame, train_end: str, validation_end: str):
    train = df.loc[:train_end].copy()
    validation = df.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1) : validation_end].copy()
    test = df.loc[pd.Timestamp(validation_end) + pd.Timedelta(days=1) :].copy()
    return train, validation, test
