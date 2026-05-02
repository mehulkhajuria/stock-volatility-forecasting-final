from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import RAW_DIR, START_DATE, SYMBOLS, ensure_directories


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    return df


def download_symbol(symbol: str, start: str = START_DATE, end: str | None = None) -> pd.DataFrame:
    """Download one symbol and cache it as a CSV file."""
    ensure_directories()
    cache_path = RAW_DIR / f"{symbol}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        if len(df) > 100:
            return df

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = _flatten_columns(df)
            if df.empty:
                raise RuntimeError(f"Yahoo Finance returned no rows for {symbol}.")
            df.index.name = "Date"
            df.to_csv(cache_path)
            return df
        except Exception as exc:  # pragma: no cover - depends on network
            last_error = exc
            time.sleep(2 + attempt)
    raise RuntimeError(f"Failed to download {symbol} after 3 attempts: {last_error}")


def download_all(symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols or SYMBOLS:
        frames[symbol] = download_symbol(symbol)
    return frames


def data_inventory(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        rows.append(
            {
                "symbol": csv_path.stem,
                "rows": len(df),
                "start": df["Date"].min().date().isoformat(),
                "end": df["Date"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)
