"""Data download utilities for USA private markets model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import yfinance as yf
from fredapi import Fred

from .config import get_all_tickers, get_fred_series_ids


def download_yahoo_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(ticker, axis=1, level=1)
        if data.empty:
            print(f"[warn] No data for {ticker}")
            continue
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        frame = data[[col]].rename(columns={col: "adj_close"})
        frame["ticker"] = ticker
        frames.append(frame.reset_index().rename(columns={"Date": "date"}))
    if not frames:
        raise RuntimeError("No price data downloaded.")
    return pd.concat(frames, ignore_index=True)


def _get_fred_client() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY environment variable is not set.")
    return Fred(api_key=api_key)


def download_fred_series(
    series_ids: Dict[str, Iterable[str]],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    fred = _get_fred_client()
    all_ids = [sid for group in series_ids.values() for sid in group]
    frames = []
    for sid in all_ids:
        data = fred.get_series(sid, observation_start=start, observation_end=end)
        if data is None:
            print(f"[warn] No FRED data for {sid}")
            continue
        series = pd.Series(data, name=sid)
        series.index = pd.to_datetime(series.index)
        frames.append(series)
    if not frames:
        raise RuntimeError("No FRED series downloaded.")
    return pd.concat(frames, axis=1).sort_index()


def main(
    data_dir: str = "data",
    price_start: str = "2005-01-01",
    fred_start: str = "2000-01-01",
) -> None:
    base = Path(data_dir)
    price_dir = base / "raw" / "prices"
    fred_dir = base / "raw" / "fred"
    price_dir.mkdir(parents=True, exist_ok=True)
    fred_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Yahoo Finance prices...")
    prices = download_yahoo_prices(get_all_tickers(), price_start)
    prices.to_csv(price_dir / "prices.csv", index=False)

    print("Downloading FRED series...")
    fred_df = download_fred_series(get_fred_series_ids(), fred_start)
    fred_df.to_csv(fred_dir / "fred_series.csv", index=True)
    print("Data download complete.")


if __name__ == "__main__":
    main()

