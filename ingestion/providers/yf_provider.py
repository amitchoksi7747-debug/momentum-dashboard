from __future__ import annotations

import time
from typing import List

import pandas as pd
import yfinance as yf

from ingestion.prices import normalize_symbol
from ingestion.providers.base import PriceProvider

CHUNK_SIZE = 100  # yfinance batch download — large single calls degrade / partially fail


def _flatten_multi_ticker(df: pd.DataFrame, requested_yf_symbols: List[str]) -> pd.DataFrame:
    """yf.download(tickers=[...]) returns a (field, ticker) or (ticker, field) MultiIndex
    depending on version/group_by. Normalize to long format regardless."""
    if df.empty:
        return pd.DataFrame(columns=["YFSymbol", "Date", "Open", "High", "Low", "Close", "Volume"])

    frames = []
    if isinstance(df.columns, pd.MultiIndex):
        level0 = set(df.columns.get_level_values(0))
        # group_by='ticker' -> level0 is tickers; group_by='column' (default) -> level0 is fields
        if set(requested_yf_symbols) & level0:
            tickers = [t for t in requested_yf_symbols if t in level0]
            for t in tickers:
                sub = df[t].reset_index()
                sub["YFSymbol"] = t
                frames.append(sub)
        else:
            for t in requested_yf_symbols:
                cols = [(field, t) for field in ["Open", "High", "Low", "Close", "Volume"] if (field, t) in df.columns]
                if not cols:
                    continue
                sub = df[cols].copy()
                sub.columns = [c[0] for c in cols]
                sub = sub.reset_index()
                sub["YFSymbol"] = t
                frames.append(sub)
    else:
        # single ticker requested, columns are flat
        sub = df.reset_index()
        sub["YFSymbol"] = requested_yf_symbols[0]
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["YFSymbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"Date": "Date"})
    keep = [c for c in ["YFSymbol", "Date", "Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    return out[keep]


class YFinanceProvider(PriceProvider):
    name = "yfinance"

    def get_daily(self, symbols: List[str], start: str, end: str, retries: int = 2) -> pd.DataFrame:
        yf_map = {normalize_symbol(s): s for s in symbols}
        yf_symbols = list(yf_map.keys())

        all_frames = []
        for i in range(0, len(yf_symbols), CHUNK_SIZE):
            chunk = yf_symbols[i : i + CHUNK_SIZE]
            for attempt in range(retries):
                try:
                    raw = yf.download(
                        tickers=chunk,
                        start=start,
                        end=end,
                        auto_adjust=True,
                        progress=False,
                        threads=True,
                        group_by="ticker",
                    )
                    flat = _flatten_multi_ticker(raw, chunk)
                    all_frames.append(flat)
                    break
                except Exception:  # noqa: BLE001
                    if attempt < retries - 1:
                        time.sleep(2 * (attempt + 1))
                    else:
                        all_frames.append(pd.DataFrame(columns=["YFSymbol", "Date", "Open", "High", "Low", "Close", "Volume"]))

        if not all_frames:
            return pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "Source"])

        out = pd.concat(all_frames, ignore_index=True)
        if out.empty:
            return pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "Source"])

        out["Symbol"] = out["YFSymbol"].map(yf_map)
        out["Date"] = pd.to_datetime(out["Date"]).dt.date.astype(str)
        out["Source"] = self.name
        out = out.dropna(subset=["Symbol", "Close"])
        return out[["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "Source"]].reset_index(drop=True)
