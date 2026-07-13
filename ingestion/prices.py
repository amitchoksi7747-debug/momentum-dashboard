from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if symbol.startswith("NSE:"):
        return symbol.split(":", 1)[1] + ".NS"
    if symbol.startswith("BSE:"):
        return symbol.split(":", 1)[1] + ".BO"
    return symbol


def _bse_fallback(symbol: str) -> str | None:
    """Return BSE (.BO) equivalent for an NSE symbol, or None if already BSE."""
    if symbol.endswith(".NS"):
        return symbol[:-3] + ".BO"
    return None


def _download_with_fallback(yf_symbol: str, **kwargs) -> tuple[pd.DataFrame, str]:
    """Try yf_symbol first; if empty, attempt BSE fallback. Returns (df, symbol_used)."""
    hist = yf.download(yf_symbol, **kwargs)
    if not hist.empty:
        return hist, yf_symbol
    fallback = _bse_fallback(yf_symbol)
    if fallback:
        hist_bo = yf.download(fallback, **kwargs)
        if not hist_bo.empty:
            return hist_bo, fallback
    return hist, yf_symbol


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns (ticker as second level) to simple names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def fetch_latest_prices(symbols: Iterable[str]) -> pd.DataFrame:
    records = []
    end = datetime.utcnow().date() + timedelta(days=1)
    start = end - timedelta(days=10)

    for original_symbol in symbols:
        yf_symbol = normalize_symbol(original_symbol)
        try:
            hist, used_symbol = _download_with_fallback(
                yf_symbol, start=start, end=end, auto_adjust=False, progress=False, interval="1d"
            )
            if hist.empty:
                records.append({"Symbol": original_symbol, "YFSymbol": yf_symbol, "Price": None, "PrevClose": None, "AsOfDate": None, "PriceSource": "yfinance_empty"})
                continue
            hist = hist.reset_index()
            hist = _flatten_columns(hist)
            last = hist.iloc[-1]
            prev_close = hist.iloc[-2]["Close"] if len(hist) > 1 else last["Close"]
            source = "yfinance_bse_fallback" if used_symbol != yf_symbol else "yfinance"
            records.append({
                "Symbol": original_symbol,
                "YFSymbol": used_symbol,
                "Price": float(last["Close"]),
                "PrevClose": float(prev_close),
                "AsOfDate": pd.to_datetime(last["Date"]).date().isoformat(),
                "PriceSource": source,
            })
        except Exception as exc:
            records.append({"Symbol": original_symbol, "YFSymbol": yf_symbol, "Price": None, "PrevClose": None, "AsOfDate": None, "PriceSource": f"error:{exc}"})
    return pd.DataFrame(records)


def fetch_price_history(symbols: Iterable[str], period: str = "1y") -> pd.DataFrame:
    frames = []
    for original_symbol in symbols:
        yf_symbol = normalize_symbol(original_symbol)
        try:
            hist, used_symbol = _download_with_fallback(
                yf_symbol, period=period, auto_adjust=False, progress=False, interval="1d"
            )
            if hist.empty:
                continue
            hist = hist.reset_index()
            hist = _flatten_columns(hist)
            hist["Symbol"] = original_symbol
            hist["YFSymbol"] = used_symbol
            frames.append(hist)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Symbol", "YFSymbol"])
    out = pd.concat(frames, ignore_index=True)
    # Normalise column names: strip whitespace, no spaces
    out.columns = [str(c).strip().replace(" ", "") for c in out.columns]
    return out


def fetch_benchmark_history(benchmark_symbol: str, period: str = "1y") -> pd.DataFrame:
    hist = yf.download(benchmark_symbol, period=period, auto_adjust=False, progress=False, interval="1d")
    if hist.empty:
        return pd.DataFrame(columns=["Date", "Close", "Benchmark"])
    hist = hist.reset_index()
    hist = _flatten_columns(hist)
    hist["Benchmark"] = benchmark_symbol
    return hist
