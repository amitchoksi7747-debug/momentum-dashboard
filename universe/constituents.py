from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

NIFTYINDICES_BASE = "https://niftyindices.com/IndexConstituent/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# index_key -> constituent CSV filename on niftyindices.com (verified reachable, no auth needed)
INDEX_CSV_MAP: Dict[str, str] = {
    "NIFTY_LARGEMIDCAP_250": "ind_niftylargemidcap250list.csv",
    "NIFTY_500": "ind_nifty500list.csv",
    "NIFTY_MIDSMALLCAP_400": "ind_niftymidsmallcap400list.csv",
    "NIFTY_TOTAL_MARKET": "ind_niftytotalmarket_list.csv",
}


def _download_csv(filename: str, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    url = NIFTYINDICES_BASE + filename
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            expected = {"Company Name", "Industry", "Symbol", "Series", "ISIN Code"}
            missing = expected - set(df.columns)
            if missing:
                raise ValueError(f"{filename}: missing expected columns {missing}")
            return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to download constituent list {filename}: {last_exc}")


def fetch_constituents(index_key: str, raw_cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Download current constituents for one of INDEX_CSV_MAP's index keys.

    Returns columns: Symbol (canonical NSE:X form), CompanyName, Industry, ISIN.
    Only EQ series rows are kept (drops any non-equity rows some lists include).
    """
    if index_key not in INDEX_CSV_MAP:
        raise ValueError(f"Unknown index_key {index_key!r}; expected one of {list(INDEX_CSV_MAP)}")

    df = _download_csv(INDEX_CSV_MAP[index_key])
    df = df[df["Series"].astype(str).str.strip().str.upper() == "EQ"].copy()
    df["Symbol"] = "NSE:" + df["Symbol"].astype(str).str.strip()
    out = df.rename(columns={"Company Name": "CompanyName", "ISIN Code": "ISIN"})[
        ["Symbol", "CompanyName", "Industry", "ISIN"]
    ].reset_index(drop=True)

    if raw_cache_dir:
        cache_dir = Path(raw_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_dir / f"{index_key}.csv", index=False)

    return out


def refresh_all_constituents(raw_cache_dir: str | Path | None = None) -> Dict[str, pd.DataFrame]:
    """Download all 4 index constituent lists. Raises if any single one fails —
    a partial universe would silently corrupt eligibility/ranking downstream."""
    return {key: fetch_constituents(key, raw_cache_dir=raw_cache_dir) for key in INDEX_CSV_MAP}


def build_universe_membership_table(constituents: Dict[str, pd.DataFrame], as_of: str) -> pd.DataFrame:
    """Flatten {index_key: df} into a long table: index_name, symbol, company_name, industry, isin, as_of.

    v1 is current-constituents-only (no point-in-time history) per the live-tracking use case —
    the `as_of` column simply records when this snapshot was taken.
    """
    frames = []
    for index_key, df in constituents.items():
        f = df.copy()
        f["IndexName"] = index_key
        f["AsOfDate"] = as_of
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["IndexName", "Symbol", "CompanyName", "Industry", "ISIN", "AsOfDate"])
    return pd.concat(frames, ignore_index=True)[
        ["IndexName", "Symbol", "CompanyName", "Industry", "ISIN", "AsOfDate"]
    ]
