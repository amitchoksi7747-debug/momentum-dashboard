"""Fetch mutual-fund NAV history from api.mfapi.in (free, keyless).

Endpoint: https://api.mfapi.in/mf/{scheme_code} → {meta:{...}, data:[{date:"DD-MM-YYYY", nav:"12.34"}, ...]}
Data is newest-first. This module normalizes it to a tidy, oldest-first DataFrame.
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://api.mfapi.in/mf/{code}"
HEADERS = {"User-Agent": "vb-momentum-dashboard/1.0"}


def fetch_scheme_navs(scheme_code: int, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """Return NAV history for one scheme: columns Date (datetime), NAV (float), oldest first.
    Also returns the scheme's official name via the frame's .attrs['scheme_name']."""
    url = BASE_URL.format(code=scheme_code)
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("data", [])
            if not rows:
                return pd.DataFrame(columns=["Date", "NAV"])
            df = pd.DataFrame(rows)
            df["Date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
            df["NAV"] = pd.to_numeric(df["nav"], errors="coerce")
            df = df.dropna(subset=["Date", "NAV"]).sort_values("Date").reset_index(drop=True)
            df = df[["Date", "NAV"]]
            df.attrs["scheme_name"] = payload.get("meta", {}).get("scheme_name", str(scheme_code))
            return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to fetch NAVs for scheme {scheme_code}: {last_exc}")


def fetch_many(fund_specs: Iterable[dict]) -> pd.DataFrame:
    """fund_specs: iterable of {scheme_code, short_name, category}. Returns a long DataFrame:
    SchemeCode, ShortName, Category, OfficialName, Date, NAV — for every fund that returns data."""
    frames = []
    for spec in fund_specs:
        code = spec["scheme_code"]
        try:
            navs = fetch_scheme_navs(code)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: scheme {code} ({spec.get('short_name')}) failed: {exc}")
            continue
        if navs.empty:
            print(f"  WARNING: scheme {code} ({spec.get('short_name')}) returned no NAVs")
            continue
        navs = navs.copy()
        navs["SchemeCode"] = code
        navs["ShortName"] = spec["short_name"]
        navs["Category"] = spec.get("category", "")
        navs["OfficialName"] = navs.attrs.get("scheme_name", "")
        frames.append(navs)
        time.sleep(0.3)  # be polite to the free API
    if not frames:
        return pd.DataFrame(columns=["SchemeCode", "ShortName", "Category", "OfficialName", "Date", "NAV"])
    return pd.concat(frames, ignore_index=True)[
        ["SchemeCode", "ShortName", "Category", "OfficialName", "Date", "NAV"]
    ]
