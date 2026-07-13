from __future__ import annotations

import io
import zipfile
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

# NSE's archive server (nsearchives.nseindia.com) returns 403/404 to a cold request; it needs a
# session cookie obtained by first hitting the main site. Verified working 2026-07-07.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
WARMUP_URL = "https://www.nseindia.com/all-reports"
BHAVCOPY_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date_str}_F_0000.csv.zip"

# Circuit-day proxy: NSE's UDiFF bhavcopy does not carry the price-band limit itself,
# so we approximate a circuit-lock day as high == low with a materially large move —
# a stock that's genuinely locked at a band trades zero range for the day.
CIRCUIT_PROXY_MOVE_THRESHOLD = 0.049


def _get_session() -> requests.Session:
    s = requests.Session()
    s.get(WARMUP_URL, headers=HEADERS, timeout=15)
    return s


def fetch_bhavcopy(trade_date: date, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Download and parse one day's NSE UDiFF bhavcopy. Returns columns:
    Symbol, Date, Open, High, Low, Close, PrevClose, Volume (equities / 'EQ' series only).

    Raises on network failure or if the date has no bhavcopy (holiday/weekend) — caller should
    treat that as 'no trading day', not a hard error.
    """
    sess = session or _get_session()
    date_str = trade_date.strftime("%Y%m%d")
    url = BHAVCOPY_URL_TEMPLATE.format(date_str=date_str)
    resp = sess.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        return pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close", "PrevClose", "Volume"])
    resp.raise_for_status()

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    inner_name = zf.namelist()[0]
    with zf.open(inner_name) as f:
        raw = pd.read_csv(f)

    eq = raw[raw["SctySrs"].astype(str).str.strip().str.upper() == "EQ"].copy()
    eq["Symbol"] = "NSE:" + eq["TckrSymb"].astype(str).str.strip()
    eq["Date"] = trade_date.isoformat()
    out = eq.rename(
        columns={
            "OpnPric": "Open",
            "HghPric": "High",
            "LwPric": "Low",
            "ClsPric": "Close",
            "PrvsClsgPric": "PrevClose",
            "TtlTradgVol": "Volume",
        }
    )[["Symbol", "Date", "Open", "High", "Low", "Close", "PrevClose", "Volume"]]
    return out.reset_index(drop=True)


def flag_circuit_days(bhavcopy_df: pd.DataFrame) -> pd.DataFrame:
    """Given a bhavcopy frame (single or multi-day), return Symbol, Date, IsCircuit (0/1)
    using the high==low + large-move proxy."""
    if bhavcopy_df.empty:
        return pd.DataFrame(columns=["Symbol", "Date", "IsCircuit"])
    df = bhavcopy_df.copy()
    move = (df["Close"] - df["PrevClose"]).abs() / df["PrevClose"].replace(0, pd.NA)
    locked_range = (df["High"] == df["Low"])
    df["IsCircuit"] = (locked_range & (move >= CIRCUIT_PROXY_MOVE_THRESHOLD)).astype(int)
    return df[["Symbol", "Date", "IsCircuit"]]
