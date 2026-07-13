from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS symbols (
        Symbol TEXT PRIMARY KEY,
        CompanyName TEXT,
        ISIN TEXT,
        FirstSeen TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prices_eod (
        Symbol TEXT NOT NULL,
        Date TEXT NOT NULL,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Volume REAL,
        Source TEXT,
        Quarantined INTEGER DEFAULT 0,
        PRIMARY KEY (Symbol, Date)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_prices_eod_date ON prices_eod (Date)",
    """
    CREATE TABLE IF NOT EXISTS universe_membership (
        IndexName TEXT NOT NULL,
        Symbol TEXT NOT NULL,
        CompanyName TEXT,
        Industry TEXT,
        ISIN TEXT,
        AsOfDate TEXT,
        PRIMARY KEY (IndexName, Symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mcap_class (
        Symbol TEXT NOT NULL,
        Category TEXT NOT NULL,
        MarketCap_Cr REAL,
        AsOfDate TEXT NOT NULL,
        PRIMARY KEY (Symbol, AsOfDate)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS circuit_days (
        Symbol TEXT NOT NULL,
        Date TEXT NOT NULL,
        IsCircuit INTEGER NOT NULL,
        PRIMARY KEY (Symbol, Date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS corporate_actions (
        Symbol TEXT NOT NULL,
        ExDate TEXT NOT NULL,
        ActionType TEXT,
        Factor REAL,
        Detail TEXT,
        PRIMARY KEY (Symbol, ExDate, ActionType)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_state (
        Strategy TEXT NOT NULL,
        Symbol TEXT NOT NULL,
        Units REAL NOT NULL,
        EntryDate TEXT,
        EntryRank REAL,
        PRIMARY KEY (Strategy, Symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_nav (
        Strategy TEXT NOT NULL,
        Date TEXT NOT NULL,
        NAV REAL NOT NULL,
        Cash REAL,
        NHoldings INTEGER,
        PRIMARY KEY (Strategy, Date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_trades (
        Strategy TEXT NOT NULL,
        Date TEXT NOT NULL,
        Symbol TEXT NOT NULL,
        Side TEXT NOT NULL,
        Units REAL,
        Price REAL,
        Reason TEXT,
        PRIMARY KEY (Strategy, Date, Symbol, Side)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_rebalances (
        Strategy TEXT NOT NULL,
        SignalDate TEXT NOT NULL,
        ExecDate TEXT,
        SheetJson TEXT,
        PRIMARY KEY (Strategy, SignalDate)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_meta (
        Strategy TEXT PRIMARY KEY,
        InceptionDate TEXT,
        NotionalCapital REAL,
        CurrentCash REAL DEFAULT 0
    )
    """,
]

# CREATE TABLE IF NOT EXISTS silently no-ops on a table that already exists with an older
# shape, so any column added after a table has shipped needs an explicit migration entry here
# too (learned the hard way: strategy_meta.CurrentCash and mcap_class.MarketCap_Cr were both
# added after their tables already existed in a live DB).
COLUMN_MIGRATIONS = [
    ("strategy_meta", "CurrentCash", "REAL DEFAULT 0"),
    ("mcap_class", "MarketCap_Cr", "REAL"),
]


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    for stmt in SCHEMA_STATEMENTS:
        conn.execute(stmt)
    for table, column, coltype in COLUMN_MIGRATIONS:
        existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing_cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    conn.commit()


def upsert_prices_eod(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert-or-replace rows keyed by (Symbol, Date). Returns row count written."""
    if df.empty:
        return 0
    cols = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "Source", "Quarantined"]
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    conn.executemany(
        """
        INSERT INTO prices_eod (Symbol, Date, Open, High, Low, Close, Volume, Source, Quarantined)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Symbol, Date) DO UPDATE SET
            Open=excluded.Open, High=excluded.High, Low=excluded.Low, Close=excluded.Close,
            Volume=excluded.Volume, Source=excluded.Source, Quarantined=excluded.Quarantined
        """,
        df.itertuples(index=False, name=None),
    )
    conn.commit()
    return len(df)


def replace_universe_membership(conn: sqlite3.Connection, index_name: str, df: pd.DataFrame) -> int:
    """Full refresh of one index's membership list (v1 = current constituents only)."""
    conn.execute("DELETE FROM universe_membership WHERE IndexName = ?", (index_name,))
    if not df.empty:
        conn.executemany(
            """
            INSERT INTO universe_membership (IndexName, Symbol, CompanyName, Industry, ISIN, AsOfDate)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            df.itertuples(index=False, name=None),
        )
    conn.commit()
    return len(df)


def load_prices_wide(conn: sqlite3.Connection, symbols: list[str] | None = None) -> pd.DataFrame:
    """Return a wide DataFrame: index=Date, columns=Symbol, values=Close. Excludes quarantined bars."""
    return load_field_wide(conn, "Close", symbols=symbols)


def load_field_wide(conn: sqlite3.Connection, field: str, symbols: list[str] | None = None) -> pd.DataFrame:
    """Same as load_prices_wide but for an arbitrary prices_eod column (e.g. 'Open')."""
    query = f"SELECT Date, Symbol, {field} FROM prices_eod WHERE Quarantined = 0"
    params: tuple = ()
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        query += f" AND Symbol IN ({placeholders})"
        params = tuple(symbols)
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    wide = df.pivot_table(index="Date", columns="Symbol", values=field, aggfunc="last")
    return wide.sort_index()


def load_circuit_flags_wide(conn: sqlite3.Connection, symbols: list[str] | None = None) -> pd.DataFrame:
    """Wide DataFrame: index=Date, columns=Symbol, values=IsCircuit (0/1). Missing cells (no
    bhavcopy row that day, e.g. before backfill started, or beyond it) are left NaN — callers
    should treat NaN as 0 (no known circuit) via .fillna(0), matching eligibility.py's convention."""
    query = "SELECT Date, Symbol, IsCircuit FROM circuit_days"
    params: tuple = ()
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        query += f" WHERE Symbol IN ({placeholders})"
        params = tuple(symbols)
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    wide = df.pivot_table(index="Date", columns="Symbol", values="IsCircuit", aggfunc="max")
    return wide.sort_index()
