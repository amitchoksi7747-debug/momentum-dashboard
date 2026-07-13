"""Live portfolio state machine — replaces backtesting entirely for this system's use case.

Two operations, both idempotent and SQLite-backed so a crashed/re-run pipeline resumes cleanly:

  1. build_rebalance_plan()  — monthly. Pure decision step: given today's ranks and current
     holdings, decide sells/holds/buys. Written to strategy_rebalances with ExecDate=NULL
     ("pending"). This is the artifact the user reviews/trades from.
  2. settle_pending_rebalance() — run once the next trading day's prices exist. Applies the
     plan at that day's Open (next_open execution, per config), updates strategy_state (units)
     and strategy_meta (cash), logs strategy_trades, and marks the plan settled (ExecDate set).

  mark_to_market() — daily. NAV = sum(units * close) + cash, upserted keyed by (Strategy, Date).

DEFAULT CHOSEN (flag for user confirmation, like other spec defaults in
docs/momentum_strategies_plan.md): every settled rebalance resets ALL final holdings
(survivors + new entries) to equal weight (1/top_n of total portfolio value at execution),
not just the newly-bought names. This is the standard convention for monthly-rebalanced
momentum indices (e.g. NSE's own Nifty200 Momentum 30) and avoids "equal weight at entry,
un-managed drift forever after" ambiguity. If the live system should instead only trade the
entries/exits and let survivor weights drift, change `_target_slot_value` accordingly.

A held stock that has fallen out of its index (universe list) but still passes rank/DEMA/
circuit checks is NOT force-exited (universe_exit_on_removal: false, confirmed). To evaluate
it fairly against the worst-rank-held threshold, ranking is computed over the union of
(current universe, currently-held symbols) — see build_rebalance_plan.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from strategies.eligibility import compute_eligibility
from strategies.ranking import compute_ranks


@dataclass
class RebalancePlan:
    strategy: str
    signal_date: str
    sells: list  # list of dicts: Symbol, Reason, UnitsBefore, EntryRank
    holds: list  # Symbol, AvgRank, EntryDate
    buys: list  # Symbol, AvgRank
    top_n: int
    worst_rank_held: int

    def to_json(self) -> str:
        return json.dumps(
            {
                "strategy": self.strategy,
                "signal_date": self.signal_date,
                "sells": self.sells,
                "holds": self.holds,
                "buys": self.buys,
                "top_n": self.top_n,
                "worst_rank_held": self.worst_rank_held,
            },
            default=str,
        )

    def to_sheet_df(self) -> pd.DataFrame:
        return rebalance_sheet_from_dict(
            {"sells": self.sells, "holds": self.holds, "buys": self.buys}
        )


def rebalance_sheet_from_dict(plan_dict: dict) -> pd.DataFrame:
    """Shared formatting for a rebalance sheet, used both by RebalancePlan.to_sheet_df() (fresh
    plan, in-memory) and by strategy_metrics.py (reading a persisted SheetJson back out)."""
    rows = []
    for s in plan_dict.get("sells", []):
        rows.append({"Action": "SELL", "Symbol": s["Symbol"], "AvgRank": None, "Reason": s["Reason"]})
    for h in plan_dict.get("holds", []):
        rows.append({"Action": "HOLD", "Symbol": h["Symbol"], "AvgRank": h.get("AvgRank"), "Reason": "still eligible, rank within hold band"})
    for b in plan_dict.get("buys", []):
        avg_rank = b.get("AvgRank")
        reason = f"rank {avg_rank:.2f} among best available" if avg_rank is not None else "among best available"
        rows.append({"Action": "BUY", "Symbol": b["Symbol"], "AvgRank": avg_rank, "Reason": reason})
    return pd.DataFrame(rows, columns=["Action", "Symbol", "AvgRank", "Reason"])


# ---------------------------------------------------------------------------
# State accessors
# ---------------------------------------------------------------------------

def get_current_holdings(conn: sqlite3.Connection, strategy: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM strategy_state WHERE Strategy = ? AND Units > 0", conn, params=(strategy,)
    )


def get_strategy_meta(conn: sqlite3.Connection, strategy: str) -> Optional[dict]:
    df = pd.read_sql_query("SELECT * FROM strategy_meta WHERE Strategy = ?", conn, params=(strategy,))
    return None if df.empty else df.iloc[0].to_dict()


def initialize_strategy(conn: sqlite3.Connection, strategy: str, notional_capital: float) -> None:
    """Seed strategy_meta before the first-ever rebalance. Safe to call once; raises if the
    strategy already has meta (use a different path to change capital after go-live)."""
    existing = get_strategy_meta(conn, strategy)
    if existing is not None:
        raise ValueError(f"Strategy {strategy} already initialized (InceptionDate={existing['InceptionDate']}). "
                          "Refusing to silently reset live state.")
    conn.execute(
        "INSERT INTO strategy_meta (Strategy, InceptionDate, NotionalCapital, CurrentCash) VALUES (?, NULL, ?, ?)",
        (strategy, notional_capital, notional_capital),
    )
    conn.commit()


def _available_capital(conn: sqlite3.Connection, strategy: str, holdings: pd.DataFrame, price_at: pd.Series) -> float:
    meta = get_strategy_meta(conn, strategy)
    if meta is None:
        raise ValueError(f"Strategy {strategy} not initialized — call initialize_strategy() first (go-live step).")
    holdings_value = 0.0
    for _, row in holdings.iterrows():
        px = price_at.get(row["Symbol"])
        if px is not None and pd.notna(px):
            holdings_value += row["Units"] * px
    return holdings_value + float(meta["CurrentCash"])


# ---------------------------------------------------------------------------
# 1. Decide — build_rebalance_plan
# ---------------------------------------------------------------------------

def build_rebalance_plan(
    conn: sqlite3.Connection,
    strategy: str,
    universe_symbols: list[str],
    rules: dict,
    closes: pd.DataFrame,
    circuit_flags: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> RebalancePlan:
    holdings = get_current_holdings(conn, strategy)
    held_symbols = set(holdings["Symbol"])

    # Evaluate ranking over the union of (universe, held) so a stock that fell out of its
    # index but is still held gets a fair rank instead of being silently force-excluded.
    ranking_universe = sorted(set(universe_symbols) | held_symbols)
    ranks = compute_ranks(closes, circuit_flags, signal_date, ranking_universe, rules)

    eval_symbols = [s for s in ranking_universe if s in closes.columns]
    elig = compute_eligibility(closes[eval_symbols], circuit_flags, signal_date, rules)

    worst_rank_held = rules["worst_rank_held"]
    top_n = rules["top_n"]

    survivors, sells = [], []
    for _, row in holdings.iterrows():
        sym = row["Symbol"]
        if sym in ranks.index and ranks.loc[sym, "AvgRank"] <= worst_rank_held:
            survivors.append({"Symbol": sym, "AvgRank": float(ranks.loc[sym, "AvgRank"]), "EntryDate": row["EntryDate"]})
        else:
            reasons = []
            if sym in elig.index and not bool(elig.loc[sym, "Eligible"]):
                if not bool(elig.loc[sym, "HistoryOK"]):
                    reasons.append("insufficient price history")
                if not bool(elig.loc[sym, "TrendOK"]):
                    reasons.append("below 200DEMA")
                if not bool(elig.loc[sym, "CircuitOK"]):
                    reasons.append(f"circuits {int(elig.loc[sym, 'CircuitDaysTrailing'])} > 25 in trailing year")
            elif sym in ranks.index:
                reasons.append(f"rank {ranks.loc[sym, 'AvgRank']:.2f} > worst-rank-held {worst_rank_held}")
            else:
                reasons.append("no longer rankable (missing recent price data)")
            sells.append({
                "Symbol": sym, "Reason": "; ".join(reasons),
                "UnitsBefore": float(row["Units"]), "EntryRank": row["EntryRank"],
            })

    open_slots = max(top_n - len(survivors), 0)
    candidate_pool = ranks[(~ranks.index.isin(held_symbols)) & (ranks.index.isin(universe_symbols))]
    new_entries_df = candidate_pool.head(open_slots)
    buys = [{"Symbol": sym, "AvgRank": float(r["AvgRank"])} for sym, r in new_entries_df.iterrows()]

    holds = [h for h in survivors]  # renamed for sheet clarity; still eligible & within top-n-hold band

    return RebalancePlan(
        strategy=strategy,
        signal_date=pd.Timestamp(signal_date).date().isoformat(),
        sells=sells,
        holds=holds,
        buys=buys,
        top_n=top_n,
        worst_rank_held=worst_rank_held,
    )


def save_pending_rebalance(conn: sqlite3.Connection, plan: RebalancePlan, force: bool = False) -> None:
    existing = pd.read_sql_query(
        "SELECT ExecDate FROM strategy_rebalances WHERE Strategy = ? AND SignalDate = ?",
        conn, params=(plan.strategy, plan.signal_date),
    )
    if not existing.empty and existing.iloc[0]["ExecDate"] is not None and not force:
        raise ValueError(
            f"Rebalance for {plan.strategy} on {plan.signal_date} was already settled "
            f"(ExecDate={existing.iloc[0]['ExecDate']}); refusing to overwrite. Pass force=True to override."
        )
    conn.execute(
        """
        INSERT INTO strategy_rebalances (Strategy, SignalDate, ExecDate, SheetJson)
        VALUES (?, ?, NULL, ?)
        ON CONFLICT(Strategy, SignalDate) DO UPDATE SET SheetJson=excluded.SheetJson, ExecDate=NULL
        """,
        (plan.strategy, plan.signal_date, plan.to_json()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 2. Apply — settle_pending_rebalance
# ---------------------------------------------------------------------------

def get_pending_rebalances(conn: sqlite3.Connection, strategy: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM strategy_rebalances WHERE Strategy = ? AND ExecDate IS NULL ORDER BY SignalDate",
        conn, params=(strategy,),
    )


def settle_pending_rebalance(
    conn: sqlite3.Connection,
    strategy: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    exec_date: pd.Timestamp,
) -> Optional[dict]:
    """Apply the oldest pending rebalance whose signal_date < exec_date, using exec_date's Open
    price (falling back to Close if Open is missing for a symbol that day). No-op if there is
    no pending rebalance ready to settle — safe to call every day."""
    pending = get_pending_rebalances(conn, strategy)
    if pending.empty:
        return None
    row = pending.iloc[0]
    signal_date = pd.Timestamp(row["SignalDate"])
    if signal_date >= pd.Timestamp(exec_date):
        return None

    plan = json.loads(row["SheetJson"])
    holdings = get_current_holdings(conn, strategy)
    holdings_by_symbol = {r["Symbol"]: r for _, r in holdings.iterrows()}

    def exec_price(symbol: str) -> Optional[float]:
        px = opens.loc[exec_date, symbol] if (exec_date in opens.index and symbol in opens.columns) else None
        if px is None or pd.isna(px):
            px = closes.loc[exec_date, symbol] if (exec_date in closes.index and symbol in closes.columns) else None
        return float(px) if px is not None and pd.notna(px) else None

    all_relevant_symbols = (
        {s["Symbol"] for s in plan["sells"]} | {h["Symbol"] for h in plan["holds"]} | {b["Symbol"] for b in plan["buys"]}
    )
    prices = {sym: exec_price(sym) for sym in all_relevant_symbols}
    missing_price = [sym for sym, px in prices.items() if px is None]
    if missing_price:
        # Can't settle fairly without a price for every leg — leave pending, caller should retry
        # once data catches up (logged by the caller as a data-freshness alert).
        return {"status": "deferred_missing_prices", "missing_symbols": missing_price}

    meta = get_strategy_meta(conn, strategy)
    total_value = float(meta["CurrentCash"])
    for _, h in holdings.iterrows():
        total_value += h["Units"] * prices.get(h["Symbol"], 0) if h["Symbol"] in prices else h["Units"] * closes.loc[:exec_date, h["Symbol"]].ffill().iloc[-1]

    top_n = plan["top_n"]
    final_holdings = plan["holds"] + plan["buys"]  # each: {Symbol, AvgRank, ...}
    n_final = len(final_holdings)
    target_slot_value = total_value / top_n if top_n > 0 else 0.0

    trades = []
    new_state_rows = []

    for s in plan["sells"]:
        sym = s["Symbol"]
        units_before = holdings_by_symbol[sym]["Units"] if sym in holdings_by_symbol else s["UnitsBefore"]
        trades.append({"Symbol": sym, "Side": "SELL", "Units": float(units_before), "Price": prices[sym], "Reason": s["Reason"]})

    for h in final_holdings:
        sym = h["Symbol"]
        units_before = holdings_by_symbol[sym]["Units"] if sym in holdings_by_symbol else 0.0
        units_after = target_slot_value / prices[sym] if prices[sym] else 0.0
        delta = units_after - units_before
        is_new = sym not in holdings_by_symbol
        side = "BUY" if is_new else ("REWEIGHT" if abs(delta) > 1e-9 else None)
        if side:
            reason = f"rank {h['AvgRank']:.2f}" if is_new else "monthly rebalance to equal weight"
            trades.append({"Symbol": sym, "Side": side, "Units": float(delta), "Price": prices[sym], "Reason": reason})
        entry_date = holdings_by_symbol[sym]["EntryDate"] if not is_new else pd.Timestamp(exec_date).date().isoformat()
        new_state_rows.append({
            "Strategy": strategy, "Symbol": sym, "Units": units_after,
            "EntryDate": entry_date, "EntryRank": h["AvgRank"],
        })

    invested_value = target_slot_value * n_final
    remaining_cash = total_value - invested_value

    conn.execute("DELETE FROM strategy_state WHERE Strategy = ?", (strategy,))
    if new_state_rows:
        conn.executemany(
            "INSERT INTO strategy_state (Strategy, Symbol, Units, EntryDate, EntryRank) VALUES (?, ?, ?, ?, ?)",
            [(r["Strategy"], r["Symbol"], r["Units"], r["EntryDate"], r["EntryRank"]) for r in new_state_rows],
        )

    exec_date_str = pd.Timestamp(exec_date).date().isoformat()
    if trades:
        conn.executemany(
            """
            INSERT INTO strategy_trades (Strategy, Date, Symbol, Side, Units, Price, Reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(Strategy, Date, Symbol, Side) DO UPDATE SET
                Units=excluded.Units, Price=excluded.Price, Reason=excluded.Reason
            """,
            [(strategy, exec_date_str, t["Symbol"], t["Side"], t["Units"], t["Price"], t["Reason"]) for t in trades],
        )

    conn.execute(
        "UPDATE strategy_meta SET CurrentCash = ?, InceptionDate = COALESCE(InceptionDate, ?) WHERE Strategy = ?",
        (remaining_cash, exec_date_str, strategy),
    )
    conn.execute(
        "UPDATE strategy_rebalances SET ExecDate = ? WHERE Strategy = ? AND SignalDate = ?",
        (exec_date_str, strategy, row["SignalDate"]),
    )
    conn.commit()

    return {
        "status": "settled", "exec_date": exec_date_str, "n_trades": len(trades),
        "n_holdings": n_final, "remaining_cash": remaining_cash, "total_value": total_value,
    }


# ---------------------------------------------------------------------------
# Corporate actions on held positions (bonus/split) — detection (NSE CA feed) is a separate,
# not-yet-automated ingestion step (see docs/momentum_strategies_plan.md Phase 1 gaps); this
# is the mechanism that applies a known factor once detected, so a 1:1 bonus (factor=2) keeps
# the NAV curve flat/continuous instead of showing a fake -50% day when the price halves.
# ---------------------------------------------------------------------------

def apply_corporate_action(conn: sqlite3.Connection, strategy: str, symbol: str, factor: float) -> Optional[float]:
    """Multiply held units of `symbol` by `factor` (e.g. 2.0 for a 1:1 bonus, 0.5 for a 1:2
    stock split reduction convention varies — pass the units-multiplier directly). No-op
    (returns None) if the symbol isn't currently held."""
    row = conn.execute(
        "SELECT Units FROM strategy_state WHERE Strategy = ? AND Symbol = ?", (strategy, symbol)
    ).fetchone()
    if row is None:
        return None
    new_units = row[0] * factor
    conn.execute(
        "UPDATE strategy_state SET Units = ? WHERE Strategy = ? AND Symbol = ?",
        (new_units, strategy, symbol),
    )
    conn.commit()
    return new_units


# ---------------------------------------------------------------------------
# 3. Mark-to-market — daily NAV roll-forward
# ---------------------------------------------------------------------------

def mark_to_market(conn: sqlite3.Connection, strategy: str, closes: pd.DataFrame, as_of_date: pd.Timestamp) -> Optional[dict]:
    meta = get_strategy_meta(conn, strategy)
    if meta is None or meta.get("InceptionDate") is None:
        return None  # not live yet — no inception fill has happened

    holdings = get_current_holdings(conn, strategy)
    value = float(meta["CurrentCash"])
    priced, unpriced = 0, []
    for _, h in holdings.iterrows():
        if as_of_date in closes.index and h["Symbol"] in closes.columns and pd.notna(closes.loc[as_of_date, h["Symbol"]]):
            value += h["Units"] * closes.loc[as_of_date, h["Symbol"]]
            priced += 1
        else:
            unpriced.append(h["Symbol"])
            # last known price carried forward rather than treated as zero
            last_known = closes[h["Symbol"]].loc[:as_of_date].ffill() if h["Symbol"] in closes.columns else pd.Series(dtype=float)
            if not last_known.empty:
                value += h["Units"] * last_known.iloc[-1]

    notional = float(meta["NotionalCapital"])
    nav = 100.0 * value / notional if notional else None

    date_str = pd.Timestamp(as_of_date).date().isoformat()
    conn.execute(
        """
        INSERT INTO strategy_nav (Strategy, Date, NAV, Cash, NHoldings) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(Strategy, Date) DO UPDATE SET NAV=excluded.NAV, Cash=excluded.Cash, NHoldings=excluded.NHoldings
        """,
        (strategy, date_str, nav, meta["CurrentCash"], len(holdings)),
    )
    conn.commit()
    return {"date": date_str, "nav": nav, "portfolio_value": value, "unpriced_holdings": unpriced}
