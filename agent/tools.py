# Lucas Bedleg
# Email: lulu.bm9000@gmail.com
# CFO Copilot Project
# Date: September 26th 2025
# File: tools.py
# Description: Loads/normalizes finance CSVs, converts to USD, computes key metrics (revenue vs budget, gross margin, opex breakdown, 
# cash runway), builds Plotly charts, and orchestrates intent-specific answers for the CFO Copilot.

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
import plotly.express as px

# Column alias map
ALIASES = {
    "month": ["month", "date", "period"],
    "entity": ["entity", "business_unit", "bu"],
    "account": ["account", "gl_account", "acct", "account_category"],
    "currency": ["currency", "curr", "fx_ccy"],
    "amount": ["amount", "value", "amt", "usd_amount"],
    "cash_balance": ["cash_balance", "cash", "ending_cash", "cash_usd"],
    "to_usd": ["to_usd", "fx_to_usd", "rate_to_usd"],
}

# Robust readers / normalizers
def _read_smart(path: str) -> pd.DataFrame:
    """
    Auto-detect delimiter (handles semicolons) and fall back to ';' if needed.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        # Safety: some exports become one giant column like "month;entity;..."
        if df.shape[1] == 1 and ";" in df.columns[0]:
            df = pd.read_csv(path, sep=";")
        return df
    except Exception:
        return pd.read_csv(path, sep=";")

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

def _rename_by_alias(df: pd.DataFrame, wanted: str) -> pd.DataFrame:
    for c in list(df.columns):
        if c.lower() in [w.lower() for w in ALIASES.get(wanted, [])]:
            return df.rename(columns={c: wanted})
    return df

def _normalize_month_col(df: pd.DataFrame, col: str = "month") -> None:
    if col not in df.columns:
        raise KeyError(f"Missing `{col}` column in: {df.columns.tolist()}")
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.to_period("M").dt.to_timestamp()

def _to_number(series: pd.Series) -> pd.Series:
    """
    Convert numbers that may include thousands separators or comma decimals.
    """
    if series.dtype.kind in "if":
        return series
    s = (
        series.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)  # support comma decimals
    )
    return pd.to_numeric(s, errors="coerce")

def _normalize_account(series: pd.Series) -> pd.Series:
    """
    Trim spaces; make 'opex:' prefix consistent as 'Opex:' regardless of case.
    """
    s = series.astype(str).str.strip()
    mask = s.str.contains(r"^opex:", case=False, na=False)
    s.loc[mask] = "Opex:" + s.loc[mask].str.split(":", n=1).str[1].str.strip()
    return s

def _ensure_usd_fx(fx: pd.DataFrame, months: List[pd.Timestamp]) -> pd.DataFrame:
    """
    Ensure there is a USD row (to_usd=1.0) for all months we might merge on.
    """
    fx = fx.copy()
    if "currency" not in fx.columns:
        fx["currency"] = "USD"
    if "to_usd" not in fx.columns:
        fx["to_usd"] = 1.0

    have = set(zip(fx["month"], fx["currency"]))
    missing = [(m, "USD") for m in months if (m, "USD") not in have]
    if missing:
        fx = pd.concat(
            [
                fx,
                pd.DataFrame({"month": [m for m, _ in missing], "currency": "USD", "to_usd": 1.0}),
            ],
            ignore_index=True,
        )
    return fx

# Public API
def load_data(fixtures_dir: str = "fixtures"):
    # Read (delimiter-smart) and drop junk cols
    actuals = _drop_unnamed(_read_smart(f"{fixtures_dir}/actuals.csv"))
    budget  = _drop_unnamed(_read_smart(f"{fixtures_dir}/budget.csv"))
    fx      = _drop_unnamed(_read_smart(f"{fixtures_dir}/fx.csv"))
    cash    = _drop_unnamed(_read_smart(f"{fixtures_dir}/cash.csv"))

    # Rename by aliases so headers can vary
    for wanted in ["month", "entity", "account", "currency", "amount"]:
        actuals = _rename_by_alias(actuals, wanted)
        budget  = _rename_by_alias(budget,  wanted)
    for wanted in ["month", "currency", "to_usd"]:
        fx = _rename_by_alias(fx, wanted)
    for wanted in ["month", "currency", "cash_balance"]:
        cash = _rename_by_alias(cash, wanted)

    # Parse dates
    _normalize_month_col(actuals, "month")
    _normalize_month_col(budget,  "month")
    _normalize_month_col(fx,      "month")
    _normalize_month_col(cash,    "month")

    # Normalize numerics
    actuals["amount"] = _to_number(actuals["amount"])
    budget["amount"]  = _to_number(budget["amount"])
    if "to_usd" in fx:
        fx["to_usd"] = _to_number(fx["to_usd"])
    cash["cash_balance"] = _to_number(cash["cash_balance"])

    # Ensure currency column for cash
    if "currency" not in cash.columns:
        cash["currency"] = "USD"

    # Normalize account labels 
    actuals["account"] = _normalize_account(actuals["account"])
    budget["account"]  = _normalize_account(budget["account"])

    # Make sure we have USD FX rows for all months in the data
    all_months = sorted(set(actuals["month"]) | set(budget["month"]) | set(cash["month"]))
    fx = _ensure_usd_fx(fx, all_months)

    return actuals, budget, fx, cash

def to_usd(df: pd.DataFrame, fx: pd.DataFrame, amount_col="amount") -> pd.DataFrame:
    merged = df.merge(fx, on=["month", "currency"], how="left")
    merged["to_usd"] = merged["to_usd"].fillna(1.0)
    merged["usd"] = merged[amount_col] * merged["to_usd"]
    return merged

# Metrics
def revenue_usd(df_actuals_usd: pd.DataFrame, df_budget_usd: pd.DataFrame, month: datetime) -> Tuple[float, float]:
    a = df_actuals_usd[(df_actuals_usd["month"] == month) & (df_actuals_usd["account"].str.lower() == "revenue")]["usd"].sum()
    b = df_budget_usd [(df_budget_usd ["month"] == month) & (df_budget_usd ["account"].str.lower() == "revenue")]["usd"].sum()
    return float(a), float(b)

def cogs_usd(df: pd.DataFrame, month: datetime) -> float:
    return float(df[(df["month"] == month) & (df["account"].str.lower() == "cogs")]["usd"].sum())

def opex_by_category_usd(df: pd.DataFrame, month: datetime) -> pd.Series:
    m = df[(df["month"] == month) & (df["account"].str.startswith("Opex:", na=False))]
    cat = m["account"].str.replace("Opex:", "", regex=False)
    return m.assign(category=cat).groupby("category")["usd"].sum().sort_values(ascending=False)

def ebitda_proxy(a_usd: pd.DataFrame, month: datetime) -> float:
    rev  = float(a_usd[(a_usd["month"] == month) & (a_usd["account"].str.lower() == "revenue")]["usd"].sum())
    cogs = float(a_usd[(a_usd["month"] == month) & (a_usd["account"].str.lower() == "cogs")]["usd"].sum())
    opex = float(a_usd[(a_usd["month"] == month) & (a_usd["account"].str.startswith("Opex:", na=False))]["usd"].sum())
    return rev - cogs - opex

def gross_margin_pct(a_usd: pd.DataFrame, month: datetime) -> float:
    rev = float(a_usd[(a_usd["month"] == month) & (a_usd["account"].str.lower() == "revenue")]["usd"].sum())
    if rev == 0:
        return np.nan
    cogs = float(a_usd[(a_usd["month"] == month) & (a_usd["account"].str.lower() == "cogs")]["usd"].sum())
    return (rev - cogs) / rev * 100.0

def cash_runway_months(cash_usd: pd.DataFrame, a_usd: pd.DataFrame, ref_month: datetime) -> float:
    months = pd.period_range(ref_month.to_period("M") - 2, ref_month.to_period("M"), freq="M").to_timestamp()
    last3 = a_usd[a_usd["month"].isin(months)]
    rev  = last3[last3["account"].str.lower() == "revenue"].groupby("month")["usd"].sum()
    cogs = last3[last3["account"].str.lower() == "cogs"].groupby("month")["usd"].sum()
    opex = last3[last3["account"].str.startswith("Opex:", na=False)].groupby("month")["usd"].sum()
    monthly_burn = (cogs.reindex(months, fill_value=0) + opex.reindex(months, fill_value=0) - rev.reindex(months, fill_value=0)).clip(lower=0)
    avg_burn = monthly_burn.mean()
    cash_now = cash_usd[cash_usd["month"] == ref_month]["usd"].sum()
    if avg_burn <= 0:
        return float("inf")
    return cash_now / avg_burn

# Charts
def chart_gm_trend(a_usd: pd.DataFrame, months: List[pd.Timestamp]):
    rows = [{"month": m, "gm_pct": gross_margin_pct(a_usd, m)} for m in months]
    df = pd.DataFrame(rows)
    fig = px.line(df, x="month", y="gm_pct", markers=True, title="Gross Margin % Trend")
    fig.update_layout(yaxis_ticksuffix="%", xaxis_title="", yaxis_title="GM %")
    return fig

def chart_opex_breakdown(series_by_cat: pd.Series, month: datetime):
    df = series_by_cat.reset_index()
    df.columns = ["category", "usd"]
    fig = px.bar(df, x="category", y="usd", title=f"Opex by Category — {month.strftime('%b %Y')}")
    fig.update_layout(xaxis_title="", yaxis_title="USD")
    return fig

def chart_rev_vs_budget(a_usd_m: float, b_usd_m: float, month: datetime):
    df = pd.DataFrame({"type": ["Actual", "Budget"], "usd": [a_usd_m, b_usd_m]})
    fig = px.bar(df, x="type", y="usd", text_auto=".2s", title=f"Revenue vs Budget — {month.strftime('%b %Y')}")
    fig.update_layout(xaxis_title="", yaxis_title="USD")
    return fig

# Orchestration for the app
def answer(intent: str, question: str, actuals: pd.DataFrame, budget: pd.DataFrame, fx: pd.DataFrame, cash: pd.DataFrame):
    a_usd = to_usd(actuals, fx)
    b_usd = to_usd(budget,  fx)
    c_usd = to_usd(cash.rename(columns={"cash_balance": "amount"}), fx, amount_col="amount")

    month = a_usd["month"].max()

    from .classifier import parse_month_in_question, parse_last_n_months
    parsed = parse_month_in_question(question)
    if parsed is not None:
        month = parsed

    if intent == "revenue_vs_budget":
        a, b = revenue_usd(a_usd, b_usd, month)
        fig = chart_rev_vs_budget(a, b, month)
        delta = a - b
        pct = (delta / b * 100.0) if b else np.nan
        text = f"{month.strftime('%B %Y')} Revenue — Actual: ${a:,.0f}, Budget: ${b:,.0f} ({delta:+,.0f} vs budget; {pct:+.1f}%)."
        return text, fig

    if intent == "gross_margin_trend":
        n = parse_last_n_months(question, default_n=3)
        months = pd.period_range(month.to_period("M") - (n - 1), month.to_period("M")).to_timestamp().tolist()
        fig = chart_gm_trend(a_usd, months)
        latest_pct = gross_margin_pct(a_usd, month)
        text = f"Gross Margin % for last {n} months; latest ({month.strftime('%b %Y')}): {latest_pct:.1f}%."
        return text, fig

    if intent == "opex_breakdown":
        s = opex_by_category_usd(a_usd, month)
        fig = chart_opex_breakdown(s, month)
        top = s.head(3)
        text = "Opex breakdown - " + ", ".join([f"{k}: ${v:,.0f}" for k, v in top.items()])
        return text, fig

    if intent == "cash_runway":
        months_runway = cash_runway_months(c_usd, a_usd, month)
        text = f"Estimated cash runway: {months_runway:.1f} months (based on avg net burn over last 3 months)."
        fig = px.line(c_usd.groupby("month")["usd"].sum().reset_index(),
                      x="month", y="usd", markers=True, title="Cash Balance Trend")
        fig.update_layout(xaxis_title="", yaxis_title="USD")
        return text, fig

    return "Sorry - I didn't understand. Try: 'What was June 2025 revenue vs budget?'", None