# Lucas Bedleg
# Email: lulu.bm9000@gmail.com
# CFO Copilot Project
# Date: September 26th 2025
# File: classifier.py
# Description: Provides intent classification and date parsing for the CFO Copilot, detecting user queries about revenue, 
# margins, opex, or cash flow, and extracting relevant time periods from natural language.

import re
from datetime import datetime
from dateutil import parser as dateparser

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January","February","March","April","May","June",
     "July","August","September","October","November","December"], start=1)}

def classify_intent(q: str) -> str:
    q = q.lower()
    if "revenue" in q and "budget" in q:
        return "revenue_vs_budget"
    if "gross margin" in q:
        return "gross_margin_trend"
    if "opex" in q and ("breakdown" in q or "by category" in q):
        return "opex_breakdown"
    if "cash runway" in q or ("runway" in q and "cash" in q):
        return "cash_runway"
    return "unknown"

def parse_month_in_question(q: str) -> datetime | None:
    """
    Returns a datetime set to first of month if a 'June 2025' like phrase exists.
    """
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}", q, re.I)
    if m:
        return dateparser.parse(m.group(0)).replace(day=1)
    # support YYYY-MM or YYYY/MM
    m2 = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", q)
    if m2:
        return datetime(int(m2.group(1)), int(m2.group(2)), 1)
    return None

def parse_last_n_months(q: str, default_n: int = 3) -> int:
    m = re.search(r"last\s+(\d+)\s+months?", q, re.I)
    return int(m.group(1)) if m else default_n