# Lucas Bedleg
# Email: lulu.bm9000@gmail.com
# CFO Copilot Project
# Date: September 26th 2025
# File: test_metrics.py
# Description: Defines a simple pytest unit test that checks the gross_margin_pct function: 
# given $1000 revenue and $600 COGS for June 2025, the function should return a 40% gross margin.

import pandas as pd
from datetime import datetime
from agent.tools import gross_margin_pct # type: ignore

def test_gross_margin_pct():
    m = pd.to_datetime("2025-06-01")
    df = pd.DataFrame({
        "month":[m,m],
        "account":["Revenue","COGS"],
        "usd":[1000,600]
    })
    assert abs(gross_margin_pct(df, m) - 40.0) < 1e-6
