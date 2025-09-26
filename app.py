# Lucas Bedleg
# Email: lulu.bm9000@gmail.com
# CFO Copilot Project
# Date: September 26th 2025
# File: app.py
# Description: A Streamlit app that serves as a CFO Copilot, allowing users to upload financial data and 
# ask natural language finance questions to get answers and visualizations.

import streamlit as st
from agent import classify_intent, load_data, answer

@st.cache_data(show_spinner=False)
def load_all():
    return load_data("fixtures")

st.set_page_config(page_title="CFO Copilot", layout="wide")
st.title("💼 CFO Copilot")

with st.sidebar:
    st.markdown("**Data**")
    st.caption("Place CSVs in `fixtures/`:\n- actuals.csv\n- budget.csv\n- fx.csv\n- cash.csv")

actuals, budget, fx, cash = load_all()

q = st.text_input("Ask a finance question (e.g., 'What was June 2025 revenue vs budget?')")

if st.button("Submit") or q:
    intent = classify_intent(q or "")
    with st.spinner("Thinking..."):
        text, fig = answer(intent, q or "", actuals, budget, fx, cash)
    st.success(text)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)