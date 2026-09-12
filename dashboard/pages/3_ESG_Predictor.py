import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from client import SERIES, APIError, get, num, post, show, style

st.set_page_config(page_title="ESG Risk Predictor", layout="wide")
st.title("Estimate ESG risk from a description")
st.caption(
    "Paste a company's business description. The same spaCy pipeline and pillar models behind the dashboard "
    "produce the estimate. These are model estimates, not ratings."
)

try:
    sectors = [row["sector"] for row in get("/analytics/sectors")]
except APIError as exc:
    st.error(str(exc))
    st.stop()

with st.form("predict"):
    description = st.text_area(
        "Business description",
        height=180,
        placeholder="e.g. The company explores for, produces and refines crude oil and natural gas, and operates pipelines in the United States and Canada...",
    )
    c1, c2, c3 = st.columns(3)
    sector = c1.selectbox("Sector", sectors)
    employees = c2.number_input("Full-time employees", min_value=0, value=10_000, step=1_000)
    controversy = c3.selectbox("Controversy level", ["Unknown", "None", "Low", "Moderate", "Significant", "High", "Severe"])
    submitted = st.form_submit_button("Estimate ESG risk", type="primary")

if submitted:
    if len(description.strip()) < 30:
        st.warning("Enter a description of at least 30 characters.")
        st.stop()
    payload = {
        "description": description.strip(),
        "sector": sector,
        "employees": int(employees),
        "controversy_level": None if controversy == "Unknown" else controversy,
    }
    try:
        r = post("/model/predict", json=payload)
    except APIError as exc:
        st.error(str(exc))
        st.stop()

    kpis = st.columns(4)
    kpis[0].metric("Estimated total ESG risk", f"{num(r['total'])} · {r['risk_level']}")
    for col, (key, label) in zip(kpis[1:], [("environment", "Environment"), ("social", "Social"), ("governance", "Governance")]):
        col.metric(label, num(r["pillars"][key]))

    themes = pd.Series({k.replace("_", " "): v for k, v in r["themes"].items() if v > 0}).sort_values()
    if themes.empty:
        st.info("No ESG theme terms were detected in this description.")
    else:
        fig = go.Figure(go.Bar(x=themes.values, y=themes.index, orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}: %{x:.2f} per 100 tokens<extra></extra>"))
        fig.update_layout(title="ESG themes detected (spaCy)", xaxis_title="Matches per 100 tokens")
        show(style(fig))
    st.caption(f"Model version {r['model_version']}.")
