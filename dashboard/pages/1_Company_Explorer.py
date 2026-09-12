import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from client import INK_2, SERIES, APIError, get, num, pct, post, show, style

st.set_page_config(page_title="Company Explorer", layout="wide")
st.title("Company explorer")

try:
    companies = pd.DataFrame(get("/companies")).sort_values("name")
    benchmark = get("/analytics/overview").get("benchmark") or {}
except APIError as exc:
    st.error(str(exc))
    st.stop()

labels = dict(zip(companies["symbol"], companies["name"] + " (" + companies["symbol"] + ")"))
symbols = list(labels)
symbol = st.selectbox("Company", symbols, index=symbols.index("MSFT") if "MSFT" in labels else 0, format_func=labels.get)
c = get(f"/companies/{symbol}")

st.subheader(c["name"])
st.caption(f"{c['sector'] or 'Unknown sector'} · {c['industry'] or 'Unknown industry'} · {num(c['employees'], 0)} employees")

esg = st.columns(4)
esg[0].metric("Rated ESG risk (Sustainalytics)", f"{num(c['esg_total'])} · {c['risk_level'] or 'unrated'}")
esg[1].metric(
    "Model estimate",
    f"{num(c['pred_total'])} · {c['pred_level'] or 'n/a'}",
    help="Out-of-fold: predicted by a model that never saw this company."
    if c["out_of_fold"]
    else "No published rating; estimated by the model.",
)
esg[2].metric("Controversy level", c["controversy_level"] or "n/a")
esg[3].metric("ESG risk percentile", num(c["risk_percentile"], 0))

perf = st.columns(4)
perf[0].metric("1-year return", pct(c["return_1y"]), help=f"S&P 500 (SPY): {pct(benchmark.get('return_1y'))}")
perf[1].metric("Volatility (annualised)", pct(c["volatility_1y"]))
perf[2].metric("Max drawdown", pct(c["max_drawdown_1y"]))
perf[3].metric("Sharpe ratio", num(c["sharpe_1y"], 2), help="Excess return over the 3-month T-bill divided by volatility.")
if c["as_of"]:
    st.caption(f"Stock performance over the year to {c['as_of']}.")

left, right = st.columns(2)
with left:
    pillar_labels = ["Environment", "Social", "Governance"]
    rated_values = [c["esg_environment"], c["esg_social"], c["esg_governance"]]
    predicted_values = [c["pred_environment"], c["pred_social"], c["pred_governance"]]
    fig = go.Figure()
    if any(v is not None for v in rated_values):
        fig.add_bar(x=pillar_labels, y=rated_values, name="Rated (Sustainalytics)", marker_color=SERIES[0],
                    text=[num(v) for v in rated_values], textposition="outside", textfont_color=INK_2)
    fig.add_bar(x=pillar_labels, y=predicted_values, name="Model estimate", marker_color=SERIES[1],
                text=[num(v) for v in predicted_values], textposition="outside", textfont_color=INK_2)
    fig.update_layout(barmode="group", title="Risk by pillar", yaxis_title="Risk score (lower is better)")
    show(style(fig))
with right:
    themes = pd.Series({k.replace("_", " "): v for k, v in (c.get("themes") or {}).items() if v > 0}).sort_values()
    if themes.empty:
        st.info("No ESG theme terms were found in this company's description.")
    else:
        fig = go.Figure(go.Bar(x=themes.values, y=themes.index, orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}: %{x:.2f} per 100 tokens<extra></extra>"))
        fig.update_layout(title="ESG themes in the business description", xaxis_title="Matches per 100 tokens (spaCy)")
        show(style(fig))

with st.expander("Business description"):
    st.write(c["description"] or "Not available.")

st.divider()
st.subheader("AI ESG briefing (Gemini)")
st.caption("Written only from the facts listed below, returned as structured JSON, and cached in PostgreSQL per input.")
key = f"report_{symbol}"
generate, regenerate = st.columns([1, 5])
try:
    if generate.button("Generate briefing", type="primary"):
        with st.spinner("Asking Gemini..."):
            st.session_state[key] = post(f"/companies/{symbol}/report")
    if key in st.session_state and regenerate.button("Regenerate"):
        with st.spinner("Asking Gemini..."):
            st.session_state[key] = post(f"/companies/{symbol}/report", params={"refresh": "true"})
except APIError as exc:
    st.error(str(exc))

result = st.session_state.get(key)
if result:
    report = result["report"]
    st.markdown(report["summary"])
    cols = st.columns(3)
    for col, (label, field) in zip(cols, [("Environmental", "environment"), ("Social", "social"), ("Governance", "governance")]):
        col.markdown(f"**{label}**")
        col.markdown(report[field])
    risks, strengths = st.columns(2)
    risks.markdown("**Key risks**\n" + "\n".join(f"- {r}" for r in report["key_risks"]))
    strengths.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in report["strengths"]))
    if report["data_caveats"]:
        st.caption("Caveats: " + " ".join(report["data_caveats"]))
    st.caption(f"{result['model']} · {'from cache' if result['cached'] else 'freshly generated'} · {result['created_at']}")
    with st.expander("Facts sent to Gemini"):
        st.json(result["facts"])
