import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from client import PILLARS, RISK_ORDER, SERIES, SURFACE, APIError, get, num, pct, show, style

st.set_page_config(page_title="ESG Performance Analytics", layout="wide")
st.title("ESG Performance Analytics")
st.caption(
    "S&P 500 companies: Sustainalytics ESG risk ratings, ML estimates from business descriptions, "
    "and one-year stock performance. Lower ESG risk scores are better."
)

try:
    companies = pd.DataFrame(get("/companies"))
    overview = get("/analytics/overview")
except APIError as exc:
    st.error(str(exc))
    st.stop()

chosen = st.multiselect("Filter sectors", sorted(companies["sector"].dropna().unique()), placeholder="All sectors")
view = companies[companies["sector"].isin(chosen)] if chosen else companies
rated = view.dropna(subset=["esg_total"])
benchmark = overview.get("benchmark") or {}

kpis = st.columns(5)
kpis[0].metric("Companies", f"{len(view):,}")
kpis[1].metric("With ESG rating", f"{len(rated):,}")
kpis[2].metric("Average ESG risk", num(rated["esg_total"].mean()))
kpis[3].metric("High or severe risk", pct(rated["risk_level"].isin(["High", "Severe"]).mean() if len(rated) else None, 0))
kpis[4].metric(
    "Median 1-year return", pct(view["return_1y"].median()), help=f"S&P 500 (SPY) over the same year: {pct(benchmark.get('return_1y'))}"
)

left, right = st.columns(2)
with left:
    counts = rated["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    fig = go.Figure(
        go.Bar(x=counts.index, y=counts.values, marker_color=SERIES[0], hovertemplate="%{x}: %{y} companies<extra></extra>")
    )
    fig.update_layout(title="Companies by ESG risk level", yaxis_title="Companies")
    show(style(fig))
with right:
    by_sector = rated.groupby("sector")[[col for col, _ in PILLARS]].mean()
    by_sector = by_sector.loc[by_sector.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for color, (col, label) in zip(SERIES, PILLARS):
        fig.add_bar(
            y=by_sector.index,
            x=by_sector[col],
            name=label,
            orientation="h",
            marker_color=color,
            marker_line=dict(color=SURFACE, width=2),
            hovertemplate=f"%{{y}}<br>{label}: %{{x:.1f}}<extra></extra>",
        )
    fig.update_layout(barmode="stack", title="Average ESG risk by pillar and sector", xaxis_title="Average risk score")
    show(style(fig))

points = view.dropna(subset=["esg_total", "return_1y"])
fig = go.Figure(
    go.Scatter(
        x=points["esg_total"],
        y=points["return_1y"],
        mode="markers",
        marker=dict(size=9, color=SERIES[0], opacity=0.75, line=dict(width=2, color=SURFACE)),
        customdata=points[["name", "sector"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>ESG risk %{x:.1f}<br>1-year return %{y:.1%}<extra></extra>",
    )
)
fig.update_layout(
    title="ESG risk vs one-year stock return",
    xaxis_title="ESG risk score (lower is better)",
    yaxis_title="1-year return",
    yaxis_tickformat=".0%",
)
show(style(fig, 420))
if len(points) > 2:
    rho = points[["esg_total", "return_1y"]].rank().corr().iloc[0, 1]
    st.caption(f"Rank correlation between ESG risk and 1-year return: ρ = {rho:.2f} across {len(points)} companies. Correlation, not causation.")

with st.expander("View chart data"):
    st.dataframe(by_sector.round(2).rename(columns=dict(PILLARS)), width="stretch")

table_cols = {"symbol": "Symbol", "name": "Company", "sector": "Sector", "esg_total": "ESG risk", "risk_level": "Level"}
high, low = st.columns(2)
high.subheader("Highest ESG risk")
high.dataframe(rated.nlargest(10, "esg_total")[list(table_cols)].rename(columns=table_cols), hide_index=True, width="stretch")
low.subheader("Lowest ESG risk")
low.dataframe(rated.nsmallest(10, "esg_total")[list(table_cols)].rename(columns=table_cols), hide_index=True, width="stretch")
