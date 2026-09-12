import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Validated reference data-viz palette: categorical slots 1-3, de-emphasis gray, diverging poles, chart ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#c3c2b7"
RAISES, LOWERS = "#e34948", "#2a78d6"
INK, INK_2, INK_MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

RISK_ORDER = ["Negligible", "Low", "Medium", "High", "Severe"]
PILLARS = [("esg_environment", "Environment"), ("esg_social", "Social"), ("esg_governance", "Governance")]


class APIError(RuntimeError):
    pass


def _request(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=90, **kwargs)
    except httpx.HTTPError as exc:
        raise APIError(f"Cannot reach the API at {API_URL}. Start it with `uvicorn app.main:app` in backend/. ({exc})") from exc
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(f"API error {response.status_code}: {detail}")
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def get(path: str, params: dict | None = None):
    return _request("GET", path, params=params)


def post(path: str, **kwargs):
    return _request("POST", path, **kwargs)


def pct(value, digits: int = 1) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}%}"


def num(value, digits: int = 1) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:,.{digits}f}"


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=13),
        title_font=dict(color=INK, size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", font_color=INK, bordercolor=GRID),
        bargap=0.3,
        barcornerradius=4,
    )
    axis = dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont_color=INK_MUTED, title_font_color=INK_2)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, width="stretch", config={"displayModeBar": False})
