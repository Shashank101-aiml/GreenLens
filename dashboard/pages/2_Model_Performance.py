import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from client import AXIS, INK_2, LOWERS, MUTED, PILLARS, RAISES, SERIES, SURFACE, APIError, get, num, pct, show, style

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model performance")

try:
    s = get("/model/summary")
except APIError as exc:
    st.error(str(exc))
    st.stop()

MODEL_NAMES = {
    "sector_mean_baseline": "Sector-mean baseline",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}

st.markdown(
    f"""
**What the model does.** One regressor per pillar estimates a company's Environmental, Social and Governance
risk score from its business description (spaCy lemmas into TF-IDF), ESG theme densities (spaCy PhraseMatcher
over a WordNet-expanded lexicon), sector, headcount and controversy level. The total is the sum of the three estimates.

**How it is evaluated.** {s['cv']} on {s['n_train']} rated companies, so every number here comes from a model that
never saw that company. The baseline predicts each company's sector average. The rated pillar scores are never
model inputs: the total is exactly the sum of the pillars, so using them would be target leakage.
"""
)

model, baseline = s["total_metrics"]["model"], s["total_metrics"]["baseline"]
kpis = st.columns(4)
kpis[0].metric("Total ESG risk MAE", num(model["mae"], 2), delta=f"{model['mae'] - baseline['mae']:+.2f} vs baseline", delta_color="inverse")
kpis[1].metric("Total ESG risk R²", num(model["r2"], 3), delta=f"{model['r2'] - baseline['r2']:+.3f} vs baseline")
kpis[2].metric(
    "Risk-level accuracy",
    pct(model["level_accuracy"]),
    delta=f"{(model['level_accuracy'] - baseline['level_accuracy']) * 100:+.1f} pts vs baseline",
)
kpis[3].metric("Within one risk band", pct(model["level_within_one"]))

st.subheader("Cross-validated error by pillar")
st.caption("Blue marks the model selected for each pillar (lowest mean absolute error).")
for col, (pillar, label) in zip(st.columns(3), PILLARS):
    metrics, chosen = s["metrics"][pillar], s["model_names"][pillar]
    fig = go.Figure(
        go.Bar(
            x=[MODEL_NAMES[n] for n in MODEL_NAMES],
            y=[metrics[n]["mae"] for n in MODEL_NAMES],
            marker_color=[SERIES[0] if n == chosen else MUTED for n in MODEL_NAMES],
            text=[f"{metrics[n]['mae']:.2f}" for n in MODEL_NAMES],
            textposition="outside",
            textfont_color=INK_2,
            hovertemplate="%{x}<br>MAE %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=f"{label}: MAE (lower is better)", yaxis_title="Mean absolute error", showlegend=False)
    with col:
        show(style(fig, 330))

with st.expander("All cross-validation metrics (table)"):
    rows = [
        {"Pillar": label, "Model": MODEL_NAMES[n], "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"],
         "Selected": "yes" if n == s["model_names"][pillar] else ""}
        for pillar, label in PILLARS
        for n, m in s["metrics"][pillar].items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.subheader("Out-of-fold predictions vs rated scores")
oof = pd.DataFrame(s["oof"])
for col, (pillar, label) in zip(st.columns(3), PILLARS):
    pred = "pred_" + pillar.removeprefix("esg_")
    hi = float(max(oof[pillar].max(), oof[pred].max())) * 1.05
    fig = go.Figure()
    fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=AXIS, width=1), hoverinfo="skip", showlegend=False)
    fig.add_scatter(
        x=oof[pillar],
        y=oof[pred],
        mode="markers",
        marker=dict(size=8, color=SERIES[0], opacity=0.7, line=dict(width=2, color=SURFACE)),
        customdata=oof[["name"]],
        hovertemplate="<b>%{customdata[0]}</b><br>Rated %{x:.1f}<br>Predicted %{y:.1f}<extra></extra>",
        showlegend=False,
    )
    fig.update_layout(title=label, xaxis_title="Rated score", yaxis_title="Predicted (out-of-fold)", xaxis_range=[0, hi], yaxis_range=[0, hi])
    with col:
        show(style(fig, 340))
st.caption("Points on the diagonal are perfect predictions.")

st.subheader("What the text model learned")
pillar_label = st.selectbox("Pillar", [label for _, label in PILLARS])
pillar = next(p for p, label in PILLARS if label == pillar_label)
terms = s["top_terms"][pillar]
weights = pd.DataFrame(terms["lowers_risk"][:10] + terms["raises_risk"][:10]).sort_values("weight")
fig = go.Figure(
    go.Bar(
        x=weights["weight"],
        y=weights["term"],
        orientation="h",
        marker_color=[RAISES if w > 0 else LOWERS for w in weights["weight"]],
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    )
)
fig.update_layout(title=f"Terms that move the {pillar_label.lower()} estimate", xaxis_title="Ridge coefficient on the TF-IDF term")
show(style(fig, 520))
st.caption(
    "Positive weights (red) raise the estimated risk; negative weights (blue) lower it. Sector is also a model input, "
    "so weights are relative to the sector average: a term typical of the lower-risk companies within a high-risk "
    "sector gets a negative weight even if the sector itself is high-risk."
)

with st.expander("ESG theme lexicon (seed terms expanded with NLTK WordNet)"):
    st.dataframe(
        pd.DataFrame([{"Theme": k.replace("_", " "), "Terms": ", ".join(v)} for k, v in s["lexicon"].items()]),
        hide_index=True,
        width="stretch",
    )
st.caption(f"Model version {s['version']}, trained {s['trained_at']}.")
