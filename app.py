import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Climate Indices - South America",
    layout="wide",
)

BASE_URL = "https://raw.githubusercontent.com/dbrisaro/climate-dashboard/main/data"

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_all():
    oni    = pd.read_csv(f"{BASE_URL}/oni.csv",    parse_dates=["date"])
    mei    = pd.read_csv(f"{BASE_URL}/mei.csv",    parse_dates=["date"])
    sam    = pd.read_csv(f"{BASE_URL}/sam.csv",    parse_dates=["date"])
    iod    = pd.read_csv(f"{BASE_URL}/iod.csv",    parse_dates=["date"])
    nino34 = pd.read_csv(f"{BASE_URL}/nino34.csv", parse_dates=["date"])
    nino12 = pd.read_csv(f"{BASE_URL}/nino12.csv", parse_dates=["date"])
    return oni, mei, sam, iod, nino34, nino12

@st.cache_data(ttl=3600)
def load_enso_probs():
    """Load official ENSO seasonal probabilities scraped from IRI/CPC."""
    try:
        df = pd.read_csv(f"{BASE_URL}/enso_probs.csv")
        if len(df) >= 3 and {"season","p_nina","p_neutral","p_nino"}.issubset(df.columns):
            return df
        return None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def load_seas5_mean():
    """Load SEAS5 ensemble-mean Nino3.4 forecast if available."""
    try:
        df = pd.read_csv(
            f"{BASE_URL}/forecasts/nino34_seas5_mean.csv",
            parse_dates=["forecast_date"],
        )
        return df if len(df) > 0 else None
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_iri_figures():
    """Scrape IRI ENSO page to get current figure URLs."""
    try:
        r = requests.get(
            "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/",
            timeout=15,
        )
        soup = BeautifulSoup(r.text, "html.parser")
        imgs = [i.get("src", "") for i in soup.find_all("img")]
        figures = [u for u in imgs if "wp-content/uploads" in u and "figure" in u]
        seen, unique = set(), []
        for f in figures:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique
    except Exception:
        return []

oni, mei, sam, iod, nino34, nino12 = load_all()

# ── Helpers ───────────────────────────────────────────────────────────────────

def latest(df, col):
    return df.dropna(subset=[col]).iloc[-1][col]

def enso_label(val):
    if val >= 1.5:  return "Strong El Nino"
    if val >= 1.0:  return "Moderate El Nino"
    if val >= 0.5:  return "Weak El Nino"
    if val <= -1.5: return "Strong La Nina"
    if val <= -1.0: return "Moderate La Nina"
    if val <= -0.5: return "Weak La Nina"
    return "Neutral"

def filt(df, y_start, y_end):
    return df[(df["date"].dt.year >= y_start) & (df["date"].dt.year <= y_end)]

def threshold_shapes(ymin=-3, ymax=3):
    return [
        dict(type="rect", xref="paper", yref="y",
             x0=0, x1=1, y0=0.5, y1=ymax,
             fillcolor="rgba(220,50,50,0.08)", line_width=0),
        dict(type="rect", xref="paper", yref="y",
             x0=0, x1=1, y0=ymin, y1=-0.5,
             fillcolor="rgba(50,100,220,0.08)", line_width=0),
        dict(type="line", xref="paper", yref="y",
             x0=0, x1=1, y0=0.5, y1=0.5,
             line=dict(color="rgba(220,50,50,0.4)", width=1, dash="dot")),
        dict(type="line", xref="paper", yref="y",
             x0=0, x1=1, y0=-0.5, y1=-0.5,
             line=dict(color="rgba(50,100,220,0.4)", width=1, dash="dot")),
    ]

def make_ts(df, x_col, y_col, title, ylabel, color, shapes=None, y_start=1990, y_end=2100):
    fig = go.Figure()
    d = filt(df, y_start, y_end)
    fig.add_trace(go.Scatter(
        x=d[x_col], y=d[y_col],
        mode="lines", name=y_col,
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba"),
    ))
    fig.add_hline(y=0, line_color="white", line_width=0.5, opacity=0.3)
    if shapes:
        fig.update_layout(shapes=shapes)
    fig.update_layout(
        title=title, yaxis_title=ylabel,
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark", showlegend=False, hovermode="x unified",
    )
    return fig

def detect_enso_events(oni_df, threshold=0.5, min_months=5):
    """Detect El Nino and La Nina events from ONI time series."""
    df = oni_df.sort_values("date").dropna(subset=["oni"]).reset_index(drop=True)
    df["phase"] = df["oni"].apply(
        lambda x: 1 if x >= threshold else (-1 if x <= -threshold else 0)
    )
    events = []
    i = 0
    while i < len(df):
        if df.loc[i, "phase"] != 0:
            phase = df.loc[i, "phase"]
            start = i
            while i < len(df) and df.loc[i, "phase"] == phase:
                i += 1
            end = i - 1
            if end - start + 1 >= min_months:
                event_rows = df.loc[start:end]
                peak_idx = event_rows["oni"].abs().idxmax()
                events.append({
                    "type": "El Nino" if phase == 1 else "La Nina",
                    "start": df.loc[start, "date"],
                    "end": df.loc[end, "date"],
                    "peak_date": df.loc[peak_idx, "date"],
                    "peak_oni": round(df.loc[peak_idx, "oni"], 2),
                    "duration_months": end - start + 1,
                    "label": f"{df.loc[start,'date'].year}/{str(df.loc[end,'date'].year)[2:]}",
                })
        else:
            i += 1
    return pd.DataFrame(events)


# ── Forecast helpers ──────────────────────────────────────────────────────────

def _norm_cdf(x):
    """Standard normal CDF via math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@st.cache_data(ttl=3600)
def compute_damped_persistence(n_leads=7, r=0.85, sigma_clim=1.0):
    """
    Damped-persistence ENSO forecast - standard benchmark model.

    Formula:
        mean(k)  = ONI_0 * r^k
        sigma(k) = sigma_clim * sqrt(1 - r^(2k))

    With r ~ 0.85 /month and sigma_clim ~ 1.0 C this reproduces
    the typical damped-persistence skill curve from the literature.
    """
    df = oni.sort_values("date").dropna(subset=["oni"])
    oni_0     = float(df["oni"].iloc[-1])
    last_date = df["date"].iloc[-1]

    rows = []
    for k in range(1, n_leads + 1):
        mean_k  = oni_0 * (r ** k)
        sigma_k = sigma_clim * math.sqrt(1.0 - r ** (2.0 * k))

        p_nino    = (1.0 - _norm_cdf((0.5  - mean_k) / sigma_k)) * 100.0
        p_nina    = _norm_cdf((-0.5 - mean_k) / sigma_k) * 100.0
        p_neutral = 100.0 - p_nino - p_nina

        fc_date = last_date + pd.DateOffset(months=k)
        rows.append({
            "date":      fc_date,
            "lead":      k,
            "mean":      round(mean_k,            3),
            "low90":     round(mean_k - 1.645 * sigma_k, 3),
            "high90":    round(mean_k + 1.645 * sigma_k, 3),
            "low50":     round(mean_k - 0.675 * sigma_k, 3),
            "high50":    round(mean_k + 0.675 * sigma_k, 3),
            "p_nino":    round(p_nino,    1),
            "p_nina":    round(p_nina,    1),
            "p_neutral": round(p_neutral, 1),
        })
    return pd.DataFrame(rows)



def make_plume_chart(fc, seas5=None):
    """Plotly figure with damped-persistence forecast plume + optional SEAS5 mean."""
    fig = go.Figure()

    # 90% envelope
    fig.add_trace(go.Scatter(
        x=list(fc["date"]) + list(fc["date"][::-1]),
        y=list(fc["high90"]) + list(fc["low90"][::-1]),
        fill="toself",
        fillcolor="rgba(239,85,59,0.12)",
        line=dict(width=0),
        name="90% range",
        hoverinfo="skip",
    ))
    # 50% envelope
    fig.add_trace(go.Scatter(
        x=list(fc["date"]) + list(fc["date"][::-1]),
        y=list(fc["high50"]) + list(fc["low50"][::-1]),
        fill="toself",
        fillcolor="rgba(239,85,59,0.25)",
        line=dict(width=0),
        name="50% range",
        hoverinfo="skip",
    ))
    # Mean forecast
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["mean"],
        mode="lines+markers",
        name="Forecast (damped persistence)",
        line=dict(color="rgb(239,85,59)", width=2.5),
        marker=dict(size=6),
        hovertemplate="%{x|%b %Y}: %{y:+.2f} C<extra></extra>",
    ))

    # Recent ONI history (last 18 months) as context
    hist = oni.sort_values("date").tail(18)
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["oni"],
        mode="lines",
        name="ONI observed",
        line=dict(color="white", width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}: %{y:+.2f} C<extra></extra>",
    ))

    # SEAS5 ensemble mean (optional)
    if seas5 is not None and len(seas5) > 0:
        fig.add_trace(go.Scatter(
            x=seas5["forecast_date"], y=seas5["anom_seas5"],
            mode="lines+markers",
            name="ECMWF SEAS5 mean",
            line=dict(color="rgb(0,204,150)", width=2.5, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="%{x|%b %Y}: %{y:+.2f} C<extra></extra>",
        ))

    fig.add_hline(y= 0.5, line_color="rgba(220,50,50,0.5)",  line_dash="dot", line_width=1)
    fig.add_hline(y=-0.5, line_color="rgba(50,100,220,0.5)", line_dash="dot", line_width=1)
    fig.add_hline(y= 0,   line_color="white", line_width=0.5, opacity=0.2)

    fig.update_layout(
        title="Nino 3.4 anomaly forecast",
        yaxis_title="Anomaly (C)",
        height=380,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def make_prob_chart(enso_probs=None, fc=None):
    """
    Grouped bar chart matching NOAA CPC style:
    La Nina (blue) | Neutral (gray) | El Nino (red) per season.
    Uses official IRI/CPC data if available, else damped-persistence fallback.
    """
    if enso_probs is not None and len(enso_probs) >= 3:
        df     = enso_probs.copy()
        x_vals = df["season"].tolist()
        title  = "ENSO seasonal probability forecast (NOAA CPC / IRI)"
        source_note = df["source"].iloc[0] if "source" in df.columns else "IRI/CPC"
        issued      = df["issued"].iloc[0]  if "issued"  in df.columns else ""
        subtitle = f"Source: {source_note}  |  Issued: {issued}"
    else:
        df     = fc.copy() if fc is not None else pd.DataFrame()
        df["season"] = df["date"].dt.strftime("%b %Y") if "date" in df.columns else []
        x_vals = df["season"].tolist()
        title  = "ENSO probability forecast (damped persistence)"
        subtitle = "Based on current ONI and exponential decay model"

    fig = go.Figure()

    # La Nina — blue
    fig.add_trace(go.Bar(
        name="La Nina",
        x=x_vals,
        y=df["p_nina"],
        marker_color="rgba(50,100,220,0.85)",
        hovertemplate="<b>%{x}</b><br>La Nina: <b>%{y:.0f}%</b><extra></extra>",
    ))
    # Neutral — gray
    fig.add_trace(go.Bar(
        name="Neutral",
        x=x_vals,
        y=df["p_neutral"],
        marker_color="rgba(160,160,160,0.70)",
        hovertemplate="<b>%{x}</b><br>Neutral: <b>%{y:.0f}%</b><extra></extra>",
    ))
    # El Nino — red
    fig.add_trace(go.Bar(
        name="El Nino",
        x=x_vals,
        y=df["p_nino"],
        marker_color="rgba(220,50,50,0.85)",
        hovertemplate="<b>%{x}</b><br>El Nino: <b>%{y:.0f}%</b><extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        bargap=0.20,
        bargroupgap=0.05,
        yaxis_title="Percent chance (%)",
        yaxis=dict(range=[0, 100], dtick=10, gridcolor="rgba(255,255,255,0.08)"),
        xaxis_title="Season",
        height=380,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.4)",
            borderwidth=0,
        ),
        hovermode="x unified",
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("Climate Indices - South America")

tab1, tab2, tab3, tab4 = st.tabs([
    "Current state",
    "Forecasts",
    "ENSO history",
    "About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 - CURRENT STATE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "Large-scale climate oscillations shape rainfall, temperature, and extreme "
        "events across Latin America. This dashboard tracks the main indices, updated "
        "daily from public sources (NOAA CPC / PSL)."
    )

    st.subheader("Current state")

    oni_val    = latest(oni,    "oni")
    nino34_val = latest(nino34, "nino34")
    nino12_val = latest(nino12, "nino12")
    mei_val    = latest(mei,    "mei")
    sam_val    = latest(sam,    "sam")
    iod_val    = latest(iod,    "iod")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ONI",           f"{oni_val:+.2f} C",    enso_label(oni_val))
    c2.metric("Nino 3.4",      f"{nino34_val:+.2f} C", enso_label(nino34_val))
    c3.metric("Nino 1+2 (Coastal)", f"{nino12_val:+.2f} C", enso_label(nino12_val))
    c4.metric("MEI",           f"{mei_val:+.2f}")
    c5.metric("SAM",           f"{sam_val:+.2f}")
    c6.metric("IOD",           f"{iod_val:+.2f} C")

    st.divider()
    st.subheader("Historical time series")

    year_min = 1979
    year_max = int(oni["date"].dt.year.max())
    y_start, y_end = st.slider(
        "Year range", min_value=year_min, max_value=year_max,
        value=(1990, year_max), step=1,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_ts(oni, "date", "oni",
            "ONI - Oceanic Nino Index", "Anomaly (C)",
            "rgb(239,85,59)", threshold_shapes(), y_start, y_end,
        ), use_container_width=True)
    with col2:
        st.plotly_chart(make_ts(nino34, "date", "nino34",
            "Nino 3.4 SST Anomaly", "Anomaly (C)",
            "rgb(99,110,250)", threshold_shapes(), y_start, y_end,
        ), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(make_ts(nino12, "date", "nino12",
            "Nino 1+2 - Coastal Nino (Peru/Ecuador)", "Anomaly (C)",
            "rgb(255,161,90)", threshold_shapes(-4, 4), y_start, y_end,
        ), use_container_width=True)
    with col4:
        st.plotly_chart(make_ts(mei, "date", "mei",
            "MEI - Multivariate ENSO Index", "MEI",
            "rgb(239,85,59)", threshold_shapes(-4, 4), y_start, y_end,
        ), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(make_ts(sam, "date", "sam",
            "SAM - Southern Annular Mode", "SAM index",
            "rgb(0,204,150)", y_start=y_start, y_end=y_end,
        ), use_container_width=True)
    with col6:
        st.plotly_chart(make_ts(iod, "date", "iod",
            "IOD - Indian Ocean Dipole", "Anomaly (C)",
            "rgb(171,99,250)", y_start=y_start, y_end=y_end,
        ), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 - FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("ENSO forecasts")
    st.markdown(
        "Interactive forecast plume and ENSO probability charts, plus official "
        "graphics from IRI and NOAA CPC. The plume uses a damped-persistence "
        "model computed from the observed ONI record - the standard benchmark "
        "in ENSO forecasting."
    )

    # ── Interactive forecast plume ────────────────────────────────────────────
    st.markdown("### Forecast plume")

    fc    = compute_damped_persistence(n_leads=7)
    seas5 = load_seas5_mean()
    st.plotly_chart(make_plume_chart(fc, seas5), use_container_width=True)

    with st.expander("About the forecast method"):
        st.markdown(
            "**Damped persistence** uses the current ONI value as a starting point "
            "and applies an exponential decay towards the climatological mean. "
            "The decay rate r = 0.85/month and a background spread of "
            "sigma = 1.0 C are consistent with published ENSO prediction skill. "
            "The shaded envelopes show 50% and 90% uncertainty ranges assuming "
            "a Gaussian error distribution."
        )

    st.divider()

    # ── ENSO probability chart ────────────────────────────────────────────────
    st.markdown("### ENSO state probabilities")

    enso_probs = load_enso_probs()
    if enso_probs is not None:
        issued = enso_probs["issued"].iloc[0] if "issued" in enso_probs.columns else ""
        source = enso_probs["source"].iloc[0] if "source" in enso_probs.columns else "IRI/CPC"
        st.caption(
            f"Source: {source} (International Research Institute for Climate and Society / NOAA CPC)  "
            f"|  Issued: {issued}  |  Updated daily."
        )
    else:
        st.caption(
            "IRI/CPC official data not yet available. "
            "Showing probabilities from the damped-persistence model."
        )

    st.plotly_chart(make_prob_chart(enso_probs, fc), use_container_width=True)

    # Probability table
    show_table = st.toggle("Show probability table", value=False)
    if show_table:
        if enso_probs is not None:
            disp = enso_probs[["season", "p_nina", "p_neutral", "p_nino"]].copy()
            disp.columns = ["Season", "La Nina (%)", "Neutral (%)", "El Nino (%)"]
            st.dataframe(disp, hide_index=True, use_container_width=True)
        else:
            disp = fc[["date", "mean", "p_nino", "p_neutral", "p_nina"]].copy()
            disp["date"] = disp["date"].dt.strftime("%b %Y")
            disp.columns = ["Month", "Mean anomaly (C)", "P(El Nino) %", "P(Neutral) %", "P(La Nina) %"]
            st.dataframe(disp, hide_index=True, use_container_width=True)

    st.divider()

    # ── IRI figures ───────────────────────────────────────────────────────────
    st.markdown("### IRI - ENSO forecast graphics")
    iri_figs = get_iri_figures()

    iri_labels = {
        "figure1": "CPC ENSO forecast",
        "figure3": "IRI ENSO forecast",
        "figure5": "Model-based prediction percentiles",
        "figure6": "Model-based prediction distribution",
        "figure7": "IOD model-based forecast",
    }

    if iri_figs:
        # Skip figure2 (historical SST time series, already shown in Current state tab)
        filtered = [u for u in iri_figs if "figure2" not in u]
        for url in filtered:
            label = next(
                (v for k, v in iri_labels.items() if k in url),
                "IRI forecast figure",
            )
            with st.expander(label, expanded=(filtered.index(url) < 2)):
                st.image(url, width=700)
    else:
        st.warning("Could not load IRI figures. Visit the source directly.")
        st.link_button(
            "Open IRI ENSO forecast page",
            "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/",
        )

    st.divider()

    # ── CPC SST image ─────────────────────────────────────────────────────────
    st.markdown("### NOAA CPC - SST anomaly")
    cpc_sst_url = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_update/sstweek_c.gif"
    try:
        r = requests.head(cpc_sst_url, timeout=10)
        if r.status_code == 200:
            st.image(cpc_sst_url, caption="NOAA CPC - Weekly SST anomaly", width=700)
        else:
            raise ValueError()
    except Exception:
        st.warning("Could not load CPC SST image.")

    st.link_button(
        "Open full NOAA CPC ENSO status report (PDF)",
        "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/lanina/enso_evolution-status-fcsts-web.pdf",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - ENSO HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ENSO event history")

    events = detect_enso_events(oni)

    if events.empty:
        st.info("No events detected.")
    else:
        el_nino = events[events["type"] == "El Nino"].sort_values("peak_oni", ascending=False)
        la_nina = events[events["type"] == "La Nina"].sort_values("peak_oni")

        # Timeline
        st.markdown("#### Event timeline (1950 - present)")
        fig_tl = go.Figure()

        oni_plot = oni.sort_values("date")
        fig_tl.add_trace(go.Scatter(
            x=oni_plot["date"], y=oni_plot["oni"],
            mode="lines", name="ONI",
            line=dict(color="white", width=1),
        ))

        for _, ev in events.iterrows():
            color = "rgba(220,50,50,0.25)" if ev["type"] == "El Nino" else "rgba(50,100,220,0.25)"
            fig_tl.add_vrect(
                x0=ev["start"], x1=ev["end"],
                fillcolor=color, line_width=0,
                annotation_text=ev["label"],
                annotation_position="top left",
                annotation_font_size=9,
            )

        fig_tl.add_hline(y=0.5,  line_color="rgba(220,50,50,0.5)",  line_dash="dot", line_width=1)
        fig_tl.add_hline(y=-0.5, line_color="rgba(50,100,220,0.5)", line_dash="dot", line_width=1)
        fig_tl.add_hline(y=0,    line_color="white", line_width=0.5, opacity=0.2)

        fig_tl.update_layout(
            height=350, template="plotly_dark",
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False, hovermode="x unified",
            yaxis_title="ONI (C)",
        )
        st.plotly_chart(fig_tl, use_container_width=True)

        st.divider()

        # Event tables
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### El Nino events")
            st.dataframe(
                el_nino[["label", "start", "end", "peak_oni", "duration_months"]]
                .rename(columns={
                    "label": "Event", "start": "Start", "end": "End",
                    "peak_oni": "Peak ONI (C)", "duration_months": "Duration (months)",
                })
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
        with col_b:
            st.markdown("#### La Nina events")
            st.dataframe(
                la_nina[["label", "start", "end", "peak_oni", "duration_months"]]
                .rename(columns={
                    "label": "Event", "start": "Start", "end": "End",
                    "peak_oni": "Peak ONI (C)", "duration_months": "Duration (months)",
                })
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

        st.divider()

        # Event comparison
        st.markdown("#### Compare El Nino events")
        st.markdown("Select events to overlay their ONI evolution (month 0 = event start).")

        event_options = el_nino["label"].tolist()
        selected = st.multiselect(
            "Select El Nino events",
            options=event_options,
            default=event_options[:4] if len(event_options) >= 4 else event_options,
        )

        if selected:
            fig_cmp = go.Figure()
            oni_sorted = oni.sort_values("date").reset_index(drop=True)

            for label in selected:
                ev = el_nino[el_nino["label"] == label].iloc[0]
                start_date = ev["start"]
                mask = oni_sorted["date"] >= start_date
                idx0 = oni_sorted[mask].index[0]
                snippet = oni_sorted.loc[idx0: idx0 + 35].copy()
                snippet["month"] = range(len(snippet))
                fig_cmp.add_trace(go.Scatter(
                    x=snippet["month"], y=snippet["oni"],
                    mode="lines", name=label, line=dict(width=2),
                ))

            fig_cmp.add_hline(y=0.5,  line_color="rgba(220,50,50,0.4)", line_dash="dot")
            fig_cmp.add_hline(y=0,    line_color="white", line_width=0.5, opacity=0.2)
            fig_cmp.update_layout(
                height=380, template="plotly_dark",
                xaxis_title="Month from event start",
                yaxis_title="ONI (C)",
                margin=dict(l=10, r=10, t=20, b=10),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 - ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("About the indices")
    st.markdown("""
**ONI - Oceanic Nino Index**
3-month running mean of SST anomalies in the Nino 3.4 region (5N-5S, 170W-120W),
relative to the 1991-2020 climatology. El Nino is declared when ONI exceeds +0.5 C
for at least 5 consecutive overlapping seasons. Source: NOAA CPC.

**Nino 3.4**
Monthly SST anomaly averaged over the Nino 3.4 region. Similar to ONI but not smoothed.
Source: NOAA CPC ERSSTv5.

**Nino 1+2 - Coastal Nino**
SST anomaly in the region closest to the South American coast (10S-0, 90W-80W).
Most directly related to the coastal El Nino that affects Peru and Ecuador.
Source: NOAA CPC ERSSTv5.

**MEI - Multivariate ENSO Index v2**
Combines sea level pressure, SST, surface winds, and outgoing longwave radiation
over the tropical Pacific. Bimonthly. Source: NOAA PSL.

**SAM - Southern Annular Mode**
Also known as the Antarctic Oscillation (AAO). Measures the north-south shift of
the westerly wind belt in the Southern Hemisphere. Positive phase: stronger
westerlies at high latitudes, drier conditions over subtropical South America.
Source: NOAA CPC.

**IOD - Indian Ocean Dipole**
Difference in SST anomalies between the western (50E-70E, 10S-10N) and eastern
(90E-110E, 10S-0) tropical Indian Ocean. Positive events can modulate ENSO
teleconnections over South America. Source: NOAA PSL / Hadley Centre.

---
Data updated daily via GitHub Actions. Source code:
[github.com/dbrisaro/climate-dashboard](https://github.com/dbrisaro/climate-dashboard)
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data sources: NOAA CPC (ONI, Nino indices, SAM) - NOAA PSL (MEI, IOD) - "
    "IRI Columbia (forecasts) - Updated daily via GitHub Actions"
)
