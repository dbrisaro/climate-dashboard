import math
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="Climate Oscillation Monitor",
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
    soi    = pd.read_csv(f"{BASE_URL}/soi.csv",    parse_dates=["date"])
    soi    = soi.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return oni, mei, sam, iod, nino34, nino12, soi

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
def load_iri_plume():
    """
    Fetch IRI multi-model ENSO plume data from the public JSON API.
    URL: https://ensoforecast.iri.columbia.edu/plumes_json/{year}/{month_0idx}
    IRI publishes around the 19th of each month; month is 0-indexed (Jan=0).
    Returns dict with keys: models, observed, averages, seasons, init_label
    or None on failure.
    """
    now = datetime.utcnow()
    # IRI publishes ~19th of the month; if before the 20th use previous month
    if now.day < 20:
        init = (now.replace(day=1) - pd.DateOffset(months=1))
    else:
        init = now.replace(day=1)

    year     = init.year
    month_0  = init.month - 1   # 0-indexed: Jan=0, Feb=1, ..., Dec=11

    url = f"https://ensoforecast.iri.columbia.edu/plumes_json/{year}/{month_0}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    # Build season labels starting from init month+1
    season_names = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
    seasons = []
    for k in range(9):
        idx = (init.month + k) % 12   # center month of 3-month season
        seasons.append(season_names[idx])

    data["seasons"]    = seasons
    data["init_label"] = init.strftime("%B %Y")
    return data


def make_iri_plume_chart(data, oni_df):
    """
    Reproduces IRI 'Model Predictions of ENSO' chart:
    - Each model gets its own color + marker symbol
    - Observed values (DJF-OBS, Feb-OBS) at left; models fan out from last obs
    - DYN AVG (dark red) and STAT AVG (dark green) as thick lines
    """
    seasons  = data.get("seasons", [])
    observed = data.get("observed", [])
    avgs     = data.get("averages", {})

    # x-axis: observed labels + forecast seasons
    obs_labels = [f"{o['month']}-OBS" for o in observed]
    obs_values = [o["data"]           for o in observed]
    last_obs   = obs_values[-1] if obs_values else 0.0

    fig = go.Figure()

    # ── Color + marker palette (one per model) ────────────────────────────────
    MODEL_COLORS = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#bcbd22","#17becf","#aec7e8",
        "#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94",
        "#f7b6d2","#dbdb8d","#9edae5","#393b79","#637939",
        "#8c6d31","#843c39","#7b4173","#3182bd","#e6550d",
        "#31a354","#756bb1",
    ]
    MODEL_MARKERS = [
        "circle","square","diamond","triangle-up","triangle-down",
        "pentagon","hexagon","star","cross","x",
        "circle-open","square-open","diamond-open",
        "triangle-up-open","triangle-down-open",
        "triangle-left","triangle-right","bowtie","hourglass",
        "circle-dot","square-dot","diamond-dot",
    ]

    # ── Individual model lines (fan from last observed point) ─────────────────
    for i, m in enumerate(data.get("models", [])):
        raw   = m.get("data", [])
        fcast = [v if v not in (-999, -999.0) else None for v in raw]
        if not any(v is not None for v in fcast):
            continue
        x_m = [obs_labels[-1]] + seasons[:len(fcast)]
        y_m = [last_obs]       + fcast
        color  = MODEL_COLORS[i % len(MODEL_COLORS)]
        marker = MODEL_MARKERS[i % len(MODEL_MARKERS)]
        fig.add_trace(go.Scatter(
            x=x_m, y=y_m,
            mode="lines+markers",
            name=m["model"],
            line=dict(color=color, width=1.4),
            marker=dict(size=8, symbol=marker, color=color),
            hovertemplate=f"<b>{m['model']}</b>: %{{y:.2f}} C<extra></extra>",
        ))

    # ── Observed line ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=obs_labels, y=obs_values,
        mode="lines+markers",
        name="Observed",
        line=dict(color="white", width=3),
        marker=dict(size=9, color="white", symbol="circle"),
        hovertemplate="<b>Observed %{x}</b>: %{y:+.2f} C<extra></extra>",
    ))

    # ── DYN AVG (dark red) and STAT AVG (dark green) ──────────────────────────
    for key, label, color in [
        ("dynamical",   "DYN AVG",  "#cc2200"),
        ("statistical", "STAT AVG", "#007700"),
    ]:
        raw   = avgs.get(key, [])
        fcast = [v if v not in (-999, -999.0, None) else None for v in raw]
        if not any(v is not None for v in fcast):
            continue
        x_a = [obs_labels[-1]] + seasons[:len(fcast)]
        y_a = [last_obs]       + fcast
        fig.add_trace(go.Scatter(
            x=x_a, y=y_a,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=3.5),
            marker=dict(size=8, color=color),
            hovertemplate=f"<b>{label}</b>: %{{y:.2f}} C<extra></extra>",
        ))

    # ── Threshold lines ───────────────────────────────────────────────────────
    fig.add_hline(y= 0.5, line_color="rgba(255,255,255,0.4)", line_dash="dot", line_width=1.2)
    fig.add_hline(y=-0.5, line_color="rgba(255,255,255,0.4)", line_dash="dot", line_width=1.2)
    fig.add_hline(y= 0,   line_color="rgba(255,255,255,0.2)", line_width=1)

    fig.update_layout(
        height=520,
        template="plotly_dark",
        yaxis_title="Nino 3.4 SST Anomaly (C)",
        xaxis=dict(tickangle=-30),
        margin=dict(l=10, r=160, t=10, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",  y=1.0,
            xanchor="left", x=1.01,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=10),
            tracegroupgap=2,
        ),
    )
    return fig


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

oni, mei, sam, iod, nino34, nino12, soi = load_all()

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

def compute_enso_climatology(oni_df, threshold=0.5):
    """
    Compute historical ENSO state climatology from the ONI record.
    Returns a DataFrame with columns: season, clim_nino, clim_neutral, clim_nina
    for each 3-month season (DJF through NDJ).
    """
    season_order = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
    df = oni_df.dropna(subset=["oni", "season"]).copy()
    rows = []
    for s in season_order:
        sub = df[df["season"] == s]["oni"]
        if len(sub) < 5:
            continue
        n       = len(sub)
        p_nino    = (sub >=  threshold).sum() / n * 100
        p_nina    = (sub <= -threshold).sum() / n * 100
        p_neutral = 100 - p_nino - p_nina
        rows.append({"season": s, "clim_nino": round(p_nino,1),
                     "clim_neutral": round(p_neutral,1), "clim_nina": round(p_nina,1)})
    return pd.DataFrame(rows)


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


def make_prob_chart(enso_probs=None, fc=None, clim=None):
    """
    Grouped bar chart matching IRI/CPC style:
    La Nina (blue) | Neutral (gray) | El Nino (red) per season,
    with optional climatology lines overlaid.
    """
    if enso_probs is not None and len(enso_probs) >= 3:
        df     = enso_probs.copy()
        x_vals = df["season"].tolist()
    else:
        df     = fc.copy() if fc is not None else pd.DataFrame()
        df["season"] = df["date"].dt.strftime("%b %Y") if "date" in df.columns else []
        x_vals = df["season"].tolist()

    fig = go.Figure()

    # ── Bars ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        name="La Nina", x=x_vals, y=df["p_nina"],
        marker_color="rgba(50,100,220,0.85)",
        hovertemplate="<b>%{x}</b><br>La Nina: <b>%{y:.0f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Neutral", x=x_vals, y=df["p_neutral"],
        marker_color="rgba(160,160,160,0.70)",
        hovertemplate="<b>%{x}</b><br>Neutral: <b>%{y:.0f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="El Nino", x=x_vals, y=df["p_nino"],
        marker_color="rgba(220,50,50,0.85)",
        hovertemplate="<b>%{x}</b><br>El Nino: <b>%{y:.0f}%</b><extra></extra>",
    ))

    # ── Climatology lines ─────────────────────────────────────────────────────
    if clim is not None and len(clim) > 0:
        # Align climatology to the same seasons as the forecast
        clim_aligned = clim[clim["season"].isin(x_vals)].set_index("season").reindex(x_vals)
        clim_specs = [
            ("clim_nina",    "La Nina clim.",  "rgba(80,130,255,0.9)"),
            ("clim_neutral", "Neutral clim.",  "rgba(200,200,200,0.9)"),
            ("clim_nino",    "El Nino clim.",  "rgba(255,80,80,0.9)"),
        ]
        for col, label, color in clim_specs:
            if col not in clim_aligned.columns:
                continue
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=clim_aligned[col].tolist(),
                mode="lines",
                name=label,
                line=dict(color=color, width=2, dash="solid"),
                hovertemplate=f"<b>{label}</b> %{{x}}: %{{y:.0f}}%<extra></extra>",
            ))

    fig.update_layout(
        barmode="group",
        bargap=0.20,
        bargroupgap=0.05,
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100], dtick=10, gridcolor="rgba(255,255,255,0.08)"),
        xaxis_title="Season",
        height=500,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="v",
            yanchor="top",  y=1.0,
            xanchor="left", x=1.01,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )
    return fig


# ── Discrete IRI-style colorscale ────────────────────────────────────────────
# 18 bins × 0.10 step from -0.90 to +0.90.
# NaN areas show as white (plot_bgcolor) — not part of the colorscale.
# Brown shades = below-normal, white = NaN gap, green shades = above-normal.
_BIN_COLORS = [
    "#6B2700",  # -0.90…-0.80  ≥ 80 % below-normal
    "#8B3A0F",  # -0.80…-0.70  ≥ 70 %
    "#CC7722",  # -0.70…-0.60  ≥ 60 %
    "#E8A850",  # -0.60…-0.50  ≥ 50 %
    "#F5CC90",  # -0.50…-0.40  ≥ 40 %
    "#FFFFFF",  # NaN gap
    "#FFFFFF",
    "#FFFFFF",
    "#FFFFFF",
    "#FFFFFF",
    "#FFFFFF",
    "#FFFFFF",
    "#FFFFFF",  # NaN gap
    "#C8EAB0",  # +0.40…+0.50  ≥ 40 % above-normal
    "#7BC87A",  # +0.50…+0.60  ≥ 50 %
    "#3D9E3D",  # +0.60…+0.70  ≥ 60 %
    "#1E6B1E",  # +0.70…+0.80  ≥ 70 %
    "#0A3D0A",  # +0.80…+0.90  ≥ 80 %
]
_N = len(_BIN_COLORS)  # 18
_TERCILE_COLORSCALE = []
for _i, _c in enumerate(_BIN_COLORS):
    _TERCILE_COLORSCALE.append([_i / _N, _c])
    _TERCILE_COLORSCALE.append([(_i + 1) / _N, _c])


def make_nmme_prob_map(df):
    """
    IRI-style tercile probability map.

    Signal:
      +prob_above  where prob_above >= 40% and is the dominant category (green)
      -prob_below  where prob_below >= 40% and is the dominant category (brown)
      NaN          everywhere else  -> shows as white (plot_bgcolor)
    """
    df = df.copy()

    # Dominant-probability signal (only where signal is meaningful)
    above_wins = (df["prob_above"] >= 0.40) & (df["prob_above"] > df["prob_below"])
    below_wins = (df["prob_below"] >= 0.40) & (df["prob_below"] > df["prob_above"])

    df["signal"] = np.nan
    df.loc[above_wins, "signal"] =  df.loc[above_wins, "prob_above"]
    df.loc[below_wins, "signal"] = -df.loc[below_wins, "prob_below"]

    pivot = df.pivot_table(index="lat", columns="lon", values="signal")
    lats  = pivot.index.values
    lons  = pivot.columns.values
    z     = pivot.values

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=lons, y=lats, z=z,
        colorscale=_TERCILE_COLORSCALE,
        zmin=-0.90, zmax=0.90,
        contours=dict(start=-0.90, end=0.90, size=0.10),
        contours_coloring="fill",   # discrete steps
        line=dict(width=0.4, color="rgba(180,180,180,0.3)"),
        colorbar=dict(
            tickvals=[-0.85, -0.75, -0.65, -0.55, -0.45,
                       0.45,  0.55,  0.65,  0.75,  0.85],
            ticktext=["≥80% below", "≥70%", "≥60%", "≥50%", "≥40%",
                      "≥40% above", "≥50%", "≥60%", "≥70%", "≥80%"],
            thickness=16, len=0.70, outlinewidth=0,
            title=dict(text="Probability", side="right"),
            tickfont=dict(size=10),
        ),
        hovertemplate=(
            "Lon: %{x:.1f}°  Lat: %{y:.1f}°<br>"
            "Prob: <b>%{z:.0%}</b>  "
            "(+ above-normal · − below-normal)<extra></extra>"
        ),
        connectgaps=False,
    ))

    border_lons, border_lats = load_sa_borders()
    if border_lons:
        fig.add_trace(go.Scatter(
            x=border_lons, y=border_lats,
            mode="lines",
            line=dict(color="rgba(60,60,60,0.85)", width=0.9),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-92, -28], title="", showgrid=False,
                   zeroline=False, constrain="domain",
                   tickfont=dict(color="#555"), tickcolor="#aaa"),
        yaxis=dict(range=[-58, 16], title="", showgrid=False,
                   zeroline=False, scaleanchor="x", scaleratio=1,
                   tickfont=dict(color="#555"), tickcolor="#aaa"),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_seas5_sa_maps():
    """
    Load SEAS5 T2m and precip anomaly grids for South America from GitHub data.
    Returns dict with 't2m' and/or 'prcp' DataFrames, or None.
    """
    result = {}
    for key, fname in [("t2m", "seas5_t2m_anom_SA"), ("prcp", "seas5_prcp_mmday_SA")]:
        try:
            df = pd.read_csv(f"{BASE_URL}/forecasts/{fname}.csv")
            if len(df) < 10 or "anom" not in df.columns:
                continue
            # Reconstruct forecast_date if missing (old CSV format)
            if "forecast_date" not in df.columns:
                if "lead_month" not in df.columns:
                    df["lead_month"] = 1
                if "init_year" in df.columns and "init_month" in df.columns:
                    df["forecast_date"] = df.apply(
                        lambda r: (pd.Timestamp(year=int(r["init_year"]),
                                                month=int(r["init_month"]), day=1)
                                   + pd.DateOffset(months=int(r["lead_month"]))
                                   ).strftime("%Y-%m"),
                        axis=1,
                    )
                else:
                    df["forecast_date"] = "unknown"
            result[key] = df
        except Exception:
            pass
    return result if result else None


@st.cache_data(ttl=3600)
def load_nmme_probs():
    """Load CPC NMME tercile probability forecasts for SA."""
    result = {}
    for key, fname in [("tmp2m", "nmme_tmp2m_probs_SA"),
                       ("prate", "nmme_prate_probs_SA")]:
        try:
            df = pd.read_csv(f"{BASE_URL}/forecasts/{fname}.csv")
            if len(df) > 100 and "prob_above" in df.columns:
                result[key] = df
        except Exception:
            pass
    return result if result else None


@st.cache_data(ttl=86400)
def load_sa_borders():
    """
    Download Natural Earth 110m country polygons and return
    lists of (lons, lats) for polygons that overlap South America.
    Result is a single pair of flat lists with None separators —
    ready for a single go.Scatter trace.
    """
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
           "/master/geojson/ne_110m_admin_0_countries.geojson")
    try:
        r = requests.get(url, timeout=15)
        features = r.json()["features"]
    except Exception:
        return [], []

    lons_out, lats_out = [], []
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        polys = (geom["coordinates"] if geom["type"] == "Polygon"
                 else geom["coordinates"] if geom["type"] != "MultiPolygon"
                 else geom["coordinates"])
        if geom["type"] == "Polygon":
            polys = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        for poly in polys:
            ring = poly[0]
            lo = [c[0] for c in ring]
            la = [c[1] for c in ring]
            # Keep only rings that overlap the SA bounding box
            if any(-92 <= x <= -28 and -58 <= y <= 16 for x, y in zip(lo, la)):
                lons_out += lo + [None]
                lats_out += la + [None]
    return lons_out, lats_out


def make_seas5_geo_map(df, colorscale, cbar_title, step, vrange=None, diverging=True):
    """
    Discrete filled-contour map of a SEAS5 field over South America.
    Uses go.Contour with explicit step size for a discrete colorbar.
    diverging=True  -> symmetric +-vrange (anomaly maps)
    diverging=False -> 0..vmax range (absolute maps like precipitation)
    """
    vals = df["anom"]
    if diverging:
        if vrange is None:
            vrange = max(abs(float(vals.quantile(0.02))),
                         abs(float(vals.quantile(0.98))), step)
        vrange = round(math.ceil(float(vrange) / step) * step, 6)
        zmin, zmax = -vrange, vrange
    else:
        zmin = 0.0
        zmax = round(math.ceil(float(vals.quantile(0.98)) / step) * step, 6)
        if zmax == 0:
            zmax = step

    pivot = df.pivot_table(index="lat", columns="lon", values="anom")
    lats  = pivot.index.values
    lons  = pivot.columns.values
    z     = pivot.values

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=lons, y=lats, z=z,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        contours=dict(start=zmin, end=zmax, size=step),
        contours_coloring="fill",
        line=dict(width=0.3, color="rgba(180,180,180,0.2)"),
        colorbar=dict(
            title=dict(text=cbar_title, side="right"),
            thickness=14, len=0.80, outlinewidth=0,
            tickfont=dict(size=10),
        ),
        hovertemplate="Lon: %{x:.1f}  Lat: %{y:.1f}<br>"
                      + cbar_title + ": <b>%{z:.2f}</b><extra></extra>",
    ))

    border_lons, border_lats = load_sa_borders()
    if border_lons:
        fig.add_trace(go.Scatter(
            x=border_lons, y=border_lats,
            mode="lines",
            line=dict(color="rgba(60,60,60,0.85)", width=0.9),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-92, -28], title="", showgrid=False,
                   zeroline=False, constrain="domain",
                   tickfont=dict(color="#555"), tickcolor="#aaa"),
        yaxis=dict(range=[-58, 16], title="", showgrid=False,
                   zeroline=False, scaleanchor="x", scaleratio=1,
                   tickfont=dict(color="#555"), tickcolor="#aaa"),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


st.title("Climate Oscillation Monitor")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Current state",
    "ENSO forecasts",
    "Seasonal forecasts",
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
    soi_val    = latest(soi,    "soi")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("ONI",                 f"{oni_val:+.2f} C",    enso_label(oni_val))
    c2.metric("Nino 3.4",            f"{nino34_val:+.2f} C", enso_label(nino34_val))
    c3.metric("Nino 1+2 (Coastal)",  f"{nino12_val:+.2f} C", enso_label(nino12_val))
    c4.metric("MEI",                 f"{mei_val:+.2f}")
    c5.metric("SOI",                 f"{soi_val:+.2f}")
    c6.metric("SAM",                 f"{sam_val:+.2f}")
    c7.metric("IOD",                 f"{iod_val:+.2f} C")

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
        st.plotly_chart(make_ts(soi, "date", "soi",
            "SOI - Southern Oscillation Index", "SOI",
            "rgb(0,180,255)", y_start=y_start, y_end=y_end,
        ), use_container_width=True)
    with col6:
        st.plotly_chart(make_ts(sam, "date", "sam",
            "SAM - Southern Annular Mode", "SAM index",
            "rgb(0,204,150)", y_start=y_start, y_end=y_end,
        ), use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
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

    # ── IRI multi-model plume ─────────────────────────────────────────────────
    st.markdown("### Multi-model forecast plume")

    iri_plume    = load_iri_plume()
    fc           = compute_damped_persistence(n_leads=7)
    seas5_nino34 = load_seas5_mean()

    if iri_plume is not None:
        n_models = len(iri_plume.get("models", []))
        st.caption(
            f"Source: IRI multi-model ensemble  |  Init: {iri_plume['init_label']}  "
            f"|  {n_models} models (dynamical + statistical)  |  "
            "API: ensoforecast.iri.columbia.edu"
        )
        st.plotly_chart(make_iri_plume_chart(iri_plume, oni), use_container_width=True)
    else:
        st.caption(
            "IRI API not available. Showing damped-persistence benchmark forecast."
        )
        st.plotly_chart(make_plume_chart(fc, seas5_nino34), use_container_width=True)

    with st.expander("About the forecast models"):
        st.markdown(
            "**Dynamical models** (orange lines) use coupled ocean-atmosphere GCMs "
            "to simulate the evolution of tropical Pacific SSTs. "
            "Models include ECMWF SEAS5, NCEP CFSv2, NASA GMAO, UKMO, and others.\n\n"
            "**Statistical models** (blue lines) use empirical relationships "
            "derived from historical ENSO data.\n\n"
            "The **all-models average** (white line) is the consensus forecast. "
            "Spread between models gives a measure of forecast uncertainty.\n\n"
            "Source: IRI/CPC - [ensoforecast.iri.columbia.edu](https://ensoforecast.iri.columbia.edu)"
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

    clim = compute_enso_climatology(oni)
    col_prob, _ = st.columns([2, 1])
    with col_prob:
        st.plotly_chart(make_prob_chart(enso_probs, fc, clim), use_container_width=True)

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

    st.link_button(
        "Open full NOAA CPC ENSO status report (PDF)",
        "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/lanina/enso_evolution-status-fcsts-web.pdf",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 - T & P FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Seasonal temperature and precipitation forecasts")
    st.markdown(
        "This tab shows two complementary sources of seasonal forecasts over South America: "
        "**CPC NMME** tercile probability maps and **ECMWF SEAS5** ensemble-mean anomaly maps."
    )

    st.markdown("### CPC NMME - Tercile probability maps")
    st.markdown(
        "Each map shows which tercile category is most likely for a **3-month overlapping season** "
        "(e.g. MJJ = May+Jun+Jul average), initialized from the same month. "
        "Colors show the dominant category: "
        "**brown = below-normal · white = equal chances (no category >= 40%) · green = above-normal.**"
    )

    with st.expander("What is lead time?"):
        st.markdown("""
**Lead time** is how far in advance a forecast is issued relative to the period it is predicting.

**Example** - if the forecast is initialized in **April 2026**:

| Season | Months covered | Lead time | Interpretation |
|--------|---------------|-----------|----------------|
| MJJ | May + Jun + Jul | 1-3 months | Short lead - most reliable |
| JJA | Jun + Jul + Aug | 2-4 months | Medium lead |
| JAS | Jul + Aug + Sep | 3-5 months | Medium-long lead |
| ASO | Aug + Sep + Oct | 4-6 months | Long lead - less reliable |

**Why does lead time matter?**
The atmosphere loses memory of its initial state over time, so forecast skill generally decreases with longer lead times.
At short leads (1-2 months), the model still "remembers" current ocean and land conditions.
At long leads (4-6 months), the signal comes almost entirely from slowly-evolving boundary forcings
like sea surface temperatures (ENSO, for example) - everything else tends to average out.

On the maps: **longer lead = more white area** (fewer grid points where any category exceeds the 40% threshold),
because the ensemble models spread out and agree less on which tercile will be most likely.
        """)


    nmme  = load_nmme_probs()
    seas5 = load_seas5_sa_maps()

    if not nmme and not seas5:
        st.info(
            "Forecast maps will be available after the next data update "
            "(runs daily via GitHub Actions)."
        )
    else:
        # ── Shared season label helper ────────────────────────────────────────
        def _seas_label(date_str):
            t = pd.Timestamp(str(date_str) + "-01")
            months = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
            m0 = t.month - 1
            return months[m0] + months[(m0+1)%12] + months[(m0+2)%12] + f" {t.year}"

        # ── Build union of available forecast dates ───────────────────────────
        all_dates = set()
        if nmme:
            ref_nmme = nmme.get("tmp2m", nmme.get("prate"))
            all_dates.update(ref_nmme["forecast_date"].unique())
        if seas5:
            ref_s5 = seas5.get("t2m", seas5.get("prcp"))
            all_dates.update(ref_s5["forecast_date"].unique())
        fc_dates = sorted(all_dates)

        # ── Init month labels ─────────────────────────────────────────────────
        captions = []
        if nmme:
            r = ref_nmme.iloc[0]
            captions.append(
                f"NMME init: {pd.Timestamp(year=int(r['init_year']), month=int(r['init_month']), day=1).strftime('%B %Y')}"
            )
        if seas5:
            r = ref_s5.iloc[0]
            if "init_year" in r and "init_month" in r:
                captions.append(
                    f"SEAS5 init: {pd.Timestamp(year=int(r['init_year']), month=int(r['init_month']), day=1).strftime('%B %Y')}"
                )

        # ── Single selector ───────────────────────────────────────────────────
        sel_date = st.radio(
            "Forecast season",
            options=fc_dates,
            format_func=_seas_label,
            horizontal=True,
        )
        st.caption("  |  ".join(captions))
        seas_str = _seas_label(sel_date)

        # ── Row 1: NMME probability maps ──────────────────────────────────────
        if nmme:
            st.markdown("#### CPC NMME - Tercile probabilities")
            st.info(
                "**Why is temperature almost entirely green (above-normal)?** "
                "The NMME terciles are relative to the 1991-2020 climatology. "
                "In 2026 the global warming trend pushes virtually all model forecasts "
                "above that baseline - this is a real physical signal, not a bug. "
                "The SEAS5 anomaly maps below show the magnitude in degrees C."
            )
            col_t, col_p = st.columns(2)
            with col_t:
                if "tmp2m" in nmme:
                    sub = nmme["tmp2m"][nmme["tmp2m"]["forecast_date"] == sel_date]
                    if len(sub) > 0:
                        st.plotly_chart(make_nmme_prob_map(sub), use_container_width=True)
                        st.caption(f"NMME temperature tercile probabilities - {seas_str}")
                    else:
                        st.info("Not available for this season.")
            with col_p:
                if "prate" in nmme:
                    sub = nmme["prate"][nmme["prate"]["forecast_date"] == sel_date]
                    if len(sub) > 0:
                        st.plotly_chart(make_nmme_prob_map(sub), use_container_width=True)
                        st.caption(f"NMME precipitation tercile probabilities - {seas_str}")
                    else:
                        st.info("Not available for this season.")

        # ── Row 2: SEAS5 anomaly maps ─────────────────────────────────────────
        if seas5:
            st.markdown("#### ECMWF SEAS5 - Ensemble-mean anomaly")
            col_st, col_sp = st.columns(2)
            with col_st:
                if "t2m" in seas5:
                    sub = seas5["t2m"][seas5["t2m"]["forecast_date"] == sel_date]
                    if len(sub) > 0:
                        st.plotly_chart(
                            make_seas5_geo_map(sub, colorscale="RdBu_r",
                                              cbar_title="T2m anomaly (C)",
                                              step=0.5, diverging=True),
                            use_container_width=True,
                        )
                        st.caption(f"SEAS5 temperature anomaly (C, vs 1993-2016) - {seas_str}")
                    else:
                        st.info("Not available for this season.")
            with col_sp:
                if "prcp" in seas5:
                    sub = seas5["prcp"][seas5["prcp"]["forecast_date"] == sel_date]
                    if len(sub) > 0:
                        st.plotly_chart(
                            make_seas5_geo_map(sub, colorscale="YlGnBu",
                                              cbar_title="Precipitation (mm/day)",
                                              step=1.0, diverging=False),
                            use_container_width=True,
                        )
                        st.caption(f"SEAS5 precipitation ensemble mean (mm/day) - {seas_str}")
                    else:
                        st.info("Not available for this season.")
            st.caption(
                "Source: ECMWF SEAS5 via Copernicus Climate Data Store (C3S)  |  "
                "T2m anomaly reference: 1993-2016  |  "
                "Precipitation: absolute ensemble mean (mm/day)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 - ENSO HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
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
# TAB 5 - ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
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

**SOI - Southern Oscillation Index**
Standardized difference in surface air pressure between Tahiti and Darwin, Australia.
Negative SOI = El Nino conditions (low pressure over Tahiti, high over Darwin).
Positive SOI = La Nina conditions. Source: NOAA CPC.

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
