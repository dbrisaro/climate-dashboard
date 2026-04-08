import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Climate Indices - South America",
    layout="wide",
)

BASE_URL = "https://raw.githubusercontent.com/dbrisaro/climate-dashboard/main/data"

@st.cache_data(ttl=3600)
def load_all():
    oni    = pd.read_csv(f"{BASE_URL}/oni.csv",    parse_dates=["date"])
    mei    = pd.read_csv(f"{BASE_URL}/mei.csv",    parse_dates=["date"])
    sam    = pd.read_csv(f"{BASE_URL}/sam.csv",    parse_dates=["date"])
    iod    = pd.read_csv(f"{BASE_URL}/iod.csv",    parse_dates=["date"])
    nino34 = pd.read_csv(f"{BASE_URL}/nino34.csv", parse_dates=["date"])
    return oni, mei, sam, iod, nino34

oni, mei, sam, iod, nino34 = load_all()

# Header
st.title("Climate Indices - South America")
st.markdown(
    """
    Large-scale climate oscillations shape rainfall, temperature, and extreme events
    across Latin America. This dashboard tracks the main indices, updated daily from
    public sources (NOAA CPC / PSL).
    """
)

# Current state
st.subheader("Current state")

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

oni_val    = latest(oni,    "oni")
nino34_val = latest(nino34, "nino34")
mei_val    = latest(mei,    "mei")
sam_val    = latest(sam,    "sam")
iod_val    = latest(iod,    "iod")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ONI",      f"{oni_val:+.2f} C",    enso_label(oni_val))
c2.metric("Nino 3.4", f"{nino34_val:+.2f} C", enso_label(nino34_val))
c3.metric("MEI",      f"{mei_val:+.2f}")
c4.metric("SAM",      f"{sam_val:+.2f}")
c5.metric("IOD",      f"{iod_val:+.2f} C")

st.divider()

# Time range selector
st.subheader("Historical time series")

year_min = 1979
year_max = oni["date"].dt.year.max()
y_start, y_end = st.slider(
    "Year range",
    min_value=year_min, max_value=year_max,
    value=(1990, year_max), step=1,
)

def filt(df):
    return df[(df["date"].dt.year >= y_start) & (df["date"].dt.year <= y_end)]

def threshold_shapes(ymin=-3, ymax=3):
    """Shaded bands for El Nino / La Nina thresholds."""
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

def make_ts(df, x_col, y_col, title, ylabel, color, shapes=None):
    fig = go.Figure()
    d = filt(df)
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
        title=title,
        yaxis_title=ylabel,
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        showlegend=False,
        hovermode="x unified",
    )
    return fig

# Row 1: ONI + Nino 3.4
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(make_ts(
        oni, "date", "oni",
        "ONI - Oceanic Nino Index", "Anomaly (C)",
        "rgb(239,85,59)", threshold_shapes(),
    ), use_container_width=True)
with col2:
    st.plotly_chart(make_ts(
        nino34, "date", "nino34",
        "Nino 3.4 SST Anomaly", "Anomaly (C)",
        "rgb(99,110,250)", threshold_shapes(),
    ), use_container_width=True)

# Row 2: MEI + SAM
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(make_ts(
        mei, "date", "mei",
        "MEI - Multivariate ENSO Index", "MEI",
        "rgb(239,85,59)", threshold_shapes(-4, 4),
    ), use_container_width=True)
with col4:
    st.plotly_chart(make_ts(
        sam, "date", "sam",
        "SAM - Southern Annular Mode", "SAM index",
        "rgb(0,204,150)",
    ), use_container_width=True)

# Row 3: IOD + About
col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(make_ts(
        iod, "date", "iod",
        "IOD - Indian Ocean Dipole", "Anomaly (C)",
        "rgb(171,99,250)",
    ), use_container_width=True)
with col6:
    st.markdown("### About the indices")
    st.markdown("""
**ONI / Nino 3.4** - 3-month running mean of SST anomalies in the Nino 3.4 region
(5N-5S, 170W-120W). Threshold: +/-0.5 C for 5 consecutive seasons.
El Nino brings drought to NE Brazil and wetter conditions to SE South America.

**MEI** - Multivariate ENSO Index v2. Combines SLP, SST, surface winds, and OLR
over the tropical Pacific. Bimonthly.

**SAM** - Southern Annular Mode (AAO). Positive phase means stronger westerlies
and drier conditions over subtropical South America.

**IOD** - Indian Ocean Dipole. Positive events can amplify El Nino teleconnections
over South America via atmospheric Rossby waves.

---
*Data updated daily from NOAA CPC / PSL via GitHub Actions.*
""")

# Footer
st.divider()
st.caption("Data sources: NOAA CPC (ONI, Nino 3.4, SAM) - NOAA PSL (MEI, IOD) - Updated daily via GitHub Actions")
