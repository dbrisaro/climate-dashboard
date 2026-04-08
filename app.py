import streamlit as st
import pandas as pd
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
        "Forecasts from IRI (International Research Institute for Climate and Society) "
        "and NOAA CPC. Images are fetched directly from the source and updated monthly."
    )

    # IRI figures
    st.markdown("### IRI - ENSO forecast")
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
            with st.expander(label, expanded=(filtered.index(url) < 3)):
                st.image(url, width=700)
    else:
        st.warning("Could not load IRI figures. Visit the source directly.")
        st.link_button(
            "Open IRI ENSO forecast page",
            "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/",
        )

    st.divider()

    # CPC
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
