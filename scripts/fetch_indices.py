"""
fetch_indices.py
Downloads climate indices from public sources and saves them as CSVs in data/.

Indices:
- ONI  (Oceanic Nino Index)         — NOAA CPC
- MEI  (Multivariate ENSO Index)    — NOAA PSL
- SAM  (Southern Annular Mode)      — NOAA CPC
- IOD  (Indian Ocean Dipole / DMI)  — NOAA PSL
"""

import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── ONI ──────────────────────────────────────────────────────────────────────

def fetch_oni():
    """
    NOAA CPC ONI table (3-month running mean of ERSST.v5 SST anomalies, Nino3.4).
    URL: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
    Format: year + 12 seasonal values (DJF, JFM, ..., NDJ)
    """
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    rows = []
    for line in resp.text.strip().splitlines():
        parts = line.split()
        # Format: SEAS YR TOTAL ANOM  (e.g. "DJF 1950 24.72 -1.53")
        if len(parts) < 4:
            continue
        if not parts[1].isdigit():
            continue
        try:
            season = parts[0]
            year = int(parts[1])
            val = float(parts[3])  # ANOM column
            rows.append({"year": year, "season": season, "oni": val})
        except ValueError:
            pass

    # Map each 3-month season to the middle month for time-series plotting
    season_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    df = pd.DataFrame(rows)
    df["month"] = df["season"].map(season_to_month)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    out = DATA_DIR / "oni.csv"
    df.to_csv(out, index=False)
    print(f"ONI saved: {len(df)} rows -> {out}")
    return df


# ── MEI ──────────────────────────────────────────────────────────────────────

def fetch_mei():
    """
    NOAA PSL MEI v2 (bimonthly).
    URL: https://psl.noaa.gov/enso/mei/data/meiv2.data
    Format: year + 12 bimonthly values (DJFM ... NDJF)
    """
    url = "https://psl.noaa.gov/enso/mei/data/meiv2.data"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    bimonths = ["DJFM","JFMA","FMAM","MAMJ","AMJJ","MJJA","JJAS","JASO","ASON","SOND","ONDS","NDJF"]
    rows = []
    for line in resp.text.strip().splitlines():
        parts = line.split()
        if len(parts) < 13:  # skip header/metadata lines (e.g. "1979 2026")
            continue
        if not parts[0].lstrip("-").isdigit():
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for i, bm in enumerate(bimonths):
            if i + 1 < len(parts):
                try:
                    val = float(parts[i + 1])
                    if val != -999.0:
                        rows.append({"year": year, "bimonth": bm, "mei": val})
                except ValueError:
                    pass

    # Map each bimonthly season to its first month for time-series plotting
    bimonth_to_month = {
        "DJFM": 1, "JFMA": 2, "FMAM": 3, "MAMJ": 4,
        "AMJJ": 5, "MJJA": 6, "JJAS": 7, "JASO": 8,
        "ASON": 9, "SOND": 10, "ONDS": 11, "NDJF": 12,
    }
    df = pd.DataFrame(rows)
    df["month"] = df["bimonth"].map(bimonth_to_month)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    out = DATA_DIR / "mei.csv"
    df.to_csv(out, index=False)
    print(f"MEI saved: {len(df)} rows -> {out}")
    return df


# ── SAM ──────────────────────────────────────────────────────────────────────

def fetch_sam():
    """
    NOAA CPC AAO/SAM monthly index.
    URL: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii
    Format: year month value
    """
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    rows = []
    for line in resp.text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append({
                    "year":  int(parts[0]),
                    "month": int(parts[1]),
                    "sam":   float(parts[2])
                })
            except ValueError:
                pass

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year","month"]].assign(day=1))
    out = DATA_DIR / "sam.csv"
    df.to_csv(out, index=False)
    print(f"SAM saved: {len(df)} rows -> {out}")
    return df


# ── IOD ──────────────────────────────────────────────────────────────────────

def fetch_iod():
    """
    Indian Ocean Dipole (DMI) monthly index from NOAA PSL.
    URL: https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data
    Format: year + 12 monthly values
    """
    url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    rows = []
    for line in resp.text.strip().splitlines():
        parts = line.split()
        if not parts or not parts[0].lstrip("-").isdigit():
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for month in range(1, 13):
            if month < len(parts):
                try:
                    val = float(parts[month])
                    if val != -9999.0 and val != -999.0:
                        rows.append({"year": year, "month": month, "iod": val})
                except ValueError:
                    pass

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year","month"]].assign(day=1))
    out = DATA_DIR / "iod.csv"
    df.to_csv(out, index=False)
    print(f"IOD saved: {len(df)} rows -> {out}")
    return df


# ── Niño 3.4 ─────────────────────────────────────────────────────────────────

def fetch_nino34():
    """
    NOAA CPC ERSSTv5 Nino indices (1991-2020 baseline).
    URL: https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
    Format: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
    We extract the NINO3.4 ANOM column (index 9).
    """
    url = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    rows34, rows12 = [], []
    for line in resp.text.strip().splitlines():
        parts = line.split()
        # Format: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
        if len(parts) < 10:
            continue
        if not parts[0].isdigit():
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            v34   = float(parts[9])  # NINO3.4 ANOM
            v12   = float(parts[3])  # NINO1+2 ANOM (Coastal Nino)
            if v34 not in (-99.99, -999.0):
                rows34.append({"year": year, "month": month, "nino34": v34})
            if v12 not in (-99.99, -999.0):
                rows12.append({"year": year, "month": month, "nino12": v12})
        except ValueError:
            pass

    df34 = pd.DataFrame(rows34)
    df34["date"] = pd.to_datetime(df34[["year", "month"]].assign(day=1))
    out34 = DATA_DIR / "nino34.csv"
    df34.to_csv(out34, index=False)
    print(f"Nino 3.4 saved: {len(df34)} rows -> {out34}")

    df12 = pd.DataFrame(rows12)
    df12["date"] = pd.to_datetime(df12[["year", "month"]].assign(day=1))
    out12 = DATA_DIR / "nino12.csv"
    df12.to_csv(out12, index=False)
    print(f"Nino 1+2 (Coastal) saved: {len(df12)} rows -> {out12}")

    return df34


# ── Atlantic Niño (ATL3) ──────────────────────────────────────────────────────
# ATL3 = SST anomaly averaged over 3°S–3°N, 20°W–0°E.
# No public pre-computed download is currently available without authentication
# (PSL and KNMI sources return 404 or require login).
# TODO: compute from NOAA ERSSTv5 NetCDF once xarray/netcdf4 are added to deps.


# ── CPC/IRI ENSO seasonal probabilities ──────────────────────────────────────

def fetch_enso_probs():
    """
    Scrape IRI/CPC official ENSO seasonal probability forecast.
    Tries IRI ENSO current page first; falls back to CPC advisory page.
    Saves data/enso_probs.csv with columns:
      season, p_nina, p_neutral, p_nino, source, issued
    """
    rows = _scrape_iri_probs() or _scrape_cpc_probs()
    if not rows:
        print("ENSO probs: could not scrape any source, skipping.")
        return None

    df = pd.DataFrame(rows).drop_duplicates(subset=["season"]).reset_index(drop=True)
    out = DATA_DIR / "enso_probs.csv"
    df.to_csv(out, index=False)
    print(f"ENSO probs saved: {len(df)} seasons -> {out}  (source: {df['source'].iloc[0]})")
    return df


def _parse_pct(text):
    """Parse a percentage string like '31%' or '31' -> float, or None."""
    try:
        return float(text.replace("%", "").replace("<", "").strip())
    except (ValueError, AttributeError):
        return None


def _scrape_iri_probs():
    """
    Scrape the IRI ENSO forecast page for the seasonal probability table.
    The table has columns: Season | La Nina (%) | Neutral (%) | El Nino (%)
    """
    try:
        url = "https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Known 3-month season codes
        season_codes = {
            "DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ",
        }

        rows = []
        for table in soup.find_all("table"):
            trs = table.find_all("tr")
            for tr in trs:
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if len(cells) < 4:
                    continue
                season = cells[0].upper().strip()
                if season not in season_codes:
                    continue
                p_nina    = _parse_pct(cells[1])
                p_neutral = _parse_pct(cells[2])
                p_nino    = _parse_pct(cells[3])
                if None in (p_nina, p_neutral, p_nino):
                    continue
                rows.append({
                    "season":    season,
                    "p_nina":    p_nina,
                    "p_neutral": p_neutral,
                    "p_nino":    p_nino,
                    "source":    "IRI",
                    "issued":    pd.Timestamp.utcnow().strftime("%Y-%m"),
                })
        return rows if rows else None
    except Exception as e:
        print(f"IRI scrape failed: {e}")
        return None


def _scrape_cpc_probs():
    """
    Scrape the CPC ENSO advisory page for the seasonal probability table.
    """
    try:
        url = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        season_codes = {
            "DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ",
        }

        rows = []
        for table in soup.find_all("table"):
            text_lower = table.get_text(" ", strip=True).lower()
            if "el ni" not in text_lower and "la ni" not in text_lower:
                continue
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if len(cells) < 4:
                    continue
                season = cells[0].upper().strip()
                if season not in season_codes:
                    continue
                # CPC column order may be El Nino | Neutral | La Nina
                # or La Nina | Neutral | El Nino — try to detect from header
                nums = [_parse_pct(c) for c in cells[1:4]]
                if None in nums:
                    continue
                rows.append({
                    "season":    season,
                    "p_nina":    nums[0],
                    "p_neutral": nums[1],
                    "p_nino":    nums[2],
                    "source":    "CPC",
                    "issued":    pd.Timestamp.utcnow().strftime("%Y-%m"),
                })
        return rows if rows else None
    except Exception as e:
        print(f"CPC scrape failed: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching climate indices...")
    fetch_oni()
    fetch_mei()
    fetch_sam()
    fetch_iod()
    fetch_nino34()
    fetch_enso_probs()
    print("Done.")
