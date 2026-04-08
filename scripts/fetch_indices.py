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

    df = pd.DataFrame(rows)
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

    df = pd.DataFrame(rows)
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

    rows = []
    for line in resp.text.strip().splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        if not parts[0].isdigit():
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            val   = float(parts[9])   # NINO3.4 ANOM
            if val not in (-99.99, -999.0):
                rows.append({"year": year, "month": month, "nino34": val})
        except ValueError:
            pass

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    out = DATA_DIR / "nino34.csv"
    df.to_csv(out, index=False)
    print(f"Niño 3.4 saved: {len(df)} rows -> {out}")
    return df


# ── Atlantic Niño (ATL3) ──────────────────────────────────────────────────────
# ATL3 = SST anomaly averaged over 3°S–3°N, 20°W–0°E.
# No public pre-computed download is currently available without authentication
# (PSL and KNMI sources return 404 or require login).
# TODO: compute from NOAA ERSSTv5 NetCDF once xarray/netcdf4 are added to deps.


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fetching climate indices...")
    fetch_oni()
    fetch_mei()
    fetch_sam()
    fetch_iod()
    fetch_nino34()
    print("Done.")
