"""
fetch_sst.py
Downloads latest monthly SST and computes anomaly (vs 1991-2020 climatology)
from NOAA OISSTv2.1 high-res via PSL OPeNDAP.

Resolution: stride 8 on the 0.25-deg grid -> ~2-deg global grid (90x180 points).
Saves: data/sst_global_latest.csv  (columns: lat, lon, sst, anom, date)
"""

import re
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE   = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres"
STRIDE = 8          # every 8th point of 0.25-deg grid  -> ~2-deg resolution
FILL   = -9.96921e36


# ── OPeNDAP ASCII helpers ─────────────────────────────────────────────────────

def _get_1d(url_suffix):
    """Fetch a 1-D coordinate array from PSL OPeNDAP and return as np.array."""
    r = requests.get(f"{BASE}/{url_suffix}", timeout=30)
    r.raise_for_status()
    body = r.text.split("---------------------------------------------")[-1]
    values = re.sub(r"\[.*?\]", "", body)
    nums = []
    for tok in re.split(r"[,\s]+", values):
        tok = tok.strip()
        if tok:
            try:
                nums.append(float(tok))
            except ValueError:
                pass
    return np.array(nums)


def _get_2d_sst(url_suffix, nlat, nlon):
    """
    Fetch a 2-D SST slice [1][nlat][nlon] and return as (nlat, nlon) np.array.
    Handles OPeNDAP line-wrapping. Missing fill value replaced with NaN.
    """
    r = requests.get(f"{BASE}/{url_suffix}", timeout=60)
    r.raise_for_status()
    body = r.text.split("---------------------------------------------")[-1]

    # Strip all [index] tags and collect every numeric token
    clean = re.sub(r"\[\d+\]", "", body)
    nums  = []
    for tok in re.split(r"[,\s]+", clean):
        tok = tok.strip()
        if tok:
            try:
                nums.append(float(tok))
            except ValueError:
                pass

    arr = np.array(nums[:nlat * nlon], dtype=float).reshape(nlat, nlon)
    arr[np.abs(arr) > 1e30] = np.nan
    return arr


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_sst_global():
    print("Fetching SST global maps from PSL OPeNDAP...")

    # ── 1. Coordinate arrays ──────────────────────────────────────────────────
    lat_all = _get_1d("sst.mon.mean.nc.ascii?lat")   # 720 values, -89.875..89.875
    lon_all = _get_1d("sst.mon.mean.nc.ascii?lon")   # 1440 values, 0.125..359.875

    lat_idx = np.arange(0, len(lat_all), STRIDE)
    lon_idx = np.arange(0, len(lon_all), STRIDE)
    lats    = lat_all[lat_idx]                        # 90 values
    lons    = lon_all[lon_idx]                        # 180 values
    lons_180 = np.where(lons > 180, lons - 360, lons) # convert to -180/180

    # ── 2. Latest time index and date ─────────────────────────────────────────
    time_vals = _get_1d("sst.mon.mean.nc.ascii?time") # days since 1800-01-01
    t_last    = len(time_vals) - 1
    epoch     = pd.Timestamp("1800-01-01")
    last_date = epoch + pd.Timedelta(days=float(time_vals[-1]))
    date_str  = last_date.strftime("%Y-%m")
    month_idx = last_date.month - 1   # 0-indexed for climatology
    print(f"  Latest SST month: {date_str}  (time index {t_last})")

    # ── 3. Latest SST ─────────────────────────────────────────────────────────
    lat0, lat1 = lat_idx[0],  lat_idx[-1]
    lon0, lon1 = lon_idx[0],  lon_idx[-1]
    nlat = len(lats)
    nlon = len(lons_180)

    sst_arr = _get_2d_sst(
        f"sst.mon.mean.nc.ascii?"
        f"sst[{t_last}:1:{t_last}][{lat0}:{STRIDE}:{lat1}][{lon0}:{STRIDE}:{lon1}]",
        nlat, nlon,
    )
    print(f"  SST array shape: {sst_arr.shape}")

    # ── 4. Climatology for this month ─────────────────────────────────────────
    clim_arr = _get_2d_sst(
        f"sst.mon.ltm.1991-2020.nc.ascii?"
        f"sst[{month_idx}:1:{month_idx}][{lat0}:{STRIDE}:{lat1}][{lon0}:{STRIDE}:{lon1}]",
        nlat, nlon,
    )
    print(f"  Climatology array shape: {clim_arr.shape}")

    # ── 5. Anomaly and DataFrame ──────────────────────────────────────────────
    anom_arr = sst_arr - clim_arr
    records = []
    for i in range(nlat):
        for j in range(nlon):
            sst_val  = float(sst_arr[i, j])  if i < sst_arr.shape[0]  and j < sst_arr.shape[1]  else np.nan
            anom_val = float(anom_arr[i, j]) if i < anom_arr.shape[0] and j < anom_arr.shape[1] else np.nan
            if not np.isnan(sst_val):
                records.append({
                    "lat":  round(float(lats[i]),    3),
                    "lon":  round(float(lons_180[j]), 3),
                    "sst":  round(sst_val,            3),
                    "anom": round(anom_val,           3),
                    "date": date_str,
                })

    df = pd.DataFrame(records)
    out = DATA_DIR / "sst_global_latest.csv"
    df.to_csv(out, index=False)
    print(f"  SST saved: {len(df)} ocean points -> {out}")
    return df


if __name__ == "__main__":
    fetch_sst_global()
