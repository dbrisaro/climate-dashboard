"""
fetch_trends.py
Computes linear trends (units per decade) for SST, T2m, 10m wind speed, and SSH.

Sources:
  SST  : NOAA OISSTv2.1 monthly, 1982-present   PSL OPeNDAP (no auth required)
  T2m  : ERA5 monthly reanalysis, 1982-present   CDS API
  Wind : ERA5 monthly 10m u+v, 1982-present      CDS API
  SSH  : Satellite altimetry monthly SLA, 1993-present  CDS API

Outputs saved to data/trends/:
  sst_trend.csv   lat, lon, trend (C/decade),    period
  t2m_trend.csv   lat, lon, trend (C/decade),    period
  wind_trend.csv  lat, lon, trend (m/s/decade),  period
  ssh_trend.csv   lat, lon, trend (cm/decade),   period
"""

import re
import os
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "trends"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE   = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres"
STRIDE = 8   # ~2-degree resolution on the 0.25-deg OISST grid


# ── OPeNDAP helpers ───────────────────────────────────────────────────────────

def _get_1d(url_suffix):
    """Fetch a 1-D coordinate array from PSL OPeNDAP."""
    r = requests.get(f"{BASE}/{url_suffix}", timeout=30)
    r.raise_for_status()
    body = r.text.split("---------------------------------------------")[-1]
    clean = re.sub(r"\[.*?\]", "", body)
    nums = []
    for tok in re.split(r"[,\s]+", clean):
        tok = tok.strip()
        if tok:
            try:
                nums.append(float(tok))
            except ValueError:
                pass
    return np.array(nums)


def _get_sst_chunk(t0, t1, lat0, lat1, lon0, lon1, nlat, nlon):
    """
    Fetch SST time slice t0..t1 at strided lat/lon.
    Returns (n_months, nlat, nlon) array.
    """
    n_t = t1 - t0 + 1
    url = (f"{BASE}/sst.mon.mean.nc.ascii?"
           f"sst[{t0}:1:{t1}][{lat0}:{STRIDE}:{lat1}][{lon0}:{STRIDE}:{lon1}]")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    body = r.text.split("---------------------------------------------")[-1]
    clean = re.sub(r"\[\d+\]", "", body)
    nums = []
    for tok in re.split(r"[,\s]+", clean):
        tok = tok.strip()
        if tok:
            try:
                nums.append(float(tok))
            except ValueError:
                pass
    total = n_t * nlat * nlon
    arr = np.array(nums[:total], dtype=float).reshape(n_t, nlat, nlon)
    arr[np.abs(arr) > 1e30] = np.nan
    return arr


# ── Trend computation ─────────────────────────────────────────────────────────

def _ols_slope_per_decade(time_months, data_3d):
    """
    Compute OLS slope at each grid point.
    time_months : 1-D array, length T
    data_3d     : (T, nlat, nlon) array
    Returns     : (nlat, nlon) slope in original units per decade.
    """
    T, nlat, nlon = data_3d.shape
    t = np.asarray(time_months, dtype=float)
    t -= t.mean()   # center for numerical stability
    tt = np.sum(t ** 2)

    slope = np.full((nlat, nlon), np.nan)
    for i in range(nlat):
        for j in range(nlon):
            z = data_3d[:, i, j]
            mask = ~np.isnan(z)
            if mask.sum() < 24:   # require at least 2 years of data
                continue
            tm = t[mask]
            zm = z[mask]
            ttm = np.sum(tm ** 2)
            if ttm == 0:
                continue
            slope[i, j] = np.sum(tm * zm) / ttm

    return slope * 120.0   # convert months^-1 -> per decade


# ── SST trend (OISSTv2.1) ─────────────────────────────────────────────────────

def compute_sst_trend():
    out = DATA_DIR / "sst_trend.csv"
    if out.exists() and (datetime.utcnow().timestamp() - out.stat().st_mtime) < 86400 * 25:
        print(f"SST trend up to date, skipping recompute -> {out}")
        return

    print("Computing SST trend from OISSTv2.1 (1982-present)...")

    lat_all   = _get_1d("sst.mon.mean.nc.ascii?lat")
    lon_all   = _get_1d("sst.mon.mean.nc.ascii?lon")
    time_vals = _get_1d("sst.mon.mean.nc.ascii?time")   # days since 1800-01-01

    lat_idx  = np.arange(0, len(lat_all), STRIDE)
    lon_idx  = np.arange(0, len(lon_all), STRIDE)
    lats     = lat_all[lat_idx]
    lons     = lon_all[lon_idx]
    lons_180 = np.where(lons > 180, lons - 360, lons)
    nlat, nlon = len(lats), len(lons)
    lat0, lat1 = lat_idx[0], lat_idx[-1]
    lon0, lon1 = lon_idx[0], lon_idx[-1]

    epoch = pd.Timestamp("1800-01-01")
    dates = [epoch + pd.Timedelta(days=float(tv)) for tv in time_vals]

    start_t = next((i for i, d in enumerate(dates) if d.year == 1982 and d.month == 1), 0)
    end_t   = len(time_vals) - 1
    T       = end_t - start_t + 1
    print(f"  Period: {dates[start_t].strftime('%Y-%m')} to {dates[end_t].strftime('%Y-%m')} ({T} months)")

    # Fetch year by year and stack
    slices = []
    CHUNK  = 12
    for t0 in range(start_t, end_t + 1, CHUNK):
        t1    = min(t0 + CHUNK - 1, end_t)
        chunk = _get_sst_chunk(t0, t1, lat0, lat1, lon0, lon1, nlat, nlon)
        slices.append(chunk)
        print(f"  Fetched {dates[t0].strftime('%Y-%m')} - {dates[t1].strftime('%Y-%m')}")

    data_3d = np.concatenate(slices, axis=0)   # (T, nlat, nlon)
    trend   = _ols_slope_per_decade(np.arange(T), data_3d)

    period_str = f"{dates[start_t].year}-{dates[end_t].year}"
    records = []
    for i in range(nlat):
        for j in range(nlon):
            v = trend[i, j]
            if not np.isnan(v):
                records.append({
                    "lat":    round(float(lats[i]),    3),
                    "lon":    round(float(lons_180[j]), 3),
                    "trend":  round(float(v),          4),
                    "period": period_str,
                })
    df = pd.DataFrame(records)
    df.to_csv(out, index=False)
    print(f"SST trend saved: {len(df)} ocean points -> {out}")


# ── CDS helper ────────────────────────────────────────────────────────────────

def _get_cds_client():
    import cdsapi
    key = os.environ.get("CDSAPI_KEY")
    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    if key:
        return cdsapi.Client(url=url, key=key, quiet=True)
    return cdsapi.Client(quiet=True)


def _nc_to_trend_csv(nc_path, out_csv, derive_fn, period_str):
    """
    Load NetCDF at nc_path, apply derive_fn(ds) -> (lats, lons_180, data_3d),
    compute trend, save CSV.
    derive_fn must return (lats 1-D, lons_180 1-D, data_3d (T, nlat, nlon)).
    """
    import xarray as xr
    ds     = xr.open_dataset(nc_path)
    lats, lons_180, data_3d = derive_fn(ds)
    T      = data_3d.shape[0]
    trend  = _ols_slope_per_decade(np.arange(T), data_3d)

    records = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons_180):
            v = trend[i, j]
            if not np.isnan(v):
                records.append({
                    "lat":    round(float(lat), 3),
                    "lon":    round(float(lon), 3),
                    "trend":  round(float(v),   4),
                    "period": period_str,
                })
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Trend saved: {len(df)} grid points -> {out_csv}")
    return df


# ── ERA5 T2m trend ────────────────────────────────────────────────────────────

def compute_era5_t2m_trend():
    out = DATA_DIR / "t2m_trend.csv"
    if out.exists() and (datetime.utcnow().timestamp() - out.stat().st_mtime) < 86400 * 25:
        print(f"T2m trend up to date, skipping -> {out}")
        return

    now     = datetime.utcnow()
    nc_path = DATA_DIR / f"era5_t2m_monthly_1982_{now.year}.nc"

    if not nc_path.exists():
        print("Downloading ERA5 monthly T2m 1982-present from CDS...")
        c = _get_cds_client()
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable":     "2m_temperature",
                "year":         [str(y) for y in range(1982, now.year + 1)],
                "month":        [f"{m:02d}" for m in range(1, 13)],
                "time":         "00:00",
                "grid":         [2.5, 2.5],
                "format":       "netcdf",
            },
            str(nc_path),
        )
        print(f"Downloaded -> {nc_path} ({nc_path.stat().st_size // 1024} kB)")
    else:
        print(f"Using cached {nc_path}")

    def _derive_t2m(ds):
        lat_dim  = next(d for d in ds.dims if d in ("latitude", "lat"))
        lon_dim  = next(d for d in ds.dims if d in ("longitude", "lon"))
        time_dim = next(d for d in ds.dims if d in ("time", "valid_time"))
        vname    = next(v for v in ds.data_vars if "t2m" in v.lower() or "temperature" in v.lower())
        da       = ds[vname].sortby(time_dim)
        lats     = da[lat_dim].values
        lons     = da[lon_dim].values
        lons_180 = np.where(lons > 180, lons - 360, lons)
        data     = da.values.astype(float)
        if data.ndim == 4:
            data = data[:, 0, :, :]
        return lats, lons_180, data

    _nc_to_trend_csv(nc_path, out, _derive_t2m, f"1982-{now.year}")


# ── ERA5 wind speed trend ─────────────────────────────────────────────────────

def compute_era5_wind_trend():
    out = DATA_DIR / "wind_trend.csv"
    if out.exists() and (datetime.utcnow().timestamp() - out.stat().st_mtime) < 86400 * 25:
        print(f"Wind trend up to date, skipping -> {out}")
        return

    now     = datetime.utcnow()
    nc_path = DATA_DIR / f"era5_wind_monthly_1982_{now.year}.nc"

    if not nc_path.exists():
        print("Downloading ERA5 monthly 10m wind 1982-present from CDS...")
        c = _get_cds_client()
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable":     [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year":         [str(y) for y in range(1982, now.year + 1)],
                "month":        [f"{m:02d}" for m in range(1, 13)],
                "time":         "00:00",
                "grid":         [2.5, 2.5],
                "format":       "netcdf",
            },
            str(nc_path),
        )
        print(f"Downloaded -> {nc_path} ({nc_path.stat().st_size // 1024} kB)")
    else:
        print(f"Using cached {nc_path}")

    def _derive_wind(ds):
        lat_dim  = next(d for d in ds.dims if d in ("latitude", "lat"))
        lon_dim  = next(d for d in ds.dims if d in ("longitude", "lon"))
        time_dim = next(d for d in ds.dims if d in ("time", "valid_time"))
        u_var    = next(v for v in ds.data_vars if "u10" in v.lower() or "u_comp" in v.lower())
        v_var    = next(v for v in ds.data_vars if "v10" in v.lower() or "v_comp" in v.lower())
        da_u     = ds[u_var].sortby(time_dim)
        da_v     = ds[v_var].sortby(time_dim)
        lats     = da_u[lat_dim].values
        lons     = da_u[lon_dim].values
        lons_180 = np.where(lons > 180, lons - 360, lons)
        u = da_u.values.astype(float)
        v = da_v.values.astype(float)
        if u.ndim == 4:
            u, v = u[:, 0, :, :], v[:, 0, :, :]
        speed = np.sqrt(u ** 2 + v ** 2)
        return lats, lons_180, speed

    _nc_to_trend_csv(nc_path, out, _derive_wind, f"1982-{now.year}")


# ── SSH trend (satellite altimetry) ──────────────────────────────────────────

def compute_ssh_trend():
    out = DATA_DIR / "ssh_trend.csv"
    if out.exists() and (datetime.utcnow().timestamp() - out.stat().st_mtime) < 86400 * 25:
        print(f"SSH trend up to date, skipping -> {out}")
        return

    now     = datetime.utcnow()
    nc_path = DATA_DIR / f"ssh_monthly_1993_{now.year}.nc"

    if not nc_path.exists():
        print("Downloading satellite sea level anomaly 1993-present from CDS...")
        try:
            c = _get_cds_client()
            c.retrieve(
                "satellite-sea-level-global",
                {
                    "variable": "monthly_mean",
                    "year":     [str(y) for y in range(1993, now.year + 1)],
                    "month":    [f"{m:02d}" for m in range(1, 13)],
                    "format":   "zip",
                },
                str(nc_path.with_suffix(".zip")),
            )
            # Unzip and locate NetCDF
            import zipfile
            with zipfile.ZipFile(str(nc_path.with_suffix(".zip")), "r") as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if nc_names:
                    zf.extract(nc_names[0], str(DATA_DIR))
                    extracted = DATA_DIR / nc_names[0]
                    if extracted != nc_path:
                        extracted.rename(nc_path)
            nc_path.with_suffix(".zip").unlink(missing_ok=True)
            print(f"Downloaded -> {nc_path} ({nc_path.stat().st_size // 1024} kB)")
        except Exception as e:
            print(f"WARNING: SSH download failed: {e}")
            print("  SSH trend will not be available.")
            return
    else:
        print(f"Using cached {nc_path}")

    try:
        import xarray as xr
        ds      = xr.open_dataset(nc_path)
        sla_var = next(
            (v for v in ds.data_vars if "sla" in v.lower() or "sea_level" in v.lower()),
            list(ds.data_vars)[0],
        )
        da      = ds[sla_var]
        dims    = set(da.dims)
        lat_dim  = next(d for d in dims if d in ("latitude", "lat"))
        lon_dim  = next(d for d in dims if d in ("longitude", "lon"))
        time_dim = next(d for d in dims if "time" in d.lower())
        da       = da.sortby(time_dim)

        lats     = da[lat_dim].values
        lons     = da[lon_dim].values

        # Subsample to ~2-degree resolution
        dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
        dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
        s_lat = max(1, round(2.0 / dlat))
        s_lon = max(1, round(2.0 / dlon))
        da_sub   = da.isel({lat_dim: slice(None, None, s_lat),
                            lon_dim: slice(None, None, s_lon)})

        lats_s   = da_sub[lat_dim].values
        lons_s   = da_sub[lon_dim].values
        lons_180 = np.where(lons_s > 180, lons_s - 360, lons_s)

        data_3d  = da_sub.values.astype(float)   # SLA in metres
        if data_3d.ndim == 4:
            data_3d = data_3d[:, 0, :, :]
        T = data_3d.shape[0]

        # Trend in cm/decade (SLA in m -> * 100 = cm, already per decade from helper)
        trend = _ols_slope_per_decade(np.arange(T), data_3d) * 100.0

        period_str = f"1993-{now.year}"
        records = []
        for i, lat in enumerate(lats_s):
            for j, lon in enumerate(lons_180):
                v = trend[i, j]
                if not np.isnan(v):
                    records.append({
                        "lat":    round(float(lat), 3),
                        "lon":    round(float(lon), 3),
                        "trend":  round(float(v),   4),
                        "period": period_str,
                    })
        df = pd.DataFrame(records)
        df.to_csv(out, index=False)
        print(f"SSH trend saved: {len(df)} ocean points -> {out}")
    except Exception as e:
        print(f"ERROR processing SSH: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    compute_sst_trend()
    compute_era5_t2m_trend()
    compute_era5_wind_trend()
    compute_ssh_trend()
