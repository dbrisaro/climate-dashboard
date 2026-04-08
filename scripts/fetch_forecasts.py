"""
fetch_forecasts.py
Downloads ECMWF SEAS5 seasonal SST anomaly forecast from Copernicus C3S.
Extracts Nino3.4 (5S-5N, 170W-120W) ensemble mean and saves to CSV.

Dataset : seasonal-postprocessed-single-levels
Variable: sea_surface_temperature_anomaly  -> 'ssta' in NetCDF
Product : ensemble_mean
Leads   : months 1-6

Area subsetting is done in post-processing (download full global grid)
because CDS rejects area requests for some ensemble configurations.

API key is read from:
  1. ~/.cdsapirc           (local)
  2. CDSAPI_KEY env var    (GitHub Actions)
  3. st.secrets            (Streamlit Cloud) -- only when imported from app.py
"""

import os
import cdsapi
import requests
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data" / "forecasts"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Nino3.4 box: 5S-5N, 170W-120W
LAT_N, LAT_S =  5.0, -5.0
LON_W, LON_E = -170.0, -120.0   # will wrap to 190-240 if needed

# South America bounding box
SA_LAT_N, SA_LAT_S = 15.0, -60.0
SA_LON_W, SA_LON_E = -90.0, -30.0


def get_cds_client():
    key = os.environ.get("CDSAPI_KEY")
    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    if key:
        return cdsapi.Client(url=url, key=key, quiet=True)
    # try Streamlit secrets (only available when running inside Streamlit)
    try:
        import streamlit as st
        key = st.secrets["cds"]["key"]
        url = st.secrets["cds"].get("url", url)
        return cdsapi.Client(url=url, key=key, quiet=True)
    except Exception:
        pass
    return cdsapi.Client(quiet=True)   # fall back to ~/.cdsapirc


def fetch_seas5_nino34():
    """
    Download SEAS5 ensemble-mean SST anomaly for the current init month,
    extract Nino3.4 spatial mean, and save to:
      data/forecasts/nino34_seas5_mean.csv
    """
    now   = datetime.utcnow()
    year  = str(now.year)
    month = f"{now.month:02d}"

    nc_path = DATA_DIR / f"seas5_ssta_{year}{month}.nc"

    if not nc_path.exists():
        print(f"Downloading SEAS5 SST anomaly {year}-{month} from C3S ...")
        c = get_cds_client()
        c.retrieve(
            "seasonal-postprocessed-single-levels",
            {
                "originating_centre": "ecmwf",
                "system": "51",
                "variable": "sea_surface_temperature_anomaly",
                "product_type": "ensemble_mean",
                "year": year,
                "month": month,
                "leadtime_month": ["1", "2", "3", "4", "5", "6"],
                "format": "netcdf",
            },
            str(nc_path),
        )
        print(f"Downloaded -> {nc_path}  ({nc_path.stat().st_size/1024:.0f} kB)")
    else:
        print(f"Using cached {nc_path}")

    # ── Load and extract Nino3.4 spatial mean ────────────────────────────────
    ds = xr.open_dataset(nc_path)

    # Find SST anomaly variable (name in file is 'ssta')
    sst_var = next(
        (v for v in ds.data_vars if "sst" in v.lower()),
        list(ds.data_vars)[0],
    )
    da = ds[sst_var]

    # Identify dimension names (vary across CDS versions)
    dims = set(da.dims)

    lat_dim = next(d for d in dims if d in ("latitude", "lat"))
    lon_dim = next(d for d in dims if d in ("longitude", "lon"))
    lead_dim = next(
        (d for d in dims if d in ("forecastMonth", "leadtime_month", "step")),
        None,
    )
    time_dim = next(
        (d for d in dims if d in ("forecast_reference_time", "time")),
        None,
    )

    # Subset to Nino3.4 box (handle 0-360 and -180-180 conventions)
    lons = da[lon_dim].values
    if lons.max() > 180:
        # 0-360 convention: 170W-120W -> 190-240
        lon_sel = da.sel(
            {lat_dim: slice(LAT_N, LAT_S),
             lon_dim: slice(360 + LON_W, 360 + LON_E)}
        )
    else:
        lon_sel = da.sel(
            {lat_dim: slice(LAT_N, LAT_S),
             lon_dim: slice(LON_W, LON_E)}
        )

    # Spatial mean over Nino3.4 box
    spatial_mean = lon_sel.mean(dim=[lat_dim, lon_dim])   # shape: (lead,) or (time, lead)

    # Build rows
    rows = []
    init_date = pd.Timestamp(year=int(year), month=int(month), day=1)

    if lead_dim and lead_dim in spatial_mean.dims:
        for lead_val in spatial_mean[lead_dim].values:
            # lead_val is the forecast month number (1-6)
            try:
                lead_k = int(lead_val)
            except (TypeError, ValueError):
                continue
            fc_date = init_date + pd.DateOffset(months=lead_k)

            val = float(
                spatial_mean.sel({lead_dim: lead_val}).values
                if time_dim not in spatial_mean.dims
                else spatial_mean.sel({lead_dim: lead_val}).mean().values
            )

            rows.append({
                "init_year":     int(year),
                "init_month":    int(month),
                "forecast_date": fc_date,
                "lead_month":    lead_k,
                "anom_seas5":    round(val, 3),
            })
    else:
        # fallback: single value
        val = float(spatial_mean.mean().values)
        rows.append({
            "init_year":  int(year),
            "init_month": int(month),
            "forecast_date": init_date + pd.DateOffset(months=1),
            "lead_month": 1,
            "anom_seas5": round(val, 3),
        })

    df = pd.DataFrame(rows)
    out = DATA_DIR / "nino34_seas5_mean.csv"
    df.to_csv(out, index=False)
    print(f"SEAS5 Nino3.4 mean saved: {len(df)} rows -> {out}")
    return df


def _extract_sa_grid(nc_path, init_year, init_month):
    """
    Load a downloaded SEAS5 NetCDF, subset to South America,
    keep the lead-month dimension, and return a flat DataFrame with
    columns: lat, lon, lead_month, forecast_date, anom.
    Longitudes are normalised to -180/180.
    Any non-lat/lon/lead dims (time, member …) are averaged out.
    """
    ds  = xr.open_dataset(nc_path)
    var = list(ds.data_vars)[0]
    da  = ds[var]

    dims    = set(da.dims)
    lat_dim  = next(d for d in dims if d in ("latitude", "lat"))
    lon_dim  = next(d for d in dims if d in ("longitude", "lon"))
    lead_dim = next(
        (d for d in dims if d in ("forecastMonth", "leadtime_month", "step")),
        None,
    )

    # Subset SA bounding box (handle 0-360 and -180/180 conventions)
    lons = da[lon_dim].values
    if lons.max() > 180:
        lon_sel = da.sel({
            lat_dim: slice(SA_LAT_N, SA_LAT_S),
            lon_dim: slice(360 + SA_LON_W, 360 + SA_LON_E),
        })
    else:
        lon_sel = da.sel({
            lat_dim: slice(SA_LAT_N, SA_LAT_S),
            lon_dim: slice(SA_LON_W, SA_LON_E),
        })

    # Average over dims that are not lat/lon/lead (time, member …)
    keep = {lat_dim, lon_dim}
    if lead_dim:
        keep.add(lead_dim)
    extra = [d for d in lon_sel.dims if d not in keep]
    spatial = lon_sel.mean(dim=extra) if extra else lon_sel

    df = (
        spatial
        .to_dataframe(name="anom")
        .reset_index()
        .rename(columns={lat_dim: "lat", lon_dim: "lon"})
    )
    # Do NOT round here — callers may need to apply unit conversions first

    # Normalise longitude to -180/180
    if df["lon"].max() > 180:
        df["lon"] = df["lon"].apply(lambda x: x - 360 if x > 180 else x)

    # Add lead_month column (integer 1-6)
    if lead_dim and lead_dim in df.columns:
        df = df.rename(columns={lead_dim: "lead_month"})
        df["lead_month"] = df["lead_month"].astype(int)
    else:
        df["lead_month"] = 1

    # Compute forecast valid date from init + lead
    df["forecast_date"] = df["lead_month"].apply(
        lambda k: (pd.Timestamp(year=init_year, month=init_month, day=1)
                   + pd.DateOffset(months=int(k))).strftime("%Y-%m")
    )

    df = df[["lat", "lon", "lead_month", "forecast_date", "anom"]].dropna(subset=["anom"])
    df["init_year"]  = init_year
    df["init_month"] = init_month
    return df


def fetch_seas5_sa_maps():
    """
    Download SEAS5 ensemble-mean seasonal forecasts for South America (leads 1-3).

    T2m  : from seasonal-postprocessed-single-levels (pre-computed anomaly, °C)
    Precip: from seasonal-monthly-single-levels (raw tprate kg/m²/s → mm/day)

    Saves:
      data/forecasts/seas5_t2m_anom_SA.csv   — T2m anomaly (°C)
      data/forecasts/seas5_prcp_mmday_SA.csv — absolute precipitation (mm/day)
    """
    now   = datetime.utcnow()
    year  = str(now.year)
    month = f"{now.month:02d}"

    c = get_cds_client()

    # ── 1. T2m anomaly (post-processed) ─────────────────────────────────────
    nc_t2m  = DATA_DIR / f"seas5_t2m_SA_{year}{month}.nc"
    csv_t2m = DATA_DIR / "seas5_t2m_anom_SA.csv"

    if not nc_t2m.exists():
        print(f"Downloading SEAS5 2m_temperature_anomaly {year}-{month} …")
        try:
            c.retrieve(
                "seasonal-postprocessed-single-levels",
                {
                    "originating_centre": "ecmwf",
                    "system":             "51",
                    "variable":           "2m_temperature_anomaly",
                    "product_type":       "ensemble_mean",
                    "year":               year,
                    "month":              month,
                    "leadtime_month":     ["1", "2", "3"],
                    "format":             "netcdf",
                },
                str(nc_t2m),
            )
            print(f"Downloaded -> {nc_t2m}  ({nc_t2m.stat().st_size/1024:.0f} kB)")
        except Exception as e:
            print(f"ERROR downloading T2m anomaly: {e}")
    else:
        print(f"Using cached {nc_t2m}")

    if nc_t2m.exists():
        try:
            df = _extract_sa_grid(nc_t2m, int(year), int(month))
            df["anom"] = df["anom"].round(4)
            df.to_csv(csv_t2m, index=False)
            print(f"T2m anomaly saved: {len(df)} grid points -> {csv_t2m}")
        except Exception as e:
            print(f"ERROR processing T2m: {e}")

    # ── 2. Precipitation (raw monthly, convert to mm/day) ────────────────────
    nc_prcp  = DATA_DIR / f"seas5_prcp_SA_{year}{month}.nc"
    csv_prcp = DATA_DIR / "seas5_prcp_mmday_SA.csv"

    if not nc_prcp.exists():
        print(f"Downloading SEAS5 total_precipitation {year}-{month} …")
        try:
            c.retrieve(
                "seasonal-monthly-single-levels",
                {
                    "originating_centre": "ecmwf",
                    "system":             "51",
                    "variable":           "total_precipitation",
                    "product_type":       "ensemble_mean",
                    "year":               year,
                    "month":              month,
                    "leadtime_month":     ["1", "2", "3"],
                    "format":             "netcdf",
                },
                str(nc_prcp),
            )
            print(f"Downloaded -> {nc_prcp}  ({nc_prcp.stat().st_size/1024:.0f} kB)")
        except Exception as e:
            print(f"ERROR downloading precipitation: {e}")
    else:
        print(f"Using cached {nc_prcp}")

    if nc_prcp.exists():
        try:
            df = _extract_sa_grid(nc_prcp, int(year), int(month))
            # tprate is in m/s; convert to mm/day: × 86400 (s→day) × 1000 (m→mm)
            df["anom"] = (df["anom"] * 86_400_000).round(2)
            df.to_csv(csv_prcp, index=False)
            print(f"Precipitation saved: {len(df)} grid points -> {csv_prcp}")
        except Exception as e:
            print(f"ERROR processing precipitation: {e}")


def fetch_nmme_probs():
    """
    Download CPC NMME tercile probability forecasts for SA from the public FTP.
    URL: https://ftp.cpc.ncep.noaa.gov/NMME/prob/netcdf/{var}.{YYYYMM}.prob.seas.nc
    Variables: tmp2m (temperature), prate (precipitation)
    Probabilities: prob_above, prob_norm, prob_below  (fractions 0-1, 7 leads)

    Saves:
      data/forecasts/nmme_tmp2m_probs_SA.csv
      data/forecasts/nmme_prate_probs_SA.csv
    Columns: lat, lon, lead_month, forecast_date, prob_above, prob_norm, prob_below,
             init_year, init_month
    """
    BASE = "https://ftp.cpc.ncep.noaa.gov/NMME/prob/netcdf"
    now  = datetime.utcnow()

    # Try current month first, fall back to previous if file not yet published
    init_ym = None
    for delta in [0, -1, -2]:
        candidate = (now.replace(day=1) + pd.DateOffset(months=delta))
        ym = candidate.strftime("%Y%m")
        test_url = f"{BASE}/tmp2m.{ym}.prob.seas.nc"
        try:
            r = requests.head(test_url, timeout=10)
            if r.status_code == 200:
                init_ym = ym
                break
        except Exception:
            pass
    if init_ym is None:
        print("NMME prob files not reachable — skipping")
        return

    print(f"Using NMME init month: {init_ym}")

    for var_name, out_stem in [("tmp2m", "nmme_tmp2m_probs_SA"),
                                ("prate", "nmme_prate_probs_SA")]:
        nc_path = DATA_DIR / f"{var_name}_{init_ym}_prob.nc"
        out_csv = DATA_DIR / f"{out_stem}.csv"
        url     = f"{BASE}/{var_name}.{init_ym}.prob.seas.nc"

        if not nc_path.exists():
            print(f"Downloading {url} …")
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                nc_path.write_bytes(r.content)
                print(f"Downloaded {nc_path}  ({len(r.content)//1024} kB)")
            except Exception as e:
                print(f"ERROR downloading {url}: {e}")
                continue
        else:
            print(f"Using cached {nc_path}")

        try:
            ds = xr.open_dataset(nc_path, decode_times=False)

            # Decode target months (units: "months since 1960-01-01")
            ref    = pd.Timestamp("1960-01-01")
            t_dates = [ref + pd.DateOffset(months=int(m))
                       for m in ds["target"].values]

            lat = ds["lat"].values   # -90 to 90
            lon = ds["lon"].values   # 0 to 359

            # SA bounding box in 0-360 convention: lon 270-330
            lat_idx = np.where((lat >= -60) & (lat <= 15))[0]
            lon_idx = np.where((lon >= 270) & (lon <= 330))[0]
            lat_sa  = lat[lat_idx]
            lon_sa  = lon[lon_idx]

            slices = []
            for i, tdate in enumerate(t_dates):
                if i == 0:          # lead 0 = init month itself (NaN in prate)
                    continue
                pa = ds["prob_above"].values[i][np.ix_(lat_idx, lon_idx)]
                pn = ds["prob_norm" ].values[i][np.ix_(lat_idx, lon_idx)]
                pb = ds["prob_below"].values[i][np.ix_(lat_idx, lon_idx)]

                la_grid, lo_grid = np.meshgrid(lat_sa, lon_sa, indexing="ij")
                tmp = pd.DataFrame({
                    "lat":        la_grid.flatten().round(1),
                    "lon":        lo_grid.flatten().round(1),
                    "prob_above": pa.flatten().round(4),
                    "prob_norm":  pn.flatten().round(4),
                    "prob_below": pb.flatten().round(4),
                })
                tmp["lead_month"]    = i
                tmp["forecast_date"] = tdate.strftime("%Y-%m")
                slices.append(tmp)

            df = pd.concat(slices, ignore_index=True)
            # Convert lon 0-360 → -180/180
            df["lon"] = df["lon"].apply(lambda x: x - 360 if x > 180 else x)
            df = df.dropna(subset=["prob_above"])
            df["init_year"]  = int(init_ym[:4])
            df["init_month"] = int(init_ym[4:])

            df.to_csv(out_csv, index=False)
            print(f"{out_stem}: {len(df)} rows ({df['forecast_date'].nunique()} leads) -> {out_csv}")

        except Exception as e:
            print(f"ERROR processing {nc_path}: {e}")


if __name__ == "__main__":
    fetch_seas5_nino34()
    fetch_seas5_sa_maps()
    fetch_nmme_probs()
