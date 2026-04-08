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
    average over all lead months and any time/ensemble dims,
    and return a flat DataFrame with columns: lat, lon, anom.
    Longitudes are normalised to -180/180.
    """
    ds  = xr.open_dataset(nc_path)
    var = list(ds.data_vars)[0]
    da  = ds[var]

    dims    = set(da.dims)
    lat_dim = next(d for d in dims if d in ("latitude", "lat"))
    lon_dim = next(d for d in dims if d in ("longitude", "lon"))

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

    # Average over all dims that are not lat/lon (leads, time, member …)
    extra = [d for d in lon_sel.dims if d not in (lat_dim, lon_dim)]
    spatial = lon_sel.mean(dim=extra) if extra else lon_sel

    df = (
        spatial
        .to_dataframe(name="anom")
        .reset_index()
        .rename(columns={lat_dim: "lat", lon_dim: "lon"})
    )
    df["anom"] = df["anom"].round(4)

    # Normalise longitude to -180/180
    if df["lon"].max() > 180:
        df["lon"] = df["lon"].apply(lambda x: x - 360 if x > 180 else x)

    df = df[["lat", "lon", "anom"]].dropna(subset=["anom"])
    df["init_year"]  = init_year
    df["init_month"] = init_month
    return df


def fetch_seas5_sa_maps():
    """
    Download SEAS5 ensemble-mean 2m temperature anomaly and
    total precipitation anomaly rate for South America (leads 1-3).
    Saves:
      data/forecasts/seas5_t2m_anom_SA.csv
      data/forecasts/seas5_prcp_anom_SA.csv
    """
    now   = datetime.utcnow()
    year  = str(now.year)
    month = f"{now.month:02d}"

    VARS = [
        ("t2m",  "2m_temperature_anomaly",          "seas5_t2m_anom_SA"),
        ("prcp", "total_precipitation_anomaly_rate", "seas5_prcp_anom_SA"),
    ]

    c = get_cds_client()

    for tag, cds_var, out_stem in VARS:
        nc_path = DATA_DIR / f"seas5_{tag}_SA_{year}{month}.nc"
        out_csv = DATA_DIR / f"{out_stem}.csv"

        if not nc_path.exists():
            print(f"Downloading SEAS5 {cds_var} {year}-{month} …")
            try:
                c.retrieve(
                    "seasonal-postprocessed-single-levels",
                    {
                        "originating_centre": "ecmwf",
                        "system":             "51",
                        "variable":           cds_var,
                        "product_type":       "ensemble_mean",
                        "year":               year,
                        "month":              month,
                        "leadtime_month":     ["1", "2", "3"],
                        "format":             "netcdf",
                    },
                    str(nc_path),
                )
                print(f"Downloaded -> {nc_path}  ({nc_path.stat().st_size/1024:.0f} kB)")
            except Exception as e:
                print(f"ERROR downloading {cds_var}: {e}")
                continue
        else:
            print(f"Using cached {nc_path}")

        try:
            df = _extract_sa_grid(nc_path, int(year), int(month))
            df.to_csv(out_csv, index=False)
            print(f"Saved {out_stem}: {len(df)} grid points -> {out_csv}")
        except Exception as e:
            print(f"ERROR processing {nc_path}: {e}")


if __name__ == "__main__":
    fetch_seas5_nino34()
    fetch_seas5_sa_maps()
