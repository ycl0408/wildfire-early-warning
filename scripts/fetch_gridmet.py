#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch GridMET month-by-month

Outputs:
  - data/interim/weather_gridmet_joined.parquet
  - data/interim/weather_gridmet_features.parquet
"""

from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import pygridmet as gridmet
from shapely.geometry import box

DATA = Path("data")
RAW = DATA / "raw"
INTERIM = DATA / "interim"

GRID_FILE   = INTERIM / "grid_500m_northern_sierra.parquet"
LABELS_FILE = INTERIM / "cell_labels_balanced.parquet"

START = "2016-01-01"
END   = "2024-12-31"
VARS  = ["tmmx","tmmn","pr","vpd","vs","rmin","rmax","srad"]

import os
os.environ.setdefault("HYRIVER_HTTP_TIMEOUT", "60")
os.environ.setdefault("HYRIVER_HTTP_MAX_RETRIES", "2")
os.environ.setdefault("HYRIVER_CACHE_PATH", str(RAW / "_hyriver_cache"))

def ensure_dirs():
    RAW.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

def parse_dates(s: pd.Series) -> pd.Series:
    s = s.copy()
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64").astype(str).str.zfill(8)
        dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce", utc=True)
    else:
        dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce", utc=True)
        mask = dt.isna()
        if mask.any():
            dt2 = pd.to_datetime(s[mask], format="%Y/%m/%d", errors="coerce", utc=True)
            dt.loc[mask] = dt2
            mask = dt.isna()
        if mask.any():
            dt2 = pd.to_datetime(s[mask], errors="coerce", utc=True)
            dt.loc[mask] = dt2
    dt = dt.dt.tz_convert(None)
    return dt.dt.normalize()

def load_grid() -> pd.DataFrame:
    if not GRID_FILE.exists():
        raise FileNotFoundError(f"Missing {GRID_FILE}")
    g = pd.read_parquet(GRID_FILE)
    g = g.rename(columns={"centroid_lat":"latitude","centroid_lon":"longitude"})
    need = ["cell_id","latitude","longitude"]
    missing = [c for c in need if c not in g.columns]
    if missing:
        raise ValueError(f"Grid missing columns: {missing}")
    return g[need].dropna()


def load_labels() -> pd.DataFrame:
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Missing {LABELS_FILE}")
    df = pd.read_parquet(LABELS_FILE)

    date_col = next((c for c in ["date","acq_date","obs_date","day","dt"] if c in df.columns), None)
    if date_col is None:
        raise ValueError("Labels need a date-like column (one of: date, acq_date, obs_date, day, dt)")
    if "cell_id" not in df.columns:
        raise ValueError("Labels need a 'cell_id' column")

    out = df.copy()                          # keep ALL label columns
    out["date"] = parse_dates(out[date_col]) # normalized date
    if date_col != "date":
        out = out.drop(columns=[date_col], errors="ignore")  # avoid duplicate date col

    # Optional: clip to START..END window if defined
    try:
        if START is not None and END is not None:
            out = out[(out["date"] >= pd.Timestamp(START)) & (out["date"] <= pd.Timestamp(END))]
    except NameError:
        pass

    out = out.dropna(subset=["cell_id","date"])
    return out

def infer_bbox(grid: pd.DataFrame, pad=0.2):
    return (
        float(grid["longitude"].min()) - pad,
        float(grid["latitude"].min())  - pad,
        float(grid["longitude"].max()) + pad,
        float(grid["latitude"].max())  + pad,
    )

def unit_conversions(ds: xr.Dataset) -> xr.Dataset:
    if "tmmx" in ds: ds["tmmx_C"] = ds["tmmx"] - 273.15
    if "tmmn" in ds: ds["tmmn_C"] = ds["tmmn"] - 273.15
    if "vpd"  in ds:
        units = (ds["vpd"].attrs.get("units","") or "").lower()
        ds["vpd_kPa"] = ds["vpd"]/1000.0 if units in {"pa","pascal","pascals"} else ds["vpd"]
    return ds

def coord_names(ds: xr.Dataset):
    lat = "lat" if "lat" in ds.coords else ("y" if "y" in ds.coords else None)
    lon = "lon" if "lon" in ds.coords else ("x" if "x" in ds.coords else None)
    tim = "time" if "time" in ds.dims else None
    if not all([lat,lon,tim]): raise ValueError(f"Unexpected coords/dims. Coords={list(ds.coords)}, Dims={list(ds.dims)}")
    return tim, lat, lon

def sample_streaming(bbox, pts: pd.DataFrame) -> pd.DataFrame:
    import xarray as xr

    pts = pts.copy()
    pts["month"] = pts["date"].dt.to_period("M")
    months = sorted(pts["month"].unique())

    outs = []
    for m in months:
        sub = pts[pts["month"] == m].drop(columns="month")
        start = pd.Period(m, "M").start_time.date().isoformat()
        end   = pd.Period(m, "M").end_time.date().isoformat()

        ds = gridmet.get_bygeom(box(*bbox), (start, end), variables=VARS)
        ds = ds.chunk({"time": ds.sizes.get("time", 1)})
        ds = unit_conversions(ds)

        tim, lat, lon = coord_names(ds)
        keep = [v for v in ["tmmx_C","tmmn_C","vpd_kPa","pr","vs","rmin","rmax","srad"] if v in ds]
        if not keep:
            keep = [v for v in ["tmmx","tmmn","vpd","pr","vs","rmin","rmax","srad"] if v in ds]
        if not keep:
            raise ValueError(f"No expected GridMET variables in month {m}: {list(ds.data_vars)}")

        t = xr.DataArray(sub["date"].values,      dims="z")
        la = xr.DataArray(sub["latitude"].values, dims="z")
        lo = xr.DataArray(sub["longitude"].values,dims="z")
        sel = ds[keep].sel({tim: t, lat: la, lon: lo}, method="nearest")

        df = sel.to_dataframe().reset_index(drop=True)
        outs.append(sub.reset_index(drop=True).join(df))
        print(f"[INFO] Sampled {len(sub):,} rows for {m}")

    return pd.concat(outs, ignore_index=True)

def add_rollups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["cell_id","date"])
    def roll(g):
        if "pr" in g: g["pr_14d_sum"] = g["pr"].rolling(14,min_periods=7).sum()
        if "pr" in g: g["pr_28d_sum"] = g["pr"].rolling(28,min_periods=14).sum()
        if "tmmx_C" in g: g["tmmx_14d_mean"] = g["tmmx_C"].rolling(14,min_periods=7).mean()
        if "tmmn_C" in g: g["tmmn_14d_mean"] = g["tmmn_C"].rolling(14,min_periods=7).mean()
        if "vpd_kPa" in g: g["vpd_14d_mean"] = g["vpd_kPa"].rolling(14,min_periods=7).mean()
        if "vs" in g: g["wind_14d_mean"] = g["vs"].rolling(14,min_periods=7).mean()
        return g
    return df.groupby("cell_id", group_keys=False).apply(roll)

# MAIN
def main():
    ensure_dirs()
    grid  = load_grid()
    labels = load_labels()
    pts = labels.merge(grid, on="cell_id", how="left").dropna(subset=["latitude","longitude","date"])
    if pts.empty:
        raise ValueError("No sample points after join. Check cell_id keys / coords.")
    bbox = infer_bbox(grid)
    print(f"[INFO] AOI bbox (W,S,E,N): {bbox}")

    sampled = sample_streaming(bbox, pts)
    out_joined = INTERIM / "weather_gridmet_joined.parquet"
    sampled.to_parquet(out_joined, index=False)
    print(f"[OK] wrote {out_joined} rows={len(sampled):,}")

    feats = add_rollups(sampled)
    out_feats = INTERIM / "weather_gridmet_features.parquet"
    feats.to_parquet(out_feats, index=False)
    print(f"[OK] wrote {out_feats} rows={len(feats):,}")

if __name__ == "__main__":
    main()
