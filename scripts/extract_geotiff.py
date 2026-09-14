#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converting Geotiff data to CSV

Outputs:
  - data/interim/modis_data.parquet
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio import sample as rsample

MODIS_DIR = Path("data/raw/modis")
GRID_PATH = Path("data/interim/grid_500m_northern_sierra.parquet")
OUT_PATH = Path("data/interim/modis_data.parquet")

SCALE = 0.0001
FILL_THRESHOLD = -2000
QA_ACCEPTED = {0.0, 1.0}

DATE_PATTERNS = [
    re.compile(r".*doy(\d{4})(\d{3})", re.I),
    re.compile(r".*_(\d{4})_(\d{3})\.tif$", re.I),
    re.compile(r".*_(\d{4}-\d{2}-\d{2})\.tif$", re.I),
]

def parse_date(name: str):
    for pat in DATE_PATTERNS:
        m = pat.match(name)
        if m:
            if pat.pattern.startswith(".*doy"):
                y, doy = int(m.group(1)), int(m.group(2))
                return pd.to_datetime(f"{y}-{doy:03d}", format="%Y-%j")
            if len(m.groups()) == 2 and len(m.group(1)) == 4 and len(m.group(2)) == 3:
                y, doy = int(m.group(1)), int(m.group(2))
                return pd.to_datetime(f"{y}-{doy:03d}", format="%Y-%j")
            if len(m.groups()) == 1 and len(m.group(1)) == 10:
                return pd.to_datetime(m.group(1))
    return None

def list_tifs(root: Path, key: str):
    k = key.lower()
    return sorted([p for p in root.rglob("*.tif") if k in p.name.lower()])

def load_grid(grid_path):
    import pandas as pd
    g = pd.read_parquet(grid_path) if str(grid_path).endswith((".parquet",".pq")) else pd.read_csv(grid_path)
    # normalize column names the script expects
    g = g.rename(columns={
        "centroid_lat": "lat",
        "centroid_lon": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
    })
    need = {"cell_id","lat","lon"}
    missing = need - set(g.columns)
    if missing:
        raise ValueError(f"Grid must have columns {need}; found {list(g.columns)}")
    g["cell_id"] = g["cell_id"].astype("int64", errors="ignore")
    g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
    g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
    g = g.dropna(subset=["lat","lon"]).drop_duplicates("cell_id").sort_values("cell_id")
    return g[["cell_id","lat","lon"]]


def sample_band(tif_path: Path, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        pts = list(zip(lons, lats))
        vals = [v[0] for v in rsample.sample_gen(src, pts)]
        return np.array(vals, dtype="float32")

def map_by_date(paths):
    dmap = {}
    for p in paths:
        d = parse_date(p.name)
        if d is None:
            try:
                with rasterio.open(p) as src:
                    tag = src.tags().get("TIFFTAG_DATETIME", "")
                if tag:
                    d = pd.to_datetime(tag.split(" ")[0], errors="coerce")
            except Exception:
                d = None
        if d is not None:
            dmap[pd.to_datetime(d).normalize()] = p
    return dmap

def main():
    grid = load_grid(GRID_PATH)
    lats = grid["lat"].to_numpy()
    lons = grid["lon"].to_numpy()

    ndvi_tifs = list_tifs(MODIS_DIR, "NDVI")
    evi_tifs  = list_tifs(MODIS_DIR, "EVI")
    qa_tifs   = list_tifs(MODIS_DIR, "pixel")  # Pixel_Reliability

    if not ndvi_tifs or not evi_tifs:
        raise FileNotFoundError("NDVI/EVI .tif not found under data/raw/modis")

    ndvi_map = map_by_date(ndvi_tifs)
    evi_map  = map_by_date(evi_tifs)
    qa_map   = map_by_date(qa_tifs) if qa_tifs else {}

    dates = sorted(set(ndvi_map.keys()) & set(evi_map.keys()))
    if not dates:
        raise RuntimeError("Failed to align NDVI and EVI dates from filenames.")

    rows = []
    for d in dates:
        ndvi_raw = sample_band(ndvi_map[d], lats, lons)
        evi_raw  = sample_band(evi_map[d],  lats, lons)

        ndvi = np.where(ndvi_raw <= FILL_THRESHOLD, np.nan, ndvi_raw) * SCALE
        evi  = np.where(evi_raw  <= FILL_THRESHOLD, np.nan, evi_raw)  * SCALE

        if d in qa_map:
            qa = sample_band(qa_map[d], lats, lons).astype("float32")
            ok = np.isin(qa, list(QA_ACCEPTED))
            ndvi = np.where(ok, ndvi, np.nan)
            evi  = np.where(ok, evi,  np.nan)

        df = pd.DataFrame({
            "cell_id": grid["cell_id"],
            "date":    pd.to_datetime(d),
            "ndvi":    ndvi,
            "evi":     evi
        }).dropna(subset=["ndvi","evi"], how="all")
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}  shape={out.shape}")

if __name__ == "__main__":
    main()
