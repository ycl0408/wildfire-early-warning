#!/usr/bin/env python3
"""
Generate a lat/lon centroid grid for a rectangular AOI at ~N-meter spacing.

Defaults:
- California bbox (lon_min, lat_min, lon_max, lat_max) = (-124.48, 32.53, -114.13, 42.01)
- Resolution = 500 meters
Outputs:
- data/interim/grid_cells.csv
- data/interim/grid_cells.parquet  (smaller, preferred)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def make_grid(lon_min, lat_min, lon_max, lat_max, resolution_m=500):
    # meters per degree latitude ~ constant
    deg_per_meter_lat = 1.0 / 111320.0
    dlat = resolution_m * deg_per_meter_lat

    # adjust longitude step by cos(latitude); use mid-latitude for the AOI
    lat0 = 0.5 * (lat_min + lat_max)
    dlon = dlat / np.cos(np.deg2rad(lat0))

    # build 1D arrays then meshgrid of centers
    lats = np.arange(lat_min, lat_max, dlat)
    lons = np.arange(lon_min, lon_max, dlon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    n = lat_grid.size
    df = pd.DataFrame({
        "cell_id": np.arange(n, dtype=np.int64),
        "lat": lat_grid.ravel(),
        "lon": lon_grid.ravel()
    })
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bbox", nargs=4, type=float,
                   default=[-124.48, 32.53, -114.13, 42.01],
                   help="lon_min lat_min lon_max lat_max (default: California)")
    p.add_argument("--resolution_m", type=float, default=500.0,
                   help="approximately this many meters between grid centers")
    p.add_argument("--out_csv", type=str, default="data/interim/grid_cells.csv",
                   help="output CSV path")
    p.add_argument("--out_parquet", type=str, default="data/interim/grid_cells.parquet",
                   help="output Parquet path")
    args = p.parse_args()

    lon_min, lat_min, lon_max, lat_max = args.bbox
    Path("data/interim").mkdir(parents=True, exist_ok=True)

    print(f"Making grid for bbox={args.bbox}, resolution≈{args.resolution_m} m …")
    df = make_grid(lon_min, lat_min, lon_max, lat_max, args.resolution_m)
    print(f"Grid cells: {len(df):,}")

    # Save Parquet (preferred) and CSV (for visibility)
    try:
        df.to_parquet(args.out_parquet, index=False)
        print(f"Wrote {args.out_parquet}")
    except Exception as e:
        print(f"(Parquet save skipped: {e})")

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

if __name__ == "__main__":
    main()
