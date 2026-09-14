#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converting FIRMS data to label points

Outputs:
  - data/interim/cell_labels.parquet
"""

import pandas as pd, geopandas as gpd

GRID_PATH = "data/interim/grid_500m_northern_sierra.parquet"
FIRMS_PATH = "data/interim/firms_all_filtered.parquet"
OUT_PATH = "data/interim/cell_labels.parquet"

CONF_THRESH_MODIS = 50    # already applied
# VIIRS nominal/high already applied

def main():
    grid = gpd.read_parquet(GRID_PATH).set_crs(4326, allow_override=True)
    grid = grid.set_index("cell_id")

    df = pd.read_parquet(FIRMS_PATH)
    df = df.rename(columns={"lat":"latitude","lon":"longitude"})
    df["date"] = pd.to_datetime(df["acq_date"]).dt.date

    gdf = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=4326
    )

    # spatial join
    hits = gpd.sjoin(gdf, grid[["geometry"]], how="inner", predicate="intersects")
    # aggregate to cell-day
    agg = (hits.groupby(["cell_id","date"])
                .agg(label_occurrence=("date","size"),
                     label_severity=("frp","max"))
                .reset_index())
    agg["label_occurrence"] = 1
    agg["label_severity"] = agg["label_severity"].fillna(0)

    # (optional) build a full index if you want explicit zeros later
    agg.to_parquet(OUT_PATH, index=False)
    print("[OK] wrote", OUT_PATH, "positives:", agg.shape[0])

if __name__ == "__main__":
    main()
