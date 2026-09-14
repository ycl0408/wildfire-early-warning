#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Norther Sierra grid file

Outputs:
  - data/interim/grid_500m_northern_sierra.parquet
  - data/interim/grid_500m_northern_sierra.geojson
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union

OUT_DIR = "data/interim"
AOI_NAME = "northern_sierra"
BUFFER_METERS = 10_000  # 10 km
GRID_SIZE_M = 500       # 500 m
CRS_LONLAT = "EPSG:4326"
CRS_PROJECTED = "EPSG:3310"  # California Albers, meters

TOWNS = {
    "Susanville":            (40.4163, -120.6530),
    "South Lake Tahoe":      (38.9399, -119.9772),
    "Carson Pass":           (38.6971, -120.0019),
    "Sonora":                (37.9841, -120.3821),
    "Chico":                 (39.7285, -121.8375),
    "Paradise":              (39.7596, -121.6219),
}

os.makedirs(OUT_DIR, exist_ok=True)

north_lat = TOWNS["Susanville"][0]
south_lat = TOWNS["Sonora"][0]
west_lon = min(TOWNS["Chico"][1], TOWNS["Paradise"][1])
east_lon = max(TOWNS["South Lake Tahoe"][1], TOWNS["Carson Pass"][1])

aoi_rect = box(west_lon, south_lat, east_lon, north_lat)
aoi_gdf = gpd.GeoDataFrame(pd.DataFrame({"name": [AOI_NAME]}), geometry=[aoi_rect], crs=CRS_LONLAT)

aoi_proj = aoi_gdf.to_crs(CRS_PROJECTED)
aoi_proj["geometry"] = aoi_proj.buffer(BUFFER_METERS)
aoi_buffered = aoi_proj.to_crs(CRS_LONLAT)

aoi_path = os.path.join(OUT_DIR, f"aoi_{AOI_NAME}.geojson")
aoi_buffered.to_file(aoi_path, driver="GeoJSON")
print(f"[OK] Saved AOI to {aoi_path}")

aoi_m = aoi_buffered.to_crs(CRS_PROJECTED)
minx, miny, maxx, maxy = aoi_m.total_bounds

xs = np.arange(minx, maxx, GRID_SIZE_M)
ys = np.arange(miny, maxy, GRID_SIZE_M)

aoi_poly = aoi_m.geometry.iloc[0]

cells = []
append = cells.append
for x in xs:
    stripe = box(x, miny, x + GRID_SIZE_M, maxy)
    if not stripe.intersects(aoi_poly):
        continue
    for y in ys:
        cell = box(x, y, x + GRID_SIZE_M, y + GRID_SIZE_M)
        if cell.intersects(aoi_poly):
            append(cell)

grid = gpd.GeoDataFrame(geometry=cells, crs=CRS_PROJECTED)


grid["cell_id"] = range(len(grid))
grid_ll = grid.to_crs(CRS_LONLAT)
grid_ll["centroid"] = grid_ll.centroid
grid_ll["centroid_lon"] = grid_ll["centroid"].x
grid_ll["centroid_lat"] = grid_ll["centroid"].y
grid_ll = grid_ll.drop(columns=["centroid"])

grid_parquet = os.path.join(OUT_DIR, f"grid_500m_{AOI_NAME}.parquet")
grid_geojson = os.path.join(OUT_DIR, f"grid_500m_{AOI_NAME}.geojson")

grid_ll.to_parquet(grid_parquet, engine="pyarrow", index=False)
grid_ll.to_file(grid_geojson, driver="GeoJSON")
print(f"[OK] Saved grid to {grid_parquet} and {grid_geojson}")


area_km2 = aoi_m.area.iloc[0] / 1_000_000
n_cells = len(grid_ll)
print(f"[SUMMARY] AOI area ≈ {area_km2:,.1f} km²;  Grid cells: {n_cells:,} (500 m)")
