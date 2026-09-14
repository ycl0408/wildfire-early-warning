#!/usr/bin/env python3
import os
import pandas as pd

# === CONFIG (change if your paths/columns differ) ===
IN_PATHS = [
    "data/processed/modeling_table_v1.parquet"
]
COL_ID  = "cell_id"
COL_LAT = "latitude"
COL_LON = "longitude"
OUT_CSV = "data/processed/points_for_gee.csv"

def load_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            if p.endswith(".parquet"):
                return pd.read_parquet(p), p
            elif p.endswith(".csv"):
                return pd.read_csv(p), p
    raise FileNotFoundError(f"None of these files exist:\n  " + "\n  ".join(paths))

def main():
    df, used = load_first_existing(IN_PATHS)
    needed = {COL_ID, COL_LAT, COL_LON}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {used}")

    # Keep only what GEE needs
    pts = df[[COL_ID, COL_LAT, COL_LON]].copy()

    # Deduplicate (prefer unique cell_id; otherwise unique lat/lon)
    if pts[COL_ID].isna().any():
        pts = pts.dropna(subset=[COL_LAT, COL_LON]).drop_duplicates(subset=[COL_LAT, COL_LON])
    else:
        pts = pts.drop_duplicates(subset=[COL_ID]).dropna(subset=[COL_LAT, COL_LON])

    # Sanity: bounds
    bad = (pts[COL_LAT].abs() > 90) | (pts[COL_LON].abs() > 180)
    if bad.any():
        raise ValueError(f"Found {bad.sum()} rows with invalid lat/lon bounds.")

    # Optional: round to 6 decimals to keep file small, but keep full precision if you prefer
    pts[COL_LAT] = pts[COL_LAT].round(6)
    pts[COL_LON] = pts[COL_LON].round(6)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pts.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {len(pts):,} points → {OUT_CSV}")

if __name__ == "__main__":
    main()
