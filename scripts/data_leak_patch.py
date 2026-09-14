#!/usr/bin/env python3
"""
Simple data leak patch (floor snap with cap, drop unmatched).

What it does
------------
- For rows where comp_date > date OR comp_date is NaT, replace comp_date with the nearest
  composite date in the PAST for the same cell_id, based on the NDVI/EVI file.
- Enforce a max lookback (16, 32, or 64 days). If no candidate within the cap -> DROP the row.
- Update only: comp_date, ndvi, evi. All other columns remain unchanged.

Inputs
------
--obs      : modeling table (.parquet or .csv) with columns:
             cell_id, date, comp_date, ndvi, evi, ... (others untouched)
--veg      : NDVI/EVI file (.parquet or .csv) with columns:
             cell_id, date, ndvi, evi   (veg.date is treated as composite date)
--lookback : one of {16, 32, 64} days
--out      : output file (.parquet or .csv)

Usage
-----
python scripts/data_leak_patch.py \
  --obs data/processed/modeling_table_with_veg.parquet \
  --veg data/interim/modis_data.parquet \
  --lookback 16 \
  --out data/processed/modeling_table_patched.parquet
"""
import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Floor comp_date to nearest past composite per cell_id (with cap).")
    ap.add_argument("--obs", required=True, help="Observations file (.parquet or .csv) with cell_id, date, comp_date, ndvi, evi.")
    ap.add_argument("--veg", required=True, help="NDVI/EVI file (.parquet or .csv) with cell_id, date, ndvi, evi.")
    ap.add_argument("--lookback", required=True, type=int, choices=[16, 32, 64], help="Max days to look back (16, 32, or 64).")
    ap.add_argument("--out", required=True, help="Output file (.parquet or .csv).")
    return ap.parse_args()


def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)


def to_datetime_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def build_availability_and_values(veg: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], pd.DataFrame]:
    """
    From VEG (cell_id, date, ndvi, evi):
      - Build availability arrays per cell_id (sorted datetime64[ns]).
      - Build a lookup frame indexed by (cell_id, comp_date) -> [ndvi, evi].
    """
    v = veg.copy()
    v["cell_id"] = pd.to_numeric(v["cell_id"], errors="raise").astype("int64")
    v["date"] = to_datetime_naive(v["date"])
    v = v[v["date"].notna()]

    # Availability = either NDVI or EVI is present
    has_veg = v["ndvi"].notna() | v["evi"].notna()
    avail = (
        v.loc[has_veg, ["cell_id", "date", "ndvi", "evi"]]
         .rename(columns={"date": "comp_date"})
         .drop_duplicates(subset=["cell_id", "comp_date"])
         .sort_values(["cell_id", "comp_date"], kind="mergesort")
         .reset_index(drop=True)
    )

    arrays: Dict[int, np.ndarray] = {
        cid: g["comp_date"].to_numpy(dtype="datetime64[ns]")
        for cid, g in avail.groupby("cell_id", sort=False)
    }
    lookup = avail.set_index(["cell_id", "comp_date"])[["ndvi", "evi"]]
    return arrays, lookup


def main():
    args = parse_args()

    # Load inputs
    df = read_any(args.obs)
    veg = read_any(args.veg)

    # Basic checks / dtype normalization
    required_obs = {"cell_id", "date", "comp_date", "ndvi", "evi"}
    missing = required_obs - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: --obs is missing required columns: {sorted(missing)}")

    df["cell_id"] = pd.to_numeric(df["cell_id"], errors="raise").astype("int64")
    df["date"] = to_datetime_naive(df["date"])
    df["comp_date"] = to_datetime_naive(df["comp_date"])

    arrays, lookup = build_availability_and_values(veg)
    lookback = int(args.lookback)

    n_orig = len(df)

    # Identify rows to patch: comp_date > date OR comp_date is NaT
    patch_mask = df["comp_date"].isna() | (df["comp_date"] > df["date"])
    idx = np.where(patch_mask.values)[0]

    patched = df.copy()
    patched_count = 0
    dropped_due_to_cap = 0

    # Group patch indices by cell for vectorized floor
    idx_by_cell: Dict[int, np.ndarray] = {}
    for i in idx:
        cid = int(patched.at[i, "cell_id"])
        idx_by_cell.setdefault(cid, []).append(i)

    for cid, indices in idx_by_cell.items():
        indices = np.asarray(indices, dtype=int)
        dates = patched.loc[indices, "date"].to_numpy(dtype="datetime64[ns]")

        comp_arr = arrays.get(cid)
        if comp_arr is None or comp_arr.size == 0:
            # No availability for this cell → drop these rows later
            dropped_due_to_cap += len(indices)
            # Mark with sentinel NaT so we can drop in one shot
            patched.loc[indices, "comp_date"] = pd.NaT
            patched.loc[indices, ["ndvi", "evi"]] = np.nan
            continue

        # Floor each date to greatest comp_date <= date
        insert_pos = comp_arr.searchsorted(dates, side="right") - 1
        has_past = insert_pos >= 0

        new_comp = np.full(dates.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        new_comp[has_past] = comp_arr[insert_pos[has_past]]

        # Enforce lookback cap
        deltas = (dates - new_comp).astype("timedelta64[D]").astype("float")
        within_cap = ~np.isnat(new_comp) & (deltas <= lookback)

        # Apply updates for those within the cap
        ok_idx = indices[within_cap]
        if ok_idx.size:
            # set comp_date
            patched.loc[ok_idx, "comp_date"] = new_comp[within_cap]
            # set ndvi/evi from lookup
            key = pd.MultiIndex.from_arrays(
                [np.full(ok_idx.size, cid, dtype="int64"),
                 patched.loc[ok_idx, "comp_date"].to_numpy(dtype="datetime64[ns]")],
                names=["cell_id", "comp_date"]
            )
            vals = lookup.reindex(key)
            patched.loc[ok_idx, "ndvi"] = vals["ndvi"].to_numpy()
            patched.loc[ok_idx, "evi"]  = vals["evi"].to_numpy()
            patched_count += ok_idx.size

        # Mark the rest (no past or beyond cap) to drop
        bad_idx = indices[~within_cap]
        if bad_idx.size:
            patched.loc[bad_idx, "comp_date"] = pd.NaT
            patched.loc[bad_idx, ["ndvi", "evi"]] = np.nan
            dropped_due_to_cap += bad_idx.size

    # Drop unmatched rows (no valid comp within cap)
    before_drop = len(patched)
    keep_mask = patched["comp_date"].notna()
    patched = patched.loc[keep_mask].copy()
    n_kept = len(patched)
    n_dropped = n_orig - n_kept

    # Safety: no future-peek remains
    if not (patched["comp_date"] <= patched["date"]).all():
        raise AssertionError("Future-peek detected after patch (should not happen).")

    # Report
    kept_pct = 100.0 * n_kept / max(1, n_orig)
    print(f"[simple_patch] Original rows:       {n_orig}")
    print(f"[simple_patch] Rows needing patch:  {len(idx)}")
    print(f"[simple_patch] Patched within cap:  {patched_count}")
    print(f"[simple_patch] Dropped (no past or > {lookback}d): {n_dropped}")
    print(f"[simple_patch] ✅ {kept_pct:.2f}% of data points kept")

    # Save
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    (patched.to_parquet(out, index=False) if out.lower().endswith(".parquet") else patched.to_csv(out, index=False))
    print(f"[simple_patch] Wrote: {out}  rows={n_kept}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
