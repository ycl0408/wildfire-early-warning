#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

MODEL_TABLE = "data/processed/modeling_table.parquet"
MODIS_DATA  = "data/interim/modis_data.parquet"
OUT_PATH    = "data/processed/modeling_table_with_veg.parquet"
GAPS_TO_TRY = [8, 16, 24]

def as_dt(s): return pd.to_datetime(s, errors="coerce")

def coverage(df):
    cov_ndvi = df["ndvi"].notna().mean() if "ndvi" in df else 0.0
    cov_evi  = df["evi"].notna().mean()  if "evi"  in df else 0.0
    return cov_ndvi, cov_evi

def main():
    base = pd.read_parquet(MODEL_TABLE)
    mod  = pd.read_parquet(MODIS_DATA)

    base["date"] = as_dt(base["date"])
    mod["date"]  = as_dt(mod["date"])
    if "cell_id" in base and "cell_id" in mod:
        base["cell_id"] = base["cell_id"].astype(np.int64, errors="ignore")
        mod["cell_id"]  = mod["cell_id"].astype(np.int64, errors="ignore")

    mod = mod[["cell_id","date","ndvi","evi"]].copy()

    print("Ranges:")
    print("  modeling_table:", base["date"].min(), "→", base["date"].max(), f"(rows={len(base):,})")
    print("  modis_data:    ", mod["date"].min(),  "→", mod["date"].max(),  f"(rows={len(mod):,})")

    candidates = []

    mergedA = base.merge(mod, on=["cell_id","date"], how="left")
    have_comp = mergedA[["ndvi","evi"]].notna().any(axis=1)
    mergedA["comp_date"] = mergedA["date"].where(have_comp)
    covA = coverage(mergedA)
    print(f"Strategy A (exact) → NDVI {covA[0]:.1%} | EVI {covA[1]:.1%}")
    candidates.append(("Exact", mergedA, covA))

    model_dates = pd.DataFrame({"date": base["date"].dropna().drop_duplicates().sort_values().values})
    comp_dates  = pd.DataFrame({"comp_date": mod["date"].dropna().drop_duplicates().sort_values().values})

    for gap in GAPS_TO_TRY:
        matched = pd.merge_asof(
            model_dates, comp_dates,
            left_on="date", right_on="comp_date",
            direction="nearest"
        )
        matched["gap_days"] = (matched["date"] - matched["comp_date"]).abs().dt.days
        matched.loc[matched["gap_days"] > gap, "comp_date"] = pd.NaT
        date_map = dict(zip(matched["date"], matched["comp_date"]))

        cand = base.copy()
        cand["comp_date"] = cand["date"].map(date_map)

        modB = mod.rename(columns={"date":"comp_date"})
        cand = cand.merge(modB, on=["cell_id","comp_date"], how="left")

        covB = coverage(cand)
        print(f"Strategy B (±{gap}d) → NDVI {covB[0]:.1%} | EVI {covB[1]:.1%}")
        candidates.append((f"Nearest±{gap}d", cand, covB))

    def score(cov): return (cov[0] + cov[1])
    best_name, best_df, best_cov = max(candidates, key=lambda x: score(x[2]))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    best_df.to_parquet(OUT_PATH, index=False)
    print(f"\nChosen: {best_name} → NDVI {best_cov[0]:.1%} | EVI {best_cov[1]:.1%}")
    print(f"Done → {OUT_PATH}")
    have = best_df[best_df["ndvi"].notna() | best_df["evi"].notna()]
    if not have.empty:
        print("\nSample rows with vegetation indices:")
        print(have[["cell_id","date","comp_date","ndvi","evi"]].head(5).to_string(index=False))
    else:
        print("\nCheck base & mod cell_id types match and dates map into composites.")
        try:
            print("Example nearest mapping (first 8):")
            print(matched.head(8).to_string(index=False))
        except Exception:
            pass

if __name__ == "__main__":
    main()
