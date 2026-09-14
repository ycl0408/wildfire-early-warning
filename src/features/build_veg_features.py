#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

BASE_PATH = "data/processed/modeling_table_patched.parquet"
MODIS_PATH = "data/interim/modis_data.parquet"
OUT_PATH = "data/processed/modeling_table_with_veg_feats.parquet"

NDVI_EVI_MIN, NDVI_EVI_MAX = -1.2, 1.2
Z_CLIP = 5.0
MAX_GAP_DAYS = 8
LAG_SPECS = {"2w": 14, "4w": 28, "6w": 42}

def as_dt(s): return pd.to_datetime(s, errors="coerce")

def load_inputs():
    base = pd.read_parquet(BASE_PATH)
    mod  = pd.read_parquet(MODIS_PATH)

    base["date"] = as_dt(base["date"])
    mod["date"]  = as_dt(mod["date"])

    if "cell_id" in base: base["cell_id"] = base["cell_id"].astype("int64", errors="ignore")
    if "cell_id" in mod:  mod["cell_id"]  = mod["cell_id"].astype("int64", errors="ignore")

    for col in ["ndvi","evi"]:
        if col in base:
            base.loc[(base[col] < NDVI_EVI_MIN) | (base[col] > NDVI_EVI_MAX), col] = np.nan
        if col in mod:
            mod.loc[(mod[col] < NDVI_EVI_MIN) | (mod[col] > NDVI_EVI_MAX), col] = np.nan

    mod = (mod
           .dropna(subset=["cell_id","date"])
           .sort_values(["cell_id","date"])
           .drop_duplicates(subset=["cell_id","date"], keep="last"))

    return base, mod

def map_past_only(model_dates, comp_dates, max_gap_days=8):
    model_df = pd.DataFrame({"date": pd.to_datetime(model_dates.dropna().unique())}).sort_values("date")
    comp_df  = pd.DataFrame({"comp_date": pd.to_datetime(comp_dates.dropna().unique())}).sort_values("comp_date")
    matched = pd.merge_asof(
        model_df, comp_df,
        left_on="date", right_on="comp_date",
        direction="backward",
        tolerance=pd.Timedelta(days=max_gap_days)
    )
    return dict(zip(matched["date"], matched["comp_date"]))

def build_monthly_baseline(mod):
    mod["month"] = mod["date"].dt.month
    agg = (mod
           .groupby(["cell_id","month"], as_index=False)
           .agg(ndvi_mean=("ndvi","mean"),
                ndvi_std =("ndvi","std"),
                evi_mean =("evi","mean"),
                evi_std  =("evi","std"),
                n=("ndvi","size")))

    cell_std = (mod
                .groupby("cell_id", as_index=False)
                .agg(ndvi_cell_std=("ndvi","std"),
                     evi_cell_std =("evi","std")))

    agg = agg.merge(cell_std, on="cell_id", how="left")
    agg["ndvi_std"] = np.where(agg["ndvi_std"].isna() | (agg["ndvi_std"] < 1e-6),
                               agg["ndvi_cell_std"], agg["ndvi_std"])
    agg["evi_std"]  = np.where(agg["evi_std"].isna()  | (agg["evi_std"]  < 1e-6),
                               agg["evi_cell_std"],  agg["evi_std"])
    agg["ndvi_std"] = agg["ndvi_std"].fillna(0.05).clip(lower=1e-3)
    agg["evi_std"]  = agg["evi_std"].fillna(0.05).clip(lower=1e-3)
    agg = agg.drop(columns=["ndvi_cell_std","evi_cell_std"])
    return agg

def add_anomalies(base, baseline, month_col):
    out = base.merge(baseline, left_on=["cell_id", month_col], right_on=["cell_id","month"], how="left")
    out.drop(columns=["month"], inplace=True, errors="ignore")
    out["ndvi_anom"] = (out["ndvi"] - out["ndvi_mean"]) / out["ndvi_std"]
    out["evi_anom"]  = (out["evi"]  - out["evi_mean"])  / out["evi_std"]
    for col in ["ndvi_anom","evi_anom"]:
        out[col] = out[col].replace([np.inf,-np.inf], np.nan).clip(-Z_CLIP, Z_CLIP)
    return out

def add_composite_and_lags(base, mod, max_gap_days=8):
    out = base.copy()

    if "comp_date" in out.columns:
        out["comp_date"] = pd.to_datetime(out["comp_date"], errors="coerce")
        out.loc[out["comp_date"] > out["date"], "comp_date"] = pd.NaT
        comp_map = map_past_only(out["date"], mod["date"], max_gap_days=max_gap_days)
        out["comp_date"] = out["comp_date"].fillna(out["date"].map(comp_map))
    else:
        comp_map = map_past_only(out["date"], mod["date"], max_gap_days=max_gap_days)
        out["comp_date"] = out["date"].map(comp_map)

    mod_comp = mod.rename(columns={"date": "comp_date"})
    cur_vals = mod_comp[["cell_id", "comp_date", "ndvi", "evi"]].rename(
        columns={"ndvi": "ndvi_cur", "evi": "evi_cur"}
    )
    out = out.merge(cur_vals, on=["cell_id", "comp_date"], how="left")
    out["ndvi"] = out["ndvi"].where(out["ndvi"].notna(), out["ndvi_cur"])
    out["evi"]  = out["evi"].where(out["evi"].notna(),  out["evi_cur"])
    out.drop(columns=["ndvi_cur", "evi_cur"], inplace=True)

    assert (out["comp_date"] <= out["date"]).all(), "Found comp_date in the future."

    comp_src = (out[["comp_date"]].dropna().drop_duplicates()
                .sort_values("comp_date").rename(columns={"comp_date": "comp_src"}))
    comp_all = pd.DataFrame({"comp_all": sorted(mod_comp["comp_date"].dropna().unique())})

    for tag, days in LAG_SPECS.items():
        lag_targets = comp_src.copy()
        lag_targets[f"lag_target_{tag}"] = lag_targets["comp_src"] - pd.to_timedelta(days, unit="D")
        match = pd.merge_asof(
            lag_targets.sort_values(f"lag_target_{tag}"),
            comp_all.rename(columns={"comp_all": f"comp_match_{tag}"}).sort_values(f"comp_match_{tag}"),
            left_on=f"lag_target_{tag}",
            right_on=f"comp_match_{tag}",
            direction="backward",
            tolerance=pd.Timedelta(days=max_gap_days)
        )
        lag_map = dict(zip(match["comp_src"], match[f"comp_match_{tag}"]))
        out[f"comp_date_lag_{tag}"] = out["comp_date"].map(lag_map)

        lag_vals = mod_comp.rename(columns={
            "ndvi": f"ndvi_lag_{tag}",
            "evi":  f"evi_lag_{tag}",
            "comp_date": f"comp_match_key_{tag}",
        })
        out = out.merge(
            lag_vals[["cell_id", f"comp_match_key_{tag}", f"ndvi_lag_{tag}", f"evi_lag_{tag}"]],
            left_on=["cell_id", f"comp_date_lag_{tag}"],
            right_on=["cell_id", f"comp_match_key_{tag}"],
            how="left"
        ).drop(columns=[f"comp_match_key_{tag}"])

        assert (out[f"comp_date_lag_{tag}"].isna() | (out[f"comp_date_lag_{tag}"] < out["comp_date"])).all(), f"{tag} lag not earlier."

    return out

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    base, mod = load_inputs()

    print("Ranges:")
    print("  modeling_table:", base["date"].min(), "→", base["date"].max(), f"(rows={len(base):,})")
    print("  modis_data:    ", mod["date"].min(),  "→", mod["date"].max(),  f"(rows={len(mod):,})")

    df = add_composite_and_lags(base, mod, max_gap_days=MAX_GAP_DAYS)

    monthly_baseline = build_monthly_baseline(mod)

    df["month_comp"] = df["comp_date"].dt.month
    df = add_anomalies(df, monthly_baseline, month_col="month_comp")

    for tag in LAG_SPECS.keys():
        mcol = f"month_lag_{tag}"
        df[mcol] = df[f"comp_date_lag_{tag}"].dt.month
        for vcol in [f"ndvi_lag_{tag}", f"evi_lag_{tag}"]:
            tmp = df[["cell_id", mcol, vcol]].merge(
                monthly_baseline, left_on=["cell_id", mcol], right_on=["cell_id","month"], how="left"
            )
            if "ndvi" in vcol:
                z = (tmp[vcol] - tmp["ndvi_mean"]) / tmp["ndvi_std"]
                df[f"ndvi_anom_lag_{tag}"] = z.replace([np.inf,-np.inf], np.nan).clip(-Z_CLIP, Z_CLIP)
            else:
                z = (tmp[vcol] - tmp["evi_mean"]) / tmp["evi_std"]
                df[f"evi_anom_lag_{tag}"] = z.replace([np.inf,-np.inf], np.nan).clip(-Z_CLIP, Z_CLIP)

    keep_order = ["cell_id","date","comp_date","ndvi","evi","ndvi_anom","evi_anom"]
    for tag in LAG_SPECS.keys():
        keep_order += [f"comp_date_lag_{tag}", f"ndvi_lag_{tag}", f"evi_lag_{tag}",
                       f"ndvi_anom_lag_{tag}", f"evi_anom_lag_{tag}"]
    keep_order += [c for c in base.columns if c not in keep_order]
    df = df[keep_order]

    cover_cols = ["ndvi","evi","ndvi_anom","evi_anom"] + \
                 [f"ndvi_lag_{t}" for t in LAG_SPECS] + [f"evi_lag_{t}" for t in LAG_SPECS] + \
                 [f"ndvi_anom_lag_{t}" for t in LAG_SPECS] + [f"evi_anom_lag_{t}" for t in LAG_SPECS]
    cov = {c: f"{df[c].notna().mean():.1%}" for c in cover_cols if c in df.columns}

    df.to_parquet(OUT_PATH, index=False)
    print("Coverage:", cov)
    print(f"Saved → {OUT_PATH}")

    show = ["cell_id","date","comp_date","ndvi","evi","ndvi_anom","evi_anom"]
    for tag in LAG_SPECS: show += [f"ndvi_lag_{tag}", f"evi_lag_{tag}", f"ndvi_anom_lag_{tag}", f"evi_anom_lag_{tag}"]
    print(df[show].head(8).to_string(index=False))

if __name__ == "__main__":
    main()
