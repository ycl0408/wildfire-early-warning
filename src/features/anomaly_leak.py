import os
import numpy as np
import pandas as pd

IN_TABLE   = "data/processed/modeling_table_with_veg_feats.parquet"
MODIS_FILE = "data/interim/modis_data.parquet"
OUT_TABLE  = "data/processed/modeling_table_final2.parquet"

STD_FLOOR  = 0.05
Z_CLIP     = 5.0

def build_causal_baseline(modis_path: str) -> pd.DataFrame:
    """
    For each cell_id and month, compute expanding (past-only) mean/std
    for NDVI/EVI at every MODIS acquisition date. Stats at date t use data ≤ t-1.
    Returns a table keyed by (cell_id, date) with ndvi_mu_past, ndvi_sd_past, evi_mu_past, evi_sd_past.
    """
    mod = pd.read_parquet(modis_path, columns=["cell_id", "date", "ndvi", "evi"]).copy()
    mod["date"] = pd.to_datetime(mod["date"], errors="coerce")
    mod["month"] = mod["date"].dt.month.astype("int8")

    mod = mod.sort_values(["cell_id", "month", "date"])

    def _expanding_past(g: pd.DataFrame) -> pd.DataFrame:
        # Shift by 1 so stats at row t only see values up to t-1
        ndvi_shift = g["ndvi"].astype("float32").shift(1)
        evi_shift  = g["evi"].astype("float32").shift(1)

        g["ndvi_mu_past"] = ndvi_shift.expanding().mean().astype("float32")
        g["ndvi_sd_past"] = ndvi_shift.expanding().std(ddof=1).astype("float32")
        g["evi_mu_past"]  = evi_shift.expanding().mean().astype("float32")
        g["evi_sd_past"]  = evi_shift.expanding().std(ddof=1).astype("float32")
        return g

    mod = mod.groupby(["cell_id", "month"], observed=True, sort=False, group_keys=False).apply(_expanding_past)

    # Stabilize std
    for c in ["ndvi_sd_past", "evi_sd_past"]:
        mod[c] = mod[c].fillna(STD_FLOOR).clip(lower=1e-3)
    # Ensure means are float32 and fill leading NaNs (no past) with NaN to later produce NaN anomalies
    for c in ["ndvi_mu_past", "evi_mu_past"]:
        mod[c] = mod[c].astype("float32")

    # Keep only keys we need for merge
    base = mod[["cell_id", "date", "ndvi_mu_past", "ndvi_sd_past", "evi_mu_past", "evi_sd_past"]].copy()
    return base

def apply_anom_causal(df: pd.DataFrame, base: pd.DataFrame, value_col: str, date_col: str, out_col: str) -> pd.Series:
    """
    Compute causal anomaly for value_col using the MODIS date in date_col.
    Joins per (cell_id, comp_date) to past-only stats (no leakage even within training years).
    """
    if value_col not in df.columns or date_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")

    tmp = df[["cell_id", value_col, date_col]].rename(columns={date_col: "_comp_date"}).copy()
    tmp["_comp_date"] = pd.to_datetime(tmp["_comp_date"], errors="coerce")

    cols = ["cell_id", "date"]
    if value_col.startswith("ndvi"):
        mu_col, sd_col = "ndvi_mu_past", "ndvi_sd_past"
    else:
        mu_col, sd_col = "evi_mu_past", "evi_sd_past"

    joined = tmp.merge(base[["cell_id", "date", mu_col, sd_col]],
                       how="left", left_on=["cell_id", "_comp_date"], right_on=cols)

    z = (joined[value_col].astype("float32") - joined[mu_col]) / joined[sd_col]
    z = z.clip(-Z_CLIP, Z_CLIP).astype("float32")
    return z.reindex(df.index)

def main():
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)

    # 1) Build causal baseline once
    base = build_causal_baseline(MODIS_FILE)

    # 2) Load your modeling table
    df = pd.read_parquet(IN_TABLE)

    need_dates = ["comp_date", "comp_date_lag_2w", "comp_date_lag_4w", "comp_date_lag_6w"]
    for c in need_dates:
        if c not in df.columns:
            raise RuntimeError(f"Missing '{c}' in {IN_TABLE} — rebuild veg features first.")

    # 3) Compute causal anomalies (current)
    if "ndvi" in df.columns:
        df["ndvi_anom"] = apply_anom_causal(df, base, "ndvi", "comp_date", "ndvi_anom")
    if "evi" in df.columns:
        df["evi_anom"]  = apply_anom_causal(df, base, "evi",  "comp_date", "evi_anom")

    # 4) Causal anomalies for lags (use each lag's comp_date)
    for tag in ["2w", "4w", "6w"]:
        nd_lag, ev_lag = f"ndvi_lag_{tag}", f"evi_lag_{tag}"
        cd_lag = f"comp_date_lag_{tag}"
        if nd_lag in df.columns:
            df[f"ndvi_anom_lag_{tag}"] = apply_anom_causal(df, base, nd_lag, cd_lag, f"ndvi_anom_lag_{tag}")
        if ev_lag in df.columns:
            df[f"evi_anom_lag_{tag}"]  = apply_anom_causal(df, base, ev_lag,  cd_lag, f"evi_anom_lag_{tag}")

    # 5) Save & quick coverage
    anom_cols = [c for c in df.columns if c.startswith(("ndvi_anom", "evi_anom"))]
    cov = {c: f"{df[c].notna().mean():.1%}" for c in anom_cols}
    print("Coverage (causal anomalies):", cov)

    df.to_parquet(OUT_TABLE, index=False)
    print(f"Saved → {OUT_TABLE}")

if __name__ == "__main__":
    main()
