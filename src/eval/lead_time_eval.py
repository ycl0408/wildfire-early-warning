#!/usr/bin/env python3
"""
Forward lead-time experiment on the final modeling table.

For each existing row (cell, t) build horizon labels from the full FIRMS cell-day table:
  label_{H}d = 1 if any FIRMS detection in the cell within (t, t+H], H in {14, 28, 42}.
Two populations:
  all_rows      : every row in the modeling table (rows sampled on fire days; same-day positives included)
  new_ignition  : rows with NO detection in the cell during [t-30d, t]  -> a fire in (t, t+H] is a new ignition
Features unchanged (vegetation up to t, weather at t). Temporal split as in the benchmark.
Usage: python src/eval/lead_time_eval.py   (reads data/processed/modeling_table_v5.parquet and data/interim/cell_labels.parquet)
Writes outputs/robustness/lead_time.csv
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "models"))
from model_benchmark import make_features, pick_feature_cols

U = str(REPO / "data") + "/"
OUT = REPO / "outputs" / "robustness"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
HORIZONS = (14, 28, 42)
LOOKBACK_CLEAN = 30

df = pd.read_parquet(U + "processed/modeling_table_v5.parquet")
df["date"] = pd.to_datetime(df["date"])
lab = pd.read_parquet(U + "interim/cell_labels.parquet")
lab["date"] = pd.to_datetime(lab["date"])
fires = lab[["cell_id", "date"]].drop_duplicates().sort_values(["cell_id", "date"])
fire_days = {cid: g["date"].values.astype("datetime64[D]") for cid, g in fires.groupby("cell_id")}

def count_between(cid, t, lo_days, hi_days):
    """number of fire days in cell within (t+lo, t+hi]"""
    arr = fire_days.get(cid)
    if arr is None: return 0
    t = np.datetime64(t, "D")
    lo = arr.searchsorted(t + np.timedelta64(lo_days, "D"), side="right")
    hi = arr.searchsorted(t + np.timedelta64(hi_days, "D"), side="right")
    return hi - lo

cells = df["cell_id"].values; dates = df["date"].values.astype("datetime64[D]")
for H in HORIZONS:
    df[f"label_{H}d"] = [int(count_between(c, t, 0, H) > 0) for c, t in zip(cells, dates)]
df["recent_fire"] = [int(count_between(c, t, -LOOKBACK_CLEAN - 1, 0) > 0) for c, t in zip(cells, dates)]
# sanity: same-day label should agree with label_occurrence
same = np.array([int(count_between(c, t, -1, 0) > 0) for c, t in zip(cells, dates)])
print("same-day reconstruction agrees with label_occurrence:", (same == df["label_occurrence"].values).mean().round(4))

feat_df = make_features(df.drop(columns=[c for c in df.columns if c.startswith("label_") and c != "label_occurrence"] + ["recent_fire"]))
feats = pick_feature_cols(feat_df)
X = feat_df[feats].values
year = df["date"].dt.year.values

def hgb():
    return HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                                          l2_regularization=0.2, max_bins=255, early_stopping=True,
                                          validation_fraction=0.1, random_state=SEED)

rows = []
for pop in ("all_rows", "new_ignition"):
    mask_pop = np.ones(len(df), bool) if pop == "all_rows" else (df["recent_fire"].values == 0)
    for H in HORIZONS:
        y = df[f"label_{H}d"].values
        for test_year in (2022, 2023, 2024):
            tr = mask_pop & (year < test_year); te = mask_pop & (year == test_year)
            if y[tr].sum() < 50 or y[te].sum() < 20 or len(np.unique(y[te])) < 2:
                continue
            m = hgb().fit(X[tr], y[tr]); p = m.predict_proba(X[te])[:, 1]
            prev = y[te].mean()
            rows.append(dict(population=pop, horizon_d=H, test_year=test_year, n_train=int(tr.sum()), n_test=int(te.sum()),
                             prevalence=prev, roc_auc=roc_auc_score(y[te], p), pr_auc=average_precision_score(y[te], p),
                             pr_lift=average_precision_score(y[te], p) / prev))
            r = rows[-1]
            print(f"[{pop:12s}] H={H:2d}d test={test_year} n_train={r['n_train']:>7,} n_test={r['n_test']:>6,} "
                  f"prev={prev:.3f} auc={r['roc_auc']:.3f} pr={r['pr_auc']:.3f} lift={r['pr_lift']:.2f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "lead_time.csv", index=False)
print("\n== mean over test years ==")
print(res.groupby(["population", "horizon_d"])[["prevalence", "roc_auc", "pr_auc", "pr_lift"]].mean().round(3))
