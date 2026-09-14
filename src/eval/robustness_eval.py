#!/usr/bin/env python3
"""
Robustness checks on the final modeling table:
  1. Rolling-origin (forward-chained) evaluation: test on 2022, 2023, 2024 in turn.
  2. Spatially blocked CV (GroupKFold over ~20 km tiles) vs. random KFold.
Feature construction is identical to src/models/model_benchmark.py.
Usage: python src/eval/robustness_eval.py [path/to/modeling_table.parquet]
Writes outputs/robustness/{rolling_origin,spatial_vs_random_cv}.csv
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import GroupKFold, KFold
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "models"))
from model_benchmark import make_features, pick_feature_cols

SEED = 42
PATH = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "data/processed/modeling_table_v5.parquet")
OUT = REPO / "outputs" / "robustness"; OUT.mkdir(parents=True, exist_ok=True)

def models(pos_rate):
    m = {
        "hgb": HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                                             l2_regularization=0.2, max_bins=255, early_stopping=True,
                                             validation_fraction=0.1, random_state=SEED),
        "rf": Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("clf", RandomForestClassifier(n_estimators=150, min_samples_leaf=50, n_jobs=-1, random_state=SEED))]),
        "logreg": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                            ("clf", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))]),
    }
    try:
        from lightgbm import LGBMClassifier
        m["lgbm"] = Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.06, num_leaves=63, min_child_samples=200,
                                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                                                     n_jobs=-1, verbose=-1))])
    except Exception:
        pass
    return m

def max_f1_thr(y, p):
    pr, rc, t = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.nanargmax(f1)); return t[max(i - 1, 0)] if len(t) else 0.5

def evaluate(y, p, thr):
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                f1=f1_score(y, (p >= thr).astype(int)))

df = pd.read_parquet(PATH)
df["date"] = pd.to_datetime(df["date"])
lat, lon, cell = df["latitude"].values, df["longitude"].values, df["cell_id"].values
df = make_features(df)
feats = pick_feature_cols(df)
X, y = df[feats].values, df["label_occurrence"].values.astype(int)
year = df["date"].dt.year.values
print(f"rows={len(df):,} features={len(feats)}")

# ---------- 1. rolling origin ----------
rows = []
for test_year in (2022, 2023, 2024):
    val_year = test_year - 1
    tr, va, te = year < val_year, year == val_year, year == test_year
    for name, mdl in models(y[tr].mean()).items():
        mdl.fit(X[tr], y[tr])
        pva, pte = mdl.predict_proba(X[va])[:, 1], mdl.predict_proba(X[te])[:, 1]
        thr = max_f1_thr(y[va], pva)
        r = evaluate(y[te], pte, thr)
        rows.append(dict(model=name, test_year=test_year, n_train=int(tr.sum()), n_test=int(te.sum()),
                         val_auc=roc_auc_score(y[va], pva), **r))
        print(f"[rolling] {name:7s} test={test_year} n_train={tr.sum():>7,} val_auc={rows[-1]['val_auc']:.3f} "
              f"test_auc={r['roc_auc']:.3f} pr={r['pr_auc']:.3f} f1={r['f1']:.3f}")
roll = pd.DataFrame(rows)
roll.to_csv(OUT / "rolling_origin.csv", index=False)
print("\n== rolling-origin summary (test ROC-AUC) ==")
print(roll.pivot(index="model", columns="test_year", values="roc_auc").round(3)
      .assign(mean=lambda d: d.mean(axis=1).round(3), std=lambda d: d.std(axis=1).round(3)))

# ---------- 2. spatial block vs random CV ----------
# ~20 km tiles: 0.18 deg lat, 0.23 deg lon at 39N
tile = (np.floor((lat - lat.min()) / 0.18).astype(int) * 1000 + np.floor((lon - lon.min()) / 0.23).astype(int))
print(f"\nspatial tiles: {len(np.unique(tile))}")
res = []
for scheme, splitter in (("spatial_block", GroupKFold(n_splits=5)), ("random", KFold(n_splits=5, shuffle=True, random_state=SEED))):
    split_iter = splitter.split(X, y, groups=tile) if scheme == "spatial_block" else splitter.split(X, y)
    for k, (tr, te) in enumerate(split_iter):
        mdl = models(y[tr].mean())["hgb"]
        mdl.fit(X[tr], y[tr])
        p = mdl.predict_proba(X[te])[:, 1]
        res.append(dict(scheme=scheme, fold=k, roc_auc=roc_auc_score(y[te], p), pr_auc=average_precision_score(y[te], p)))
        print(f"[cv] {scheme:14s} fold={k} auc={res[-1]['roc_auc']:.3f} pr={res[-1]['pr_auc']:.3f}")
cv = pd.DataFrame(res)
cv.to_csv(OUT / "spatial_vs_random_cv.csv", index=False)
print("\n== HGB 5-fold CV: spatial block vs random ==")
print(cv.groupby("scheme")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).round(3))
