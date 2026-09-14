#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_curve,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import check_random_state
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

FEATURE_BLACKLIST = {"cell_id","longitude","latitude","date","comp_date",
                     "comp_date_lag_2w","comp_date_lag_4w","comp_date_lag_6w",
                     "label_severity","split","landcover_igbp", "landcover_label", "time"}

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for base in ["ndvi","evi","ndvi_anom","evi_anom","ndvi_z","evi_z"]:
        if base not in df.columns: 
            continue
        for tag, days in [("2w",14),("4w",28),("6w",42)]:
            lag = f"{base}_lag_{tag}"
            if lag in df.columns:
                df[f"{base}_delta_{tag}"] = df[base] - df[lag]
                if base in ("ndvi","evi","ndvi_z","evi_z"):
                    df[f"{base}_slope_{tag}"] = (df[base] - df[lag]) / days
    if set(["tmmx_C","tmmn_C"]).issubset(df.columns):
        df["diurnal_range_C"] = df["tmmx_C"] - df["tmmn_C"]
    if set(["vpd_kPa","tmmx_C"]).issubset(df.columns):
        df["vpd_x_tmmx"] = df["vpd_kPa"] * df["tmmx_C"]
    if set(["vpd_kPa","vs"]).issubset(df.columns):
        df["vpd_x_wind"] = df["vpd_kPa"] * df["vs"]
    if set(["ndvi_z","vpd_kPa"]).issubset(df.columns):
        df["ndvi_z_x_vpd"] = df["ndvi_z"] * df["vpd_kPa"]
    if set(["ndvi_z","tmmx_C"]).issubset(df.columns):
        df["ndvi_z_x_tmmx"] = df["ndvi_z"] * df["tmmx_C"]

    drop_ts = ["comp_date"] + [c for c in df.columns if c.startswith("comp_date_lag_")]
    df = df.drop(columns=[c for c in drop_ts if c in df.columns])

    IGBP_GROUP = {
        0: "water",
        1: "forest", 2: "forest", 3: "forest", 4: "forest", 5: "forest",
        6: "shrub", 7: "shrub",
        8: "savanna", 9: "savanna",
        10: "grass",
        11: "wetland",
        12: "crop", 14: "crop",
        13: "urban",
        15: "barren", 16: "barren",
        17: "other",
        255: "nan",
    }   

    if "landcover_igbp" in df.columns:
        lc_group = df["landcover_igbp"].astype("float").map(IGBP_GROUP).fillna("nan")
        lc_dum = pd.get_dummies(lc_group, prefix="lc", dtype=np.uint8)
        df = pd.concat([df, lc_dum], axis=1)

    return df

def pick_feature_cols(df: pd.DataFrame) -> List[str]:
    drop = {"label_occurrence","label_severity","date"} | FEATURE_BLACKLIST
    cols = [c for c in df.columns if c not in drop]
    num = df[cols].select_dtypes(include=["number"]).columns.tolist()
    return sorted(num)

def to_py(obj):
    """Recursively convert NumPy types to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def time_splits(df: pd.DataFrame, train_end: str, val_start: str, val_end: str, test_start: str):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] <= train_end]
    val   = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
    test  = df[df["date"] >= test_start]
    return train, val, test


def pick_threshold(y_true: np.ndarray, proba: np.ndarray, target_precision: float=None) -> Tuple[float,float]:
    p, r, t = precision_recall_curve(y_true, proba)
    if target_precision is not None:
        mask = p >= target_precision
        if mask.any():
            idx = np.argmax(r[mask])
            chosen_t = t[np.where(mask)[0][idx]-1] if len(t)>0 else 0.5
            f1 = 2*p[mask][idx]*r[mask][idx]/(p[mask][idx]+r[mask][idx]+1e-12)
            return chosen_t, f1

    f1 = 2*p*r/(p+r+1e-12)
    idx = int(np.nanargmax(f1))
    thr = t[max(idx-1,0)] if len(t)>0 else 0.5
    return thr, float(np.nanmax(f1))


def evaluate_split(y, proba, thr: float, name: str) -> Dict[str,Any]:
    roc = roc_auc_score(y, proba) if len(np.unique(y))>1 else np.nan
    ap  = average_precision_score(y, proba)
    yhat = (proba >= thr).astype(int)
    f1  = f1_score(y, yhat, zero_division=0)
    return {"name":name, "roc_auc":roc, "pr_auc":ap, "f1":f1, "thr":thr}

def build_model_registry(random_state: int, pos_rate: float):
    models = {}

    models["hgb"] = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_leaf_nodes=63,
        max_depth=None,
        min_samples_leaf=200,
        l2_regularization=0.2,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )

    models["rf"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=random_state,
            class_weight=None,
        ))
    ])

    models["logreg"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)), 
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="saga", max_iter=5000,
            n_jobs=-1, class_weight="balanced",
        )),
    ])

    hgb_base = HistGradientBoostingClassifier(
        learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=0.2, max_bins=255, random_state=random_state,
    )
    try:
        models["hgb_calibrated"] = CalibratedClassifierCV(
            estimator=hgb_base, method="isotonic", cv=3
        )
    except TypeError:
        models["hgb_calibrated"] = CalibratedClassifierCV(
            base_estimator=hgb_base, method="isotonic", cv=3
        )

    models["extratrees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=30,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        ))
    ])

    models["lin_svc_cal"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True)),
        ("svc_cal", CalibratedClassifierCV(
            estimator=LinearSVC(C=0.5, class_weight="balanced", max_iter=5000),
            method="sigmoid", cv=3
        )),
    ])

    try:
        from lightgbm import LGBMClassifier
        models["lgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LGBMClassifier(
                n_estimators=1500,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=200,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary",
                random_state=random_state,
                n_jobs=-1,
            ))
        ])
    except Exception as e:
        print("[warn] LightGBM not available:", e)

    try:
        from xgboost import XGBClassifier
        spw = (1.0 - pos_rate) / max(pos_rate, 1e-6)
        models["xgb"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
                eval_metric="aucpr",
                scale_pos_weight=spw,
            ))
        ])
    except Exception as e:
        print("[warn] XGBoost not available:", e)

    return models

def main():
    ap = argparse.ArgumentParser("Model benchmark runner")
    ap.add_argument("--in", dest="inp", required=True,
                    help="CSV or Parquet file with features")
    ap.add_argument("--outdir", default="outputs/model_bench")
    ap.add_argument("--train-end", default="2022-12-31")
    ap.add_argument("--val-start", default="2023-01-01")
    ap.add_argument("--val-end", default="2023-12-31")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--rebalance", action="store_true",
                    help="Undersample negatives in TRAIN to ~1:3 pos:neg")
    ap.add_argument("--target-precision", type=float, default=None,
                    help="If set, choose threshold achieving at least this precision on VAL")
    ap.add_argument("--perm", action="store_true", help="Compute permutation importance on VAL")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.inp.endswith(".parquet"):
        df = pd.read_parquet(args.inp)
    else:
        df = pd.read_csv(args.inp)

    df = make_features(df)
    feats = pick_feature_cols(df)

    train, val, test = time_splits(df, args.train_end, args.val_start, args.val_end, args.test_start)

    if args.rebalance:
        pos = train[train.label_occurrence==1]
        neg = train[train.label_occurrence==0]
        n_pos = len(pos)
        n_neg_keep = int(n_pos*(1-0.25)/0.25)
        neg_keep = neg.sample(n=min(n_neg_keep, len(neg)), random_state=args.seed)
        train = pd.concat([pos, neg_keep]).sample(frac=1, random_state=args.seed)

    Xtr, ytr = train[feats].values, train.label_occurrence.values
    Xva, yva = val[feats].values,   val.label_occurrence.values
    Xte, yte = test[feats].values,  test.label_occurrence.values

    pos_rate = float(train.label_occurrence.mean())
    models = build_model_registry(args.seed, pos_rate)

    summary_rows = []

    for name, model in models.items():
        print(f"\n=== Training {name} on {len(train)} rows, {len(feats)} features ===")
        model.fit(Xtr, ytr)

        proba_val = model.predict_proba(Xva)[:,1]
        thr, f1_val = pick_threshold(yva, proba_val, target_precision=args.target_precision)
        proba_te = model.predict_proba(Xte)[:,1]

        res_val = evaluate_split(yva, proba_val, thr, name="VAL")
        res_te  = evaluate_split(yte, proba_te, thr, name="TEST")

        res_val["prevalence"] = float(yva.mean())
        res_val["lift_pr"] = res_val["pr_auc"]/res_val["prevalence"] if res_val["prevalence"]>0 else np.nan
        res_te["prevalence"] = float(yte.mean())
        res_te["lift_pr"] = res_te["pr_auc"]/res_te["prevalence"] if res_te["prevalence"]>0 else np.nan

        top_feats = []
        if args.perm:
            try:
                imp = permutation_importance(model, Xva, yva, n_repeats=5, random_state=args.seed)
                idx = np.argsort(-imp.importances_mean)[:20]
                top_feats = [(feats[i], float(imp.importances_mean[i]), float(imp.importances_std[i])) for i in idx]
            except Exception as e:
                top_feats = [("perm_error", str(e), 0.0)]

        out_dir_m = os.path.join(args.outdir, name)
        os.makedirs(out_dir_m, exist_ok=True)
        try:
            dump(model, os.path.join(out_dir_m, f"model_{name}.joblib"))
        except Exception:
            pass
        with open(os.path.join(out_dir_m, "features.json"), "w") as f:
            json.dump({"feature_cols": feats}, f, indent=2)
        metrics_payload = {
            "val": res_val,
            "test": res_te,
            "threshold": float(thr),
            "top_feats": top_feats,
        }
        with open(os.path.join(out_dir_m, "metrics.json"), "w") as f:
            json.dump(to_py(metrics_payload), f, indent=2, allow_nan=True)

        summary_rows.append({
            "model": name,
            "thr": float(thr),
            "val_auc": res_val["roc_auc"], "val_pr": res_val["pr_auc"], "val_f1": res_val["f1"], "val_lift": res_val["lift_pr"],
            "test_auc": res_te["roc_auc"], "test_pr": res_te["pr_auc"], "test_f1": res_te["f1"], "test_lift": res_te["lift_pr"],
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\n[OK] Wrote {summary_path}")

if __name__ == "__main__":
    main()
