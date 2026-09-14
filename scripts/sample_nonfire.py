#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling non-fire cell and date

Outputs:
  - data/interim/cell_labels_balanced.parquet
"""

import argparse, math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

def sample_nonfire(
    grid_path: str,
    positives_path: str,
    out_path: str,
    ratio: float = 1.0,
    season_only: bool = True,
    months=(6, 7, 8, 9, 10, 11),
    neighbor_buffer_m: float = 0.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    grid = gpd.read_parquet(grid_path)
    if "cell_id" not in grid.columns:
        raise ValueError("grid needs a 'cell_id' column")
    if "geometry" not in grid.columns:
        raise ValueError("grid needs a 'geometry' column")
    grid = grid.set_geometry("geometry")
    if grid.crs is None:
        grid = grid.set_crs(4326, allow_override=True)
    grid_wgs84 = grid.to_crs(4326)

    pos = pd.read_parquet(positives_path)
    if "acq_date" in pos.columns and "date" not in pos.columns:
        pos = pos.rename(columns={"acq_date": "date"})
    pos["date"] = pd.to_datetime(pos["date"]).dt.date
    pos = pos[["cell_id", "date"]].drop_duplicates()

    if season_only:
        pos = pos[pd.to_datetime(pos["date"]).dt.month.isin(months)]

    if pos.empty:
        raise ValueError("No positive rows after season filter; widen months or set --no-season.")

    neighbor_map = None
    if neighbor_buffer_m and neighbor_buffer_m > 0:
        grid_m = grid_wgs84.to_crs(3857)
        cent = grid_m.geometry.centroid
        grid_m = grid_m.copy()
        grid_m["centroid"] = cent
        grid_m = grid_m.set_geometry("centroid")

        sindex = grid_m.sindex
        neighbor_map = {}
        for cid, geom in zip(grid_m["cell_id"], grid_m.geometry):
            buf = geom.buffer(neighbor_buffer_m)
            cand_idx = list(sindex.query(buf, predicate="intersects"))
            neigh_ids = grid_m.iloc[cand_idx]["cell_id"].tolist()
            neighbor_map[cid] = set(neigh_ids)

    pos_by_date = pos.groupby("date")["cell_id"].apply(set).to_dict()
    all_cell_ids = set(grid_wgs84["cell_id"].tolist())

    neg_rows = []
    for d, pos_cells in pos_by_date.items():
        n_pos = len(pos_cells)
        n_neg = math.ceil(ratio * n_pos)

        forbidden = set(pos_cells)
        if neighbor_map is not None:
            neigh_forbid = set()
            for cid in pos_cells:
                neigh_forbid |= neighbor_map.get(cid, set())
            forbidden |= neigh_forbid

        candidates = np.array(list(all_cell_ids - forbidden))
        if len(candidates) == 0:
            continue

        if n_neg > len(candidates):
            sample_ids = candidates
        else:
            sample_ids = rng.choice(candidates, size=n_neg, replace=False)

        neg_rows.extend([(int(cid), d, 0, 0.0) for cid in sample_ids])

    neg = pd.DataFrame(neg_rows, columns=["cell_id", "date", "label_occurrence", "label_severity"])

    pos_labeled = pos.copy()
    pos_labeled["label_occurrence"] = 1
    pos_labeled["label_severity"] = 0.0

    out = pd.concat([pos_labeled, neg], ignore_index=True)
    out = out.drop_duplicates(subset=["cell_id", "date", "label_occurrence"])
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(out_path, index=False)

    pr = out["label_occurrence"].mean()
    print(f"[OK] wrote {out_path} rows={len(out):,}  pos_rate={pr:.4%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="data/interim/grid_500m_northern_sierra.parquet")
    ap.add_argument("--positives", default="data/interim/cell_labels.parquet")
    ap.add_argument("--out", default="data/interim/cell_labels_balanced.parquet")
    ap.add_argument("--ratio", type=float, default=1.0, help="neg:pos ratio per active day")
    ap.add_argument("--season", action="store_true", help="restrict to Aug–Nov (fire season)")
    ap.add_argument("--no-season", dest="season", action="store_false")
    ap.set_defaults(season=True)
    ap.add_argument("--neighbor-buffer-m", type=float, default=0.0,
                    help="exclude neighbors within this many meters (e.g., 600). 0 disables.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sample_nonfire(
        grid_path=args.grid,
        positives_path=args.positives,
        out_path=args.out,
        ratio=args.ratio,
        season_only=args.season,
        neighbor_buffer_m=args.neighbor_buffer_m,
        seed=args.seed,
    )
