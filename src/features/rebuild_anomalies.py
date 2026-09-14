#!/usr/bin/env python3
"""
rebuild_anomalies.py

Adds ndvi_anom and ndvi_z to a single-table parquet dataset.
Usage:
  python src/features/rebuild_anomalies.py --input path/to/in.parquet --output path/to/out.parquet --train-year 2022
"""
import argparse
import pandas as pd
import numpy as np

def add_ndvi_anom_and_z(df: pd.DataFrame, train_year: int = 2022):
    df = df.copy()
    df['comp_date'] = pd.to_datetime(df['comp_date'], errors='coerce')
    train_mask = df['comp_date'].notna() & df['ndvi'].notna() & (df['comp_date'].dt.year <= train_year)
    train_obs = df.loc[train_mask, ['cell_id','ndvi']].copy()
    if train_obs.empty:
        raise ValueError(f"No training-period NDVI observations found (comp_date.year <= {train_year}).")
    stats = train_obs.groupby('cell_id').agg(ndvi_mean=('ndvi','mean'), ndvi_std=('ndvi','std')).reset_index()
    stats['ndvi_std'] = stats['ndvi_std'].fillna(0.0).replace(0.0,1e-6).fillna(1e-6)
    df = df.merge(stats, how='left', on='cell_id')
    df['ndvi_anom'] = np.where(df['ndvi'].notna() & df['ndvi_mean'].notna(), df['ndvi'] - df['ndvi_mean'], np.nan)
    df['ndvi_z'] = np.where(df['ndvi_anom'].notna(), df['ndvi_anom'] / df['ndvi_std'], np.nan)
    df = df.drop(columns=['ndvi_mean','ndvi_std'], errors='ignore')
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-year", type=int, default=2022)
    args = p.parse_args()

    print("Loading:", args.input)
    df = pd.read_parquet(args.input)
    print("Computing ndvi_anom and ndvi_z (train_year =", args.train_year, ")")
    df_new = add_ndvi_anom_and_z(df, train_year=args.train_year)
    print("Saving:", args.output)
    df_new.to_parquet(args.output, index=False)
    print("Done. Sample:")
    print(df_new[['cell_id','date','ndvi','ndvi_anom','ndvi_z']].head().to_string(index=False))

if __name__ == "__main__":
    main()
