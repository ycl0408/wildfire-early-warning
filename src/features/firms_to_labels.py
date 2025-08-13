
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# CONFIG (edit paths as needed)
# -------------------------------
FIRMS_CSV = "data/raw/firms/modis_2024_United_States.csv"   # your uploaded file path
GRID_PARQUET = "data/interim/grid_cells.parquet"            # expected grid centroids (cell_id, lat, lon)
OUT_EVENTS_PARQUET = "data/interim/firms_events.parquet"    # per (cell_id, date) fire indicator
STATE_BBOX = (-124.48, 32.53, -114.13, 42.01)               # California bbox (lon_min, lat_min, lon_max, lat_max)
MAX_SNAP_KM = 1.5                                           # max distance to snap a fire point to a grid cell

# -------------------------------
# UTILITIES
# -------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    # Vectorized haversine distance (approx) in km
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def day_to_16day_bin(date):
    # Returns an integer bin 0..22 for the 16-day period of the year
    doy = date.timetuple().tm_yday
    return (doy - 1) // 16

def load_and_filter_firms(path_csv, bbox):
    df = pd.read_csv(path_csv, parse_dates=['acq_date'])
    lon_min, lat_min, lon_max, lat_max = bbox
    df = df[(df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) &
            (df['latitude']  >= lat_min) & (df['latitude']  <= lat_max)].copy()
    # Keep minimal columns
    keep = ['latitude','longitude','acq_date','acq_time','satellite','instrument','confidence','frp']
    existing_cols = [c for c in keep if c in df.columns]
    df = df[existing_cols]
    df = df.rename(columns={'acq_date':'date'})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df

def load_grid(path_parquet):
    # Expect columns: cell_id, lat, lon (centroids of your grid)
    g = pd.read_parquet(path_parquet)
    if not set(['cell_id','lat','lon']).issubset(g.columns):
        raise ValueError("grid_cells.parquet must have columns: cell_id, lat, lon")
    return g[['cell_id','lat','lon']].copy()

def snap_points_to_grid(fires_df, grid_df, max_km=1.5):
    # Use NearestNeighbors in degrees, then filter by haversine distance
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nn.fit(grid_df[['lat','lon']].values)
    dist_idx = nn.kneighbors(fires_df[['latitude','longitude']].values, return_distance=False)
    nearest = grid_df.iloc[dist_idx.flatten()].reset_index(drop=True)
    out = fires_df.reset_index(drop=True).join(nearest[['cell_id','lat','lon']], how='left')
    # compute true km distance and drop far matches
    d_km = haversine_km(out['latitude'].values, out['longitude'].values, out['lat'].values, out['lon'].values)
    out = out.assign(dist_km=d_km)
    out = out[out['dist_km'] <= max_km].copy()
    return out[['cell_id','date']]

def build_daily_events(snapped_df):
    # One row per (cell_id, date), fire=1 if any detection that day
    ev = (snapped_df
          .groupby(['cell_id','date'])
          .size()
          .reset_index(name='count'))
    ev['fire'] = (ev['count'] > 0).astype(int)
    ev = ev[['cell_id','date','fire']]
    return ev

def align_to_16day_bins(events_df):
    # Add a 16-day bin index per year to help align with MOD13Q1 steps
    events_df = events_df.copy()
    events_df['year'] = events_df['date'].dt.year
    events_df['bin16'] = events_df['date'].apply(day_to_16day_bin).astype(int)
    # Aggregate to one indicator per (cell_id, year, bin16)
    out = (events_df
           .groupby(['cell_id','year','bin16'])['fire']
           .max()
           .reset_index())
    return out

def main():
    Path("data/interim").mkdir(parents=True, exist_ok=True)

    print("Loading FIRMS and filtering to California…")
    fires = load_and_filter_firms(FIRMS_CSV, STATE_BBOX)
    print(f"FIRMS rows after bbox filter: {len(fires):,}")

    print("Loading grid centroids…")
    grid = load_grid(GRID_PARQUET)
    print(f"Grid cells: {len(grid):,}")

    print("Snapping FIRMS points to nearest grid cell…")
    snapped = snap_points_to_grid(fires, grid, MAX_SNAP_KM)
    print(f"Snapped detections: {len(snapped):,}")

    print("Building daily fire indicators (per cell_id, date)…")
    events_daily = build_daily_events(snapped)
    print(f"Daily event rows: {len(events_daily):,}")

    print(f"Saving daily events to {OUT_EVENTS_PARQUET}")
    events_daily.to_parquet(OUT_EVENTS_PARQUET, index=False)

    print("Also computing 16-day bin indicators (optional)…")
    events_bin = align_to_16day_bins(events_daily)
    out_bins = OUT_EVENTS_PARQUET.replace(".parquet", "_bin16.parquet")
    events_bin.to_parquet(out_bins, index=False)
    print(f"Saved 16-day bins to {out_bins}")

if __name__ == "__main__":
    main()
