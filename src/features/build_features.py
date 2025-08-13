
# src/features/build_features.py

def build_grid(aoi_bbox, resolution_m=500):
    """Create a simple grid covering the AOI (placeholder)."""
    pass

def compute_climatology(ndvi_time_series):
    """Return per-cell, per-DOY mean/std for NDVI/EVI."""
    pass

def compute_anomalies(ts, climatology):
    """Compute z-scores (value - mean) / std."""
    pass

def add_lags(df, cols, lags):
    """Shift selected columns by specified lags (in steps) and suffix _t_k."""
    pass
