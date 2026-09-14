import pandas as pd

# Paths (adjust if needed)
model_path = "data/processed/modeling_table_v3.parquet"
elev_csv   = "data/raw/points_elevation_30m.csv"
lc_csv     = "data/raw/points_landcover_500m.csv"
out_path   = "data/processed/modeling_table_v4.parquet"

# Load
df   = pd.read_parquet(model_path)
elev = pd.read_csv(elev_csv)
lc   = pd.read_csv(lc_csv)

# Normalize column names if Drive added extras
elev = elev[['cell_id','elevation']].rename(columns={'elevation':'elevation_m'})
lc   = lc[['cell_id','LC_Type1']].rename(columns={'LC_Type1':'landcover_igbp'})

# Merge
df = df.merge(elev, on='cell_id', how='left')
df = df.merge(lc,   on='cell_id', how='left')

# Optional: map IGBP code -> label
IGBP_MAP = {
    0:"Water",1:"Evergreen Needleleaf Forest",2:"Evergreen Broadleaf Forest",
    3:"Deciduous Needleleaf Forest",4:"Deciduous Broadleaf Forest",5:"Mixed Forests",
    6:"Closed Shrublands",7:"Open Shrublands",8:"Woody Savannas",9:"Savannas",
    10:"Grasslands",11:"Permanent Wetlands",12:"Croplands",13:"Urban and Built-up",
    14:"Cropland/Natural Vegetation Mosaic",15:"Snow and Ice",16:"Barren or Sparsely Vegetated",
    17:"Unclassified",255:"NoData"
}
df['landcover_label'] = df['landcover_igbp'].map(IGBP_MAP)

# Save
df.to_parquet(out_path, index=False)
print("Saved →", out_path)

# Quick sanity checks
print(df[['elevation_m','landcover_igbp']].describe())
print(df['landcover_igbp'].value_counts(dropna=False).head(10))
