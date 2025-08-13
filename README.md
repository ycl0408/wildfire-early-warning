# Wildfire Early Warning from Vegetation Anomalies (California, 2016–2024)

Predicting wildfire ignition risk **2–6 weeks ahead** in California using MODIS NDVI/EVI anomalies.

## Overview
This project tests whether vegetation stress signals (NDVI/EVI anomalies) can provide early wildfire risk warnings. We compute anomalies relative to each location’s historical norm, add lagged features, and train models to predict fires in the next **2, 4, or 6 weeks**.

## Data Sources
- **MODIS MOD13Q1** NDVI/EVI (NASA LP DAAC / AppEEARS / Google Earth Engine)  
- **NASA FIRMS** active fire detections  
- **SRTM** elevation, slope (optional)  

## Quickstart
```bash
# Clone repo
git clone https://github.com/ycl0408/wildfire-early-warning.git
cd wildfire-early-warning

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
1. Download MODIS & FIRMS data for California (2016–2024) into `data/raw/`.  
2. Run `notebooks/01_build_anomalies.ipynb` to create anomalies, lags, and labels.  
3. Train baseline models in `src/models/`.

## Status
- [ ] MODIS-only baseline  
- [ ] Compare Logistic Regression, Random Forest, LightGBM  
- [ ] Lead-time curve for 2/4/6 weeks  
