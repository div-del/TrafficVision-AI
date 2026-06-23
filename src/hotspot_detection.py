import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_hotspots(df_path):
    logger.info(f"Detecting hotspots from {df_path}")
    df = pd.read_csv(df_path)
    
    # Coordinates for clustering
    coords = df[['latitude', 'longitude']].values
    
    # DBSCAN (eps=0.01 is approx 1km, min_samples=10)
    db = DBSCAN(eps=0.005, min_samples=20, metric='haversine').fit(np.radians(coords))
    
    df['cluster_id'] = db.labels_
    
    # Generate hotspot reports
    hotspots = df[df['cluster_id'] != -1].copy()
    hotspot_stats = hotspots.groupby('cluster_id').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'id': 'count',
        'resolution_minutes': 'mean'
    }).rename(columns={'id': 'incident_count', 'resolution_minutes': 'avg_resolution_time'})
    
    hotspot_stats = hotspot_stats.sort_values(by='incident_count', ascending=False)
    hotspot_stats.to_csv('reports/top_hotspots.csv')
    
    logger.info(f"Detected {len(hotspot_stats)} hotspots.")
    return df, hotspot_stats

if __name__ == "__main__":
    processed_path = 'data/processed_data.csv'
    if os.path.exists(processed_path):
        df_with_clusters, stats = detect_hotspots(processed_path)
        # Save back the processed data with cluster IDs
        df_with_clusters.to_csv(processed_path, index=False)
        logger.info("Hotspot detection complete and data updated.")
    else:
        logger.error(f"Processed data {processed_path} not found.")
