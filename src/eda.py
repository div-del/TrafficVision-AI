import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_eda(df_path):
    logger.info(f"Starting EDA on {df_path}")
    df = pd.read_csv(df_path)
    
    if not os.path.exists('reports/plots'):
        os.makedirs('reports/plots')
    
    # 1. Event Cause Distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='event_cause', order=df['event_cause'].value_counts().index[:10])
    plt.title('Top 10 Event Causes')
    plt.savefig('reports/plots/event_cause_dist.png')
    plt.close()
    
    # 2. Event Type Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='event_type')
    plt.title('Event Type Distribution')
    plt.savefig('reports/plots/event_type_dist.png')
    plt.close()
    
    # 4. Zone Distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='zone', order=df['zone'].value_counts().index[:10])
    plt.title('Top 10 Zones')
    plt.savefig('reports/plots/zone_dist.png')
    plt.close()
    
    # 7. Hourly Incident Trend
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='hour')
    plt.title('Incident Frequency by Hour')
    plt.savefig('reports/plots/hourly_trend.png')
    plt.close()
    
    # 9. Resolution Time Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['resolution_minutes'], kde=True)
    plt.title('Resolution Time Distribution (Minutes)')
    plt.savefig('reports/plots/resolution_time_dist.png')
    plt.close()
    
    # 10. Severity Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='severity_class', order=['Low', 'Medium', 'High', 'Critical'])
    plt.title('Severity Class Distribution')
    plt.savefig('reports/plots/severity_dist.png')
    plt.close()
    
    logger.info("EDA completed and plots saved.")

if __name__ == "__main__":
    processed_path = 'data/processed_data.csv'
    if os.path.exists(processed_path):
        run_eda(processed_path)
    else:
        logger.error(f"Processed data {processed_path} not found.")
