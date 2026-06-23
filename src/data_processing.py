import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset shape: {df.shape}")
    return df

def generate_reports(df):
    logger.info("Generating data reports")
    
    # Missing value report
    missing_report = df.isnull().sum().to_frame(name='missing_count')
    missing_report['missing_percentage'] = (missing_report['missing_count'] / len(df)) * 100
    missing_report.to_csv('reports/missing_report.csv')
    
    # Data summary
    summary = df.describe(include='all').transpose()
    summary.to_csv('reports/data_summary.csv')
    
    # Duplicate report
    duplicates = df.duplicated().sum()
    logger.info(f"Number of duplicate rows: {duplicates}")
    
    with open('reports/duplicate_report.txt', 'w') as f:
        f.write(f"Total duplicate rows: {duplicates}\n")

def clean_data(df):
    logger.info("Cleaning data")
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Convert datetime columns
    datetime_cols = ['start_datetime', 'end_datetime', 'created_date', 'modified_datetime', 'closed_datetime']
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 3. Drop high-missing columns (>80%)
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.8].index.tolist()
    # Also drop columns mentioned in prompt
    prompt_drop = ['map_file', 'comment', 'meta_data', 'direction', 'route_path', 
                   'assigned_to_police_id', 'citizen_accident_id', 'resolved_at_address', 
                   'resolved_at_latitude', 'resolved_at_longitude', 'resolved_by_id', 'resolved_datetime', 'veh_no']
    cols_to_drop = list(set(cols_to_drop + prompt_drop))
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    logger.info(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    
    # 4. Fill missing categorical values with "Unknown"
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    # 5. Fill missing numerical values using median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    return df

def create_targets(df):
    logger.info("Creating targets")
    
    # resolution_minutes = (closed_datetime - start_datetime)
    df['resolution_minutes'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60
    
    # Remove negative durations
    df = df[df['resolution_minutes'] >= 0]
    
    # Remove outliers using IQR
    Q1 = df['resolution_minutes'].quantile(0.25)
    Q3 = df['resolution_minutes'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['resolution_minutes'] >= lower_bound) & (df['resolution_minutes'] <= upper_bound)]
    
    # Create severity_class
    def get_severity(mins):
        if mins <= 30: return 'Low'
        elif mins <= 120: return 'Medium'
        elif mins <= 300: return 'High'
        else: return 'Critical'
        
    df['severity_class'] = df['resolution_minutes'].apply(get_severity)
    
    return df

def feature_engineering(df):
    logger.info("Performing feature engineering")
    
    # Temporal features
    df['hour'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.dayofweek
    df['month'] = df['start_datetime'].dt.month
    df['quarter'] = df['start_datetime'].dt.quarter
    df['week_of_year'] = df['start_datetime'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Peak hour indicator
    def is_peak(h):
        if (7 <= h <= 10) or (17 <= h <= 21):
            return 1
        return 0
    df['is_peak_hour'] = df['hour'].apply(is_peak)
    
    # Spatial features
    df['event_density_zone'] = df.groupby('zone')['id'].transform('count')
    df['event_density_junction'] = df.groupby('junction')['id'].transform('count')
    
    # Historical stats
    df['avg_res_time_cause'] = df.groupby('event_cause')['resolution_minutes'].transform('mean')
    df['avg_res_time_zone'] = df.groupby('zone')['resolution_minutes'].transform('mean')
    df['avg_res_time_corridor'] = df.groupby('corridor')['resolution_minutes'].transform('mean')
    
    return df

if __name__ == "__main__":
    raw_path = 'data/dataset.csv'
    if os.path.exists(raw_path):
        df = load_data(raw_path)
        generate_reports(df)
        df = clean_data(df)
        df = create_targets(df)
        df = feature_engineering(df)
        
        processed_path = 'data/processed_data.csv'
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
    else:
        logger.error(f"File {raw_path} not found.")
