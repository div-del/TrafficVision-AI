import pandas as pd
import numpy as np
import os
import logging
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_severity_model(df):
    logger.info("Training Severity Classification Model")
    
    features = ['event_type', 'event_cause', 'priority', 'requires_road_closure', 
                'latitude', 'longitude', 'zone', 'junction', 'corridor', 
                'police_station', 'veh_type', 'hour', 'month', 'day_of_week', 'is_peak_hour']
    
    X = df[features]
    y = df['severity_class']
    
    cat_features = ['event_type', 'event_cause', 'priority', 'zone', 'junction', 'corridor', 'police_station', 'veh_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = CatBoostClassifier(
        iterations=1000, 
        learning_rate=0.05, 
        depth=8, 
        l2_leaf_reg=3,
        verbose=100, 
        task_type='CPU',
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_features)
    
    # Save feature importance
    importance = model.get_feature_importance(type='FeatureImportance')
    feat_imp_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values(by='importance', ascending=False)
    feat_imp_df.to_csv('models/severity_importance.csv', index=False)
    
    y_pred = model.predict(X_test)
    logger.info("Severity Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    model.save_model('models/severity_model.cbm')
    return model

def train_resolution_model(df):
    logger.info("Training Resolution Time Regression Model")
    
    features = ['event_type', 'event_cause', 'priority', 'requires_road_closure', 
                'latitude', 'longitude', 'zone', 'junction', 'corridor', 
                'police_station', 'veh_type', 'hour', 'month', 'day_of_week', 'is_peak_hour']
    
    X = df[features]
    y = df['resolution_minutes']
    
    cat_features = ['event_type', 'event_cause', 'priority', 'zone', 'junction', 'corridor', 'police_station', 'veh_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CatBoostRegressor(
        iterations=1000, 
        learning_rate=0.05, 
        depth=8, 
        l2_leaf_reg=3,
        verbose=100, 
        task_type='CPU',
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_features)
    
    # Save feature importance
    importance = model.get_feature_importance(type='FeatureImportance')
    feat_imp_df = pd.DataFrame({'feature': features, 'importance': importance}).sort_values(by='importance', ascending=False)
    feat_imp_df.to_csv('models/resolution_importance.csv', index=False)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Resolution Time MAE: {mae:.2f}")
    logger.info(f"Resolution Time R2: {r2:.2f}")
    
    model.save_model('models/resolution_model.cbm')
    return model

if __name__ == "__main__":
    processed_path = 'data/processed_data.csv'
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        # Ensure correct types
        cat_features = ['event_type', 'event_cause', 'priority', 'zone', 'junction', 'corridor', 'police_station', 'veh_type']
        for col in cat_features:
            df[col] = df[col].astype(str)
            
        train_severity_model(df)
        train_resolution_model(df)
    else:
        logger.error(f"Processed data {processed_path} not found.")
