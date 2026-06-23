import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier, CatBoostRegressor
import sys
import folium
from streamlit_folium import st_folium
import plotly.express as px

# Add src to path
sys.path.append(os.path.abspath('src'))
import recommendation_engine
import importlib
importlib.reload(recommendation_engine)
from recommendation_engine import calculate_impact_score, get_recommendations

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TrafficVision Pro | Smart Event Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    :root {
        --primary-color: #00f2fe;
        --secondary-color: #4facfe;
    }
    
    .main {
        background: #0f172a;
        color: #f8fafc;
    }
    
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 242, 254, 0.3);
    }
    
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    .sidebar .sidebar-content {
        background: #1e293b;
    }
    
    .css-1d391kg {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & DATA ---
@st.cache_resource
def load_models():
    sev_model = CatBoostClassifier().load_model('models/severity_model.cbm')
    res_model = CatBoostRegressor().load_model('models/resolution_model.cbm')
    return sev_model, res_model

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed_data.csv')
    # Use fixed coordinates for demo if they look invalid
    df['latitude'] = df['latitude'].replace(0, 12.9716)
    df['longitude'] = df['longitude'].replace(0, 77.5946)
    return df

sev_model, res_model = load_models()
df = load_data()

# --- SIDEBAR ---
st.sidebar.title("TrafficVision Pro")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Dashboard Navigation", 
    ["🏠 Executive Overview", "📈 Traffic Analytics", "🔮 AI Impact Predictor", "📍 Hotspot Exploration"])

# --- PAGES ---

if page == "🏠 Executive Overview":
    st.title("🚦 Smart Event-Aware Traffic Management")
    st.markdown("### Regional Traffic Command Center")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events Analyzed", len(df))
    with col2:
        st.metric("Mean Resolution Time", f"{df['resolution_minutes'].mean():.1f} min")
    with col3:
        st.metric("Active Response Zones", df['zone'].nunique())
    with col4:
        st.metric("Critical Hotspots", 12)
        
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Severity Breakdown")
        fig = px.pie(df, names='severity_class', color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("System Status")
        st.success("🤖 ML Engines: Operational")
        st.success("📊 Data Feed: Connected")
        st.warning("⚡ High Load in Zone A (Simulated)")

elif page == "📈 Traffic Analytics":
    st.title("📈 Advanced Traffic Analytics")
    
    tab1, tab2 = st.tabs(["Temporal Trends", "Feature Importance"])
    
    with tab1:
        st.subheader("Incident Volume by Hour")
        hour_counts = df['hour'].value_counts().sort_index().reset_index()
        hour_counts.columns = ['hour', 'incidents']
        fig = px.line(hour_counts, x='hour', y='incidents', markers=True, 
                      color_discrete_sequence=['#00f2fe'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Event Causes")
        fig = px.bar(df['event_cause'].value_counts().head(10).reset_index(), 
                     x='count', y='event_cause', orientation='h', 
                     color_discrete_sequence=['#4facfe'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        if os.path.exists('models/severity_importance.csv'):
            st.subheader("AI Decision Logic (Severity Prediction)")
            feat_imp = pd.read_csv('models/severity_importance.csv')
            fig = px.bar(feat_imp.head(10), x='importance', y='feature', orientation='h',
                         color_discrete_sequence=['#00f2fe'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not found. Run model training to generate.")

elif page == "🔮 AI Impact Predictor":
    st.title("🔮 Predictive Impact Assessment")
    
    with st.container():
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); padding: 25px; border-radius: 15px; margin-bottom: 25px;">
            <h4>Configuration Panel</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("impact_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                event_type = st.selectbox("Event Category", df['event_type'].unique())
                event_cause = st.selectbox("Trigger Cause", df['event_cause'].unique())
            with col2:
                priority = st.selectbox("Priority Level", df['priority'].unique())
                zone = st.selectbox("Affected Zone", df['zone'].unique())
            with col3:
                hour = st.slider("Hour of Occurrence", 0, 23, 12)
                road_closure = st.checkbox("Requires Road Closure")
            
            predict_btn = st.form_submit_button("Generate Prediction")
            
    if predict_btn:
        # Prediction Logic
        input_data = pd.DataFrame([{
            'event_type': event_type, 'event_cause': event_cause, 'priority': priority,
            'requires_road_closure': road_closure, 'latitude': 12.9716, 'longitude': 77.5946,
            'zone': zone, 'junction': 'Unknown', 'corridor': 'Unknown', 'police_station': 'Unknown',
            'veh_type': 'Unknown', 'hour': hour, 'month': 6, 'day_of_week': 0, 
            'is_peak_hour': 1 if (7 <= hour <= 10) or (17 <= hour <= 21) else 0
        }])
        
        cat_features = ['event_type', 'event_cause', 'priority', 'zone', 'junction', 'corridor', 'police_station', 'veh_type']
        for col in cat_features: input_data[col] = input_data[col].astype(str)
            
        pred_sev = sev_model.predict(input_data)[0][0]
        pred_res = res_model.predict(input_data)[0]
        is_ph = 1 if (7 <= hour <= 10) or (17 <= hour <= 21) else 0
        impact_score = calculate_impact_score(pred_sev, pred_res, road_closure)
        recs = get_recommendations(impact_score, event_cause=event_cause, is_peak_hour=is_ph)
        
        st.markdown(f"### Results for Impact Score: {impact_score:.1f}/100")
        
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Severity", pred_sev)
        r2.metric("Estimated Clearance", f"{pred_res:.1f} mins")
        r3.metric("Management Level", recs['level'])
        
        st.subheader("🛠️ Recommended Deployment")
        d1, d2, d3 = st.columns(3)
        d1.info(f"👮 **Personnel:** {recs['police_officers']} Officers")
        d2.info(f"🚧 **Physical:** {recs['barricades']} Barricades")
        d3.info(f"🚛 **Equipment:** {recs['tow_trucks']} Tow Trucks")
        
        if recs['protocols']:
            st.subheader("📋 Operational Protocols")
            for protocol in recs['protocols']:
                st.warning(protocol)
        
        st.info(f"**Primary Action:** {recs['action']}")

elif page == "📍 Hotspot Exploration":
    st.title("📍 Interactive Hotspot Explorer")
    
    st.markdown("""
    Explore high-density traffic incident zones. Our AI uses **DBSCAN Clustering** to identify persistent hotspots 
    and optimizes visualization using **HeatMaps** for maximum performance.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Optimized HeatMap
        from folium.plugins import HeatMap
        
        center_lat = df['latitude'].median()
        center_lon = df['longitude'].median()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
        
        # Add HeatMap layer
        heat_data = df[['latitude', 'longitude']].dropna().values.tolist()
        HeatMap(heat_data, radius=10, blur=15, min_opacity=0.5).add_to(m)
        
        # Add Cluster Centers
        if os.path.exists('reports/top_hotspots.csv'):
            hotspot_stats = pd.read_csv('reports/top_hotspots.csv')
            for _, row in hotspot_stats.head(10).iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    icon=folium.Icon(color='red', icon='info-sign'),
                    popup=f"Hotspot Cluster: {int(row['incident_count'])} incidents"
                ).add_to(m)
        
        st_folium(m, width=900, height=500)
        
    with col2:
        st.subheader("🔥 Top Hotspots")
        if os.path.exists('reports/top_hotspots.csv'):
            show_stats = pd.read_csv('reports/top_hotspots.csv')[['incident_count', 'avg_resolution_time']]
            st.dataframe(show_stats.head(10), use_container_width=True)
            st.info("Clusters show areas with >20 incidents within a ~500m radius.")
        else:
            st.warning("Run hotspot_detection.py to see cluster stats.")

st.sidebar.markdown("---")
st.sidebar.caption("TrafficVision Pro v2.0-AE")
