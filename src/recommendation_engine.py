import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_impact_score(severity, resolution_time, road_closure):
    """
    Calculate a traffic impact score from 0-100.
    """
    severity_map = {'Low': 25, 'Medium': 50, 'High': 75, 'Critical': 100}
    sev_score = severity_map.get(severity, 25)
    
    # Normalize resolution time (cap at 300 minutes for 100 score)
    res_score = min(resolution_time / 300 * 100, 100)
    
    closure_score = 100 if road_closure else 0
    
    # Weighted average
    total_score = (sev_score * 0.4) + (res_score * 0.3) + (closure_score * 0.3)
    return min(max(total_score, 0), 100)

def get_recommendations(impact_score, event_cause='Unknown', is_peak_hour=0):
    """
    Advanced recommendation engine with proportional scaling and situational protocols.
    """
    # 1. Proportional Scaling
    # Officers: Base 2, scales with impact score
    police_officers = int(2 + (impact_score / 100 * 18)) # Max ~20
    # Barricades: Base 1, scales with impact
    barricades = int(1 + (impact_score / 100 * 9))   # Max ~10
    # Tow Trucks: Only if high impact or vehicle related
    tow_trucks = int(impact_score / 40)             # Max ~2
    if 'breakdown' in event_cause.lower() or 'accident' in event_cause.lower():
        tow_trucks = max(tow_trucks, 1)

    # 2. Determine Level
    if impact_score <= 25: level = 'Low'
    elif impact_score <= 50: level = 'Medium'
    elif impact_score <= 75: level = 'High'
    else: level = 'Critical'

    # 3. Situational Protocols
    protocols = []
    if 'accident' in event_cause.lower():
        protocols.append("🚨 Emergency Medical: Request 1x Ambulance support.")
    if 'water_logging' in event_cause.lower() or 'tree_fall' in event_cause.lower():
        protocols.append("🛠️ Utility Response: Deploy Disaster Management Team.")
    if is_peak_hour:
        protocols.append("⏰ Peak Hour Alert: Prioritize public transit lanes.")
        police_officers = int(police_officers * 1.2) # 20% more during peak

    recommended_action = {
        'Low': 'Monitor and manage locally via CCTV.',
        'Medium': 'Deploy on-site personnel for manual regulation.',
        'High': 'Activate secondary diversion routes and active patrol.',
        'Critical': 'Full emergency protocol. Widespread diversions and radio alerts.'
    }.get(level)

    return {
        'level': level,
        'police_officers': police_officers,
        'barricades': barricades,
        'tow_trucks': tow_trucks,
        'action': recommended_action,
        'protocols': protocols
    }

if __name__ == "__main__":
    # Test
    score = calculate_impact_score('High', 150, True)
    recs = get_recommendations(score)
    print(f"Impact Score: {score}")
    print(f"Recommendations: {recs}")
