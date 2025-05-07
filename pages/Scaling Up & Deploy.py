import streamlit as st
import sys
import platform
import os
import uuid
import json
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------


if st.session_state.get("role") != "admin":
    # Enhanced error message with animation
    st.error("⚠️ Access Denied. Only Admins can access this Page.")
    
    # Load unauthorized access animation
    def load_lottie_url(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    
    unauthorized_animation = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_yzmugvgj.json")
    if unauthorized_animation:
        st_lottie(unauthorized_animation, height=400)
    
    st.stop()

# -----------------------------------------------------------------------------
# CUSTOM CSS FOR THE PAGE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700;900&display=swap');
      
      body {
          background: linear-gradient(135deg, #1e222e, #2d3047);
          font-family: 'Poppins', sans-serif;
          color: #E0E0E0;
      }
      
      .stApp {
          background: linear-gradient(135deg, #1e222e, #2d3047);
      }
      
      .main-title {
          text-align: center;
          font-size: 3.8rem;
          font-weight: 900;
          background: linear-gradient(90deg, #FF416C, #FF4B2B);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          margin-bottom: 1.5rem;
          padding: 20px 0;
          text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
      }
      
      .section-header {
          font-size: 2.2rem;
          font-weight: 700;
          margin-top: 2rem;
          margin-bottom: 1rem;
          background: linear-gradient(90deg, #FF416C, #FF4B2B);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          display: inline-block;
      }
      
      .subsection-header {
          font-size: 1.6rem;
          font-weight: 600;
          margin-top: 1.5rem;
          margin-bottom: 0.8rem;
          color: #FF9F76;
      }
      
      .card {
          background: rgba(40, 44, 52, 0.8);
          border-radius: 12px;
          padding: 1.5rem;
          margin: 1rem 0;
          box-shadow: 0 6px 12px rgba(0,0,0,0.2);
          transition: all 0.3s ease;
          border-left: 4px solid #FF416C;
      }
      
      .card:hover {
          transform: translateY(-5px);
          box-shadow: 0 12px 24px rgba(0,0,0,0.3);
      }
      
      .dockerfile-box, .k8s-yaml-box {
          background: #2E2E2E;
          border-radius: 8px;
          padding: 1.5rem;
          margin: 1rem 0;
          font-family: monospace;
          white-space: pre-wrap;
          word-wrap: break-word;
          box-shadow: inset 0 0 10px rgba(0,0,0,0.3);
          border: 1px solid #444;
      }
      
      .footer {
          text-align: center;
          font-size: 0.9rem;
          color: #999;
          margin-top: 2rem;
          border-top: 1px solid #444;
          padding-top: 1rem;
      }
      
      .icon {
          margin-right: 0.5rem;
          color: #FF416C;
      }
      
      .stButton > button {
          background: linear-gradient(90deg, #FF416C, #FF4B2B);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          font-weight: 600;
          border-radius: 8px;
          transition: all 0.3s ease;
      }
      
      .stButton > button:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 12px rgba(0,0,0,0.2);
      }
      
      .st-bq {
          background-color: rgba(255, 65, 108, 0.1);
          border-left-color: #FF416C;
          padding: 1rem;
      }
      
      .deployment-status {
          padding: 8px 16px;
          border-radius: 20px;
          font-weight: 500;
          display: inline-block;
          text-align: center;
      }
      
      .status-deployed {
          background-color: rgba(46, 213, 115, 0.2);
          color: #2ed573;
          border: 1px solid #2ed573;
      }
      
      .status-pending {
          background-color: rgba(255, 165, 0, 0.2);
          color: #ffa502;
          border: 1px solid #ffa502;
      }
      
      .status-failed {
          background-color: rgba(255, 71, 87, 0.2);
          color: #ff4757;
          border: 1px solid #ff4757;
      }
      
      .fancy-hr {
          border: 0;
          height: 1px;
          background-image: linear-gradient(to right, rgba(255, 65, 108, 0), rgba(255, 65, 108, 0.75), rgba(255, 65, 108, 0));
          margin: 2rem 0;
      }
      
      .code-annotation {
          background-color: rgba(255, 65, 108, 0.1);
          border-left: 3px solid #FF416C;
          padding: 0.5rem 1rem;
          margin: 0.5rem 0;
          font-size: 0.9rem;
      }
      
      /* Tab styling */
      .stTabs [data-baseweb="tab-list"] {
          gap: 8px;
      }
      
      .stTabs [data-baseweb="tab"] {
          background-color: #2E2E2E;
          border-radius: 6px 6px 0 0;
          padding: 10px 20px;
          border: none;
          color: white;
      }
      
      .stTabs [aria-selected="true"] {
          background-color: #FF416C !important;
          color: white !important;
      }
      
      /* Fancy loader */
      @keyframes pulse {
          0% { opacity: 0.4; }
          50% { opacity: 1; }
          100% { opacity: 0.4; }
      }
      
      .loader {
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 8px;
          padding: 20px;
      }
      
      .loader-dot {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background-color: #FF416C;
          animation: pulse 1.5s infinite;
      }
      
      .loader-dot:nth-child(2) {
          animation-delay: 0.3s;
      }
      
      .loader-dot:nth-child(3) {
          animation-delay: 0.6s;
      }
      
      /* Tooltip styling */
      .tooltip {
          position: relative;
          display: inline-block;
          border-bottom: 1px dotted #FF416C;
          cursor: help;
      }
      
      .tooltip .tooltiptext {
          visibility: hidden;
          width: 200px;
          background-color: #333;
          color: #fff;
          text-align: center;
          border-radius: 6px;
          padding: 5px;
          position: absolute;
          z-index: 1;
          bottom: 125%;
          left: 50%;
          margin-left: -100px;
          opacity: 0;
          transition: opacity 0.3s;
          font-size: 0.8rem;
      }
      
      .tooltip:hover .tooltiptext {
          visibility: visible;
          opacity: 1;
      }
      
      /* Resource usage gauges */
      .resource-gauge {
          margin: 1rem 0;
      }
      
      /* Cost estimator */
      .cost-estimator {
          background: rgba(255, 65, 108, 0.1);
          border-radius: 8px;
          padding: 1rem;
          margin-top: 1rem;
      }
      
      /* Cloud provider logos */
      .cloud-logo {
          max-height: 40px;
          margin-right: 10px;
          filter: grayscale(50%);
          transition: all 0.3s ease;
      }
      
      .cloud-logo:hover {
          filter: grayscale(0%);
          transform: scale(1.05);
      }
      
      .cloud-provider {
          display: flex;
          align-items: center;
          padding: 10px;
          border-radius: 8px;
          background: rgba(40, 44, 52, 0.5);
          margin-bottom: 8px;
          cursor: pointer;
          transition: all 0.3s ease;
      }
      
      .cloud-provider:hover {
          background: rgba(40, 44, 52, 0.8);
          transform: translateX(5px);
      }
      
      .cloud-provider.selected {
          background: rgba(255, 65, 108, 0.2);
          border-left: 3px solid #FF416C;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# Cloud provider logos
cloud_logos = {
    "Google Cloud": "https://www.gstatic.com/devrel-devsite/prod/v2210deb8920cd4a55bd580441aa58e7853afc04b39a9d9ac4198e1cd7fbe04ef/cloud/images/favicons/onecloud/favicon.ico",
    "AWS": "https://a0.awsstatic.com/libra-css/images/site/fav/favicon.ico",
    "Azure": "https://learn.microsoft.com/favicon.ico",
    "Docker Cloud": "https://www.docker.com/sites/default/files/d8/2019-07/Moby-logo.png",
    "Snowflake": "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/favicon.png"
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR MONGODB & CONFIGURATION
# -----------------------------------------------------------------------------
def get_mongo_client():
    connection_string = st.secrets.get("mongodb", {}).get("connection_string", "mongodb://localhost:27017")
    return MongoClient(connection_string)

def store_deployment_config(config_data):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        config_collection = db["deployment_config"]
        # Use a fixed ID to update the default configuration document.
        config_collection.update_one({"_id": "default_config"}, {"$set": config_data}, upsert=True)
        
        # Also store deployment history
        history_entry = config_data.copy()
        history_entry["_id"] = str(uuid.uuid4())  # Generate unique ID for history entry
        db["deployment_history"].insert_one(history_entry)
        
        return True
    except Exception as e:
        st.error(f"Error storing deployment configuration: {e}")
        return False
    finally:
        client.close()

def get_deployment_history():
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        history = list(db["deployment_history"].find().sort("timestamp", -1).limit(10))
        return history
    except Exception as e:
        st.error(f"Error retrieving deployment history: {e}")
        return []
    finally:
        client.close()

def get_current_config():
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        config = db["deployment_config"].find_one({"_id": "default_config"})
        return config or {}
    except Exception as e:
        st.error(f"Error retrieving current configuration: {e}")
        return {}
    finally:
        client.close()

# -----------------------------------------------------------------------------
# DOCKERFILE GENERATION
# -----------------------------------------------------------------------------
def generate_dockerfile(include_comments=True, python_version=None, with_caching=True):
    if not python_version:
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    else:
        py_version = python_version
        
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as req_file:
            req_content = req_file.read()
    else:
        req_content = "# No requirements.txt found. Please add your dependencies here."

    dockerfile_content = f"""# Generated Dockerfile for Civic Catalyst Application
FROM python:{py_version}-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
"""

    if include_comments:
        dockerfile_content += """
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    git \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*
"""
    else:
        dockerfile_content += """
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && apt-get clean && rm -rf /var/lib/apt/lists/*
"""

    # Add caching optimization if requested
    if with_caching:
        dockerfile_content += """
# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt
"""
    else:
        dockerfile_content += """
# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
"""

    dockerfile_content += """
# Copy the entire application
COPY . .

# Expose port (Streamlit default)
EXPOSE 8501

# Set healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    return dockerfile_content

def generate_docker_compose():
    compose_yaml = """version: '3.8'

services:
  streamlit:
    build: .
    container_name: civiccatalyst-app
    ports:
      - "8501:8501"
    volumes:
      - .:/app
    environment:
      - MONGO_URI=${MONGO_URI}
      - JWT_SECRET=${JWT_SECRET}
    restart: unless-stopped
    networks:
      - civic-network
    depends_on:
      - mongodb

  mongodb:
    image: mongo:latest
    container_name: civiccatalyst-mongodb
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_USERNAME}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_PASSWORD}
    restart: unless-stopped
    networks:
      - civic-network

  nginx:
    image: nginx:alpine
    container_name: civiccatalyst-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf:/etc/nginx/conf.d
      - ./nginx/certs:/etc/nginx/certs
    restart: unless-stopped
    networks:
      - civic-network
    depends_on:
      - streamlit

networks:
  civic-network:
    driver: bridge

volumes:
  mongo-data:
"""
    return compose_yaml

def generate_k8s_yaml(replicas=3, with_autoscaling=True, with_monitoring=True):
    # Generates a comprehensive Kubernetes deployment YAML for the app.
    app_name = "civiccatalyst-app"
    k8s_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}-deployment
  labels:
    app: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: {app_name}
"""

    if with_monitoring:
        k8s_yaml += """      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8501"
"""

    k8s_yaml += f"""    spec:
      containers:
      - name: {app_name}
        image: YOUR_IMAGE_NAME_HERE
        imagePullPolicy: Always
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        ports:
        - containerPort: 8501
          name: http
        env:
        - name: STREAMLIT_SERVER_ENABLECORS
          value: "false"
        - name: STREAMLIT_SERVER_HEADLESS
          value: "true"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 15
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
  labels:
    app: {app_name}
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
    protocol: TCP
    name: http
  selector:
    app: {app_name}
"""

    if with_autoscaling:
        k8s_yaml += f"""---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {app_name}-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {app_name}-deployment
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""

    if with_monitoring:
        k8s_yaml += f"""---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {app_name}-monitor
  labels:
    app: {app_name}
spec:
  selector:
    matchLabels:
      app: {app_name}
  endpoints:
  - port: http
    path: /_stcore/metrics
    interval: 15s
"""

    return k8s_yaml

# -----------------------------------------------------------------------------
# COST ESTIMATION
# -----------------------------------------------------------------------------
def estimate_deployment_cost(deployment_target, replicas, storage_gb, has_database, region=None):
    # Very simplified cost estimation model
    base_costs = {
        "Google Cloud": 0.031,  # per hour for e2-standard-2
        "AWS": 0.034,           # per hour for t3.medium
        "Azure": 0.029,         # per hour for B2s
        "Docker Cloud": 0.02,   # per hour for 2GB plan
        "Snowflake": 2.0        # per hour for standard edition
    }
    
    storage_costs = {
        "Google Cloud": 0.02,  # per GB per month
        "AWS": 0.023,          # per GB per month for gp2
        "Azure": 0.019,        # per GB per month for standard SSD
        "Docker Cloud": 0.01,  # per GB per month
        "Snowflake": 23        # per TB per month
    }
    
    db_costs = {
        "Google Cloud": 0.02,  # per hour for small instance
        "AWS": 0.022,          # per hour for small instance
        "Azure": 0.018,        # per hour for small instance
        "Docker Cloud": 0.01,  # per hour (shared)
        "Snowflake": 0         # included
    }
    
    # Regional adjustments (very simplified)
    region_multipliers = {
        "us-east": 1.0,
        "us-west": 1.05,
        "europe": 1.1,
        "asia": 1.2,
        "default": 1.0
    }
    
    region_mult = region_multipliers.get(region, region_multipliers["default"])
    
    # Calculate compute cost
    compute_cost = base_costs[deployment_target] * 24 * 30 * replicas * region_mult
    
    # Calculate storage cost
    storage_cost = storage_costs[deployment_target] * storage_gb * region_mult
    
    # Calculate database cost if applicable
    db_cost = db_costs[deployment_target] * 24 * 30 * region_mult if has_database else 0
    
    # Total monthly cost
    total_cost = compute_cost + storage_cost + db_cost
    
    return {
        "compute_cost": round(compute_cost, 2),
        "storage_cost": round(storage_cost, 2),
        "db_cost": round(db_cost, 2),
        "total_cost": round(total_cost, 2)
    }

# -----------------------------------------------------------------------------
# SIMULATION FUNCTIONS
# -----------------------------------------------------------------------------
def simulate_deployment(target, steps=5):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(steps):
        # Simulating deployment steps
        progress = (i + 1) / steps
        progress_bar.progress(progress)
        
        if i == 0:
            status_text.info(f"Preparing deployment to {target}...")
        elif i == 1:
            status_text.info(f"Building container image...")
        elif i == 2:
            status_text.info(f"Pushing image to registry...")
        elif i == 3:
            status_text.info(f"Applying Kubernetes manifests...")
        elif i == 4:
            status_text.success(f"Successfully deployed to {target}!")
        
        time.sleep(1)  # Simulate processing time
    
    return True

def generate_resource_metrics(days=7):
    """Generate simulated resource usage metrics for visualization"""
    date_range = pd.date_range(end=datetime.now(), periods=days)
    
    # CPU usage data with daily patterns
    cpu_data = []
    for date in date_range:
        # Create daily pattern with higher usage during work hours
        for hour in range(24):
            usage_factor = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12)) if 8 <= hour <= 20 else 0.3
            cpu_usage = max(10, min(90, 40 * usage_factor + np.random.randint(-5, 10)))
            
            cpu_data.append({
                'timestamp': date + timedelta(hours=hour),
                'value': cpu_usage,
                'metric': 'CPU Usage (%)'
            })
    
    # Memory usage with gradual increase
    memory_data = []
    base_memory = 30
    for date in date_range:
        for hour in range(24):
            daily_factor = date.dayofweek / 7  # Slight increase over the week
            memory_usage = base_memory + daily_factor * 10 + 5 * np.sin(hour / 4) + np.random.randint(-3, 5)
            memory_usage = max(20, min(85, memory_usage))
            
            memory_data.append({
                'timestamp': date + timedelta(hours=hour),
                'value': memory_usage,
                'metric': 'Memory Usage (%)'
            })
    
    # Network traffic
    network_data = []
    for date in date_range:
        for hour in range(24):
            # Higher traffic during work hours
            hour_factor = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12)) if 8 <= hour <= 20 else 0.2
            network_traffic = 50 * hour_factor + np.random.randint(-10, 20)
            network_traffic = max(5, min(150, network_traffic))
            
            network_data.append({
                'timestamp': date + timedelta(hours=hour),
                'value': network_traffic,
                'metric': 'Network Traffic (MB/s)'
            })
    
    # Combine all metrics
    all_data = pd.DataFrame(cpu_data + memory_data + network_data)
    
    return all_data

# -----------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------
def plot_resource_metrics(metrics_df):
    # Create a line chart for each metric
    fig = px.line(
        metrics_df, 
        x='timestamp', 
        y='value', 
        color='metric',
        title='Resource Usage Over Time',
        labels={'timestamp': 'Time', 'value': 'Value', 'metric': 'Metric'},
        line_shape='spline',
        color_discrete_map={
            'CPU Usage (%)': '#FF416C',
            'Memory Usage (%)': '#FF9F76',
            'Network Traffic (MB/s)': '#FFC371'
        }
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(40, 44, 52, 0.8)',
        paper_bgcolor='rgba(40, 44, 52, 0)',
        font_color='#E0E0E0',
        legend_title_font_color='#E0E0E0',
        xaxis_title_font_color='#E0E0E0',
        yaxis_title_font_color='#E0E0E0',
        title_font_color='#FF416C',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
    )
    
    return fig

def create_gauge_chart(value, title, min_val=0, max_val=100, threshold_good=30, threshold_warning=70):
    """Create a gauge chart for resource usage visualization"""
    if value <= threshold_good:
        color = "green"
    elif value <= threshold_warning:
        color = "orange"
    else:
        color = "red"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': '#E0E0E0', 'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': '#E0E0E0'},
            'bar': {'color': color},
            'bgcolor': 'rgba(40, 44, 52, 0.8)',
            'borderwidth': 2,
            'bordercolor': '#E0E0E0',
            'steps': [
                {'range': [min_val, threshold_good], 'color': 'rgba(46, 213, 115, 0.3)'},
                {'range': [threshold_good, threshold_warning], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [threshold_warning, max_val], 'color': 'rgba(255, 71, 87, 0.3)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'font': {'color': '#E0E0E0', 'size': 20}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig

# -----------------------------------------------------------------------------
# MAIN PAGE CONTENT WITH TABS
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'><i class='fas fa-rocket icon'></i>Scaling Up & Deployment Hub</div>", unsafe_allow_html=True)

# Create main navigation tabs
tabs = st.tabs([
    "📊 Dashboard", 
    "🚀 Deployment", 
    "📄 Configuration Files", 
    "💰 Cost Estimation",
    "📈 Monitoring",
    "📜 Deployment History"
])

# -----------------------------------------------------------------------------
# TAB 1: DASHBOARD
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown("<div class='section-header'>Deployment Dashboard</div>", unsafe_allow_html=True)
    
    # System information cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🖥️ System")
        st.markdown(f"**OS:** {platform.system()} {platform.release()}")
        st.markdown(f"**Python:** {platform.python_version()}")
        st.markdown(f"**Streamlit:** {st.__version__}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔄 Current Deployment")
        
        # Get current deployment data from database
        current_config = get_current_config()
        deployment_target = current_config.get("deployment_target", "Not deployed")
        deploy_date = current_config.get("timestamp", "N/A")
        
        if deploy_date != "N/A":
            deploy_date = datetime.fromisoformat(deploy_date).strftime("%Y-%m-%d %H:%M")
        
        st.markdown(f"**Target:** {deployment_target}")
        st.markdown(f"**Last Update:** {deploy_date}")
        st.markdown(f"**Status:** <span class='deployment-status status-deployed'>Active</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Health Check")
        
        # Simulate some health metrics
        response_time = round(np.random.uniform(0.1, 0.5), 2)
        uptime = "99.98%"
        last_error = (datetime.now() - timedelta(days=np.random.randint(1, 30))).strftime("%Y-%m-%d")
        
        st.markdown(f"**Response Time:** {response_time}s")
        st.markdown(f"**Uptime:** {uptime}")
        st.markdown(f"**Last Error:** {last_error}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Resource usage gauges
    st.markdown("<div class='subsection-header'>Resource Usage</div>", unsafe_allow_html=True)
    
    gauge_cols = st.columns(3)
    with gauge_cols[0]:
        cpu_usage = int(np.random.uniform(30, 60))
        st.plotly_chart(create_gauge_chart(cpu_usage, "CPU Usage (%)"), use_container_width=True)
    
    with gauge_cols[1]:
        memory_usage = int(np.random.uniform(40, 70))
        st.plotly_chart(create_gauge_chart(memory_usage, "Memory Usage (%)"), use_container_width=True)
    
    with gauge_cols[2]:
        disk_usage = int(np.random.uniform(20, 50))
        st.plotly_chart(create_gauge_chart(disk_usage, "Disk Usage (%)"), use_container_width=True)
    
    # Quick actions
    st.markdown("<div class='subsection-header'>Quick Actions</div>", unsafe_allow_html=True)
    
    quick_cols = st.columns(4)
    with quick_cols[0]:
        if st.button("🔄 Restart Service"):
            with st.spinner("Restarting service..."):
                time.sleep(2)
                st.success("Service restarted successfully!")
    
    with quick_cols[1]:
        if st.button("🔍 Run Diagnostics"):
            with st.spinner("Running diagnostics..."):
                time.sleep(3)
                st.success("All systems operational!")
    
    with quick_cols[2]:
        if st.button("🔒 Security Scan"):
            with st.spinner("Scanning for vulnerabilities..."):
                time.sleep(3)
                st.success("No security issues found!")
    
    with quick_cols[3]:
        if st.button("📦 Update Dependencies"):
            with st.spinner("Updating dependencies..."):
                time.sleep(3)
                st.success("All dependencies updated!")
    
    # Recent logs
    st.markdown("<div class='subsection-header'>Recent Logs</div>", unsafe_allow_html=True)
    
    # Generate some fake log entries
   # Keep your log types as is but normalize the weights
    log_types = ["INFO", "WARNING", "ERROR", "INFO", "INFO"]
    log_weights = [0.28, 0.08, 0.04, 0.28, 0.32]
    log_colors = {
        "INFO": "white",
        "WARNING": "orange",
        "ERROR": "red"
    }
    
    logs = []
    for i in range(10):
        log_time = datetime.now() - timedelta(minutes=np.random.randint(1, 60))
        log_type = np.random.choice(log_types, p=log_weights)
        
        if log_type == "INFO":
            messages = [
                "Application started successfully",
                "User authentication successful",
                "Database connection established",
                "API request completed",
                "Cache refreshed"
            ]
        elif log_type == "WARNING":
            messages = [
                "Slow database query detected",
                "Rate limit approaching threshold",
                "Memory usage above 70%",
                "Deprecated API usage detected",
                "Temporary network latency"
            ]
        else:  # ERROR
            messages = [
                "Failed to connect to database",
                "API request timeout",
                "Authentication failed",
                "Disk space running low",
                "Unexpected exception in module"
            ]
        
        log_message = np.random.choice(messages)
        logs.append({
            "timestamp": log_time,
            "type": log_type,
            "message": log_message
        })
    
    # Sort logs by timestamp (newest first)
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    
    # Display logs in a styled container
    st.markdown("""
    <div style='background: rgba(40, 44, 52, 0.8); border-radius: 8px; padding: 10px; font-family: monospace; max-height: 300px; overflow-y: auto;'>
    """, unsafe_allow_html=True)
    
    for log in logs:
        timestamp = log["timestamp"].strftime("%H:%M:%S")
        log_type = log["type"]
        message = log["message"]
        color = log_colors.get(log_type, "white")
        
        st.markdown(f"""
        <div style='margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;'>
            <span style='color: #888;'>{timestamp}</span> 
            <span style='color: {color}; font-weight: bold;'>[{log_type}]</span> 
            <span style='color: #E0E0E0;'>{message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: DEPLOYMENT CONFIGURATION
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("<div class='section-header'>Deployment Configuration</div>", unsafe_allow_html=True)
    
    # Get current configuration
    current_config = get_current_config()
    current_target = current_config.get("deployment_target", "Google Cloud")
    
    # Cloud provider selection with logos
    st.markdown("<div class='subsection-header'>Select Cloud Provider</div>", unsafe_allow_html=True)
    
    # Display cloud provider options as cards
    cloud_providers = ["Google Cloud", "AWS", "Azure", "Docker Cloud", "Snowflake"]
    
    # Create a custom grid of cloud providers
    cloud_cols1 = st.columns(3)
    cloud_cols2 = st.columns(2)
    
    # Widgets to store selection
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = current_target
    
    def select_provider(provider):
        st.session_state.selected_provider = provider
    
    # First row of providers
    for i, provider in enumerate(cloud_providers[:3]):
        with cloud_cols1[i]:
            selected = st.session_state.selected_provider == provider
            selected_class = "selected" if selected else ""
            logo_url = cloud_logos.get(provider, "")
            
            html = f"""
            <div class="cloud-provider {selected_class}" onclick="selectProvider('{provider}')">
                <img src="{logo_url}" class="cloud-logo" alt="{provider} Logo">
                <span>{provider}</span>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
    
    # Second row of providers
    for i, provider in enumerate(cloud_providers[3:]):
        with cloud_cols2[i]:
            selected = st.session_state.selected_provider == provider
            selected_class = "selected" if selected else ""
            logo_url = cloud_logos.get(provider, "")
            
            html = f"""
            <div class="cloud-provider {selected_class}" onclick="selectProvider('{provider}')">
                <img src="{logo_url}" class="cloud-logo" alt="{provider} Logo">
                <span>{provider}</span>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)
    
    # JavaScript to handle selection
    st.markdown("""
    <script>
    function selectProvider(provider) {
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: provider,
            dataType: 'str',
            componentInstance: 'provider_selection'
        }, '*');
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Create a hidden component to receive the selection
    provider_selection = st.empty()
    provider_selection.text_input("Provider Selection", 
                                 value=st.session_state.selected_provider, 
                                 key="provider_selection",
                                 label_visibility="collapsed")
    
    # Use the selected provider
    deployment_target = st.session_state.selected_provider
    
    # Provider-specific configuration
    st.markdown("<div class='subsection-header'>Provider Configuration</div>", unsafe_allow_html=True)
    
    with st.expander("Basic Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            app_name = st.text_input("Application Name", value="civiccatalyst-app")
            environment = st.selectbox("Environment", ["Development", "Staging", "Production"])
            region_options = {
                "Google Cloud": ["us-east1", "us-central1", "europe-west1", "asia-east1"],
                "AWS": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
                "Azure": ["eastus", "westus2", "westeurope", "southeastasia"],
                "Docker Cloud": ["default"],
                "Snowflake": ["us-east-1", "eu-central-1", "ap-southeast-1"]
            }
            region = st.selectbox("Region", region_options.get(deployment_target, ["default"]))
        
        with col2:
            replicas = st.number_input("Number of Replicas", min_value=1, max_value=10, value=3)
            with_db = st.checkbox("Include Database", value=True, key="with_db_checkbox")
            with_monitoring = st.checkbox("Enable Monitoring", value=True)
    
    # Provider-specific settings
    if deployment_target == "Google Cloud":
        with st.expander("Google Cloud Settings", expanded=True):
            gcloud_project = st.text_input("GCP Project ID", value=current_config.get("gcloud_project", ""))
            gcloud_credentials = st.text_area("Service Account JSON (paste here)", height=150, value=current_config.get("gcloud_credentials", ""))
            gke_cluster = st.text_input("GKE Cluster Name", value="civiccatalyst-cluster")
            use_cloud_run = st.checkbox("Use Cloud Run instead of GKE", value=False)
            use_cloud_sql = st.checkbox("Use Cloud SQL for Database", value=True)
    
    elif deployment_target == "AWS":
        with st.expander("AWS Settings", expanded=True):
            aws_access_key = st.text_input("AWS Access Key", value=current_config.get("aws_access_key", ""))
            aws_secret_key = st.text_input("AWS Secret Key", type="password", value=current_config.get("aws_secret_key", ""))
            aws_region = st.text_input("AWS Region", value=current_config.get("aws_region", "us-east-1"))
            eks_cluster = st.text_input("EKS Cluster Name", value="civiccatalyst-cluster")
            use_fargate = st.checkbox("Use AWS Fargate", value=False)
            use_rds = st.checkbox("Use RDS for Database", value=True)
    
    elif deployment_target == "Azure":
        with st.expander("Azure Settings", expanded=True):
            azure_subscription_id = st.text_input("Azure Subscription ID", value=current_config.get("azure_subscription_id", ""))
            azure_resource_group = st.text_input("Resource Group", value=current_config.get("azure_resource_group", ""))
            azure_credentials = st.text_area("Azure Credentials JSON (paste here)", height=150, value=current_config.get("azure_credentials", ""))
            aks_cluster = st.text_input("AKS Cluster Name", value="civiccatalyst-cluster")
            use_app_service = st.checkbox("Use App Service", value=False)
            use_cosmos_db = st.checkbox("Use Cosmos DB", value=False)
    
    elif deployment_target == "Docker Cloud":
        with st.expander("Docker Cloud Settings", expanded=True):
            docker_username = st.text_input("Docker Hub Username", value=current_config.get("docker_username", ""))
            docker_token = st.text_input("Docker Hub Token", type="password", value=current_config.get("docker_token", ""))
            docker_image = st.text_input("Docker Image Name", value=f"{docker_username}/civiccatalyst:latest" if docker_username else "civiccatalyst:latest")
            use_docker_compose = st.checkbox("Use Docker Compose", value=True, key="use_docker_compose_checkbox")
    
    elif deployment_target == "Snowflake":
        with st.expander("Snowflake Settings", expanded=True):
            snowflake_account = st.text_input("Snowflake Account", value=current_config.get("snowflake_account", ""))
            snowflake_user = st.text_input("Snowflake Username", value=current_config.get("snowflake_user", ""))
            snowflake_password = st.text_input("Snowflake Password", type="password", value=current_config.get("snowflake_password", ""))
            snowflake_warehouse = st.text_input("Snowflake Warehouse", value=current_config.get("snowflake_warehouse", ""))
            snowflake_database = st.text_input("Snowflake Database", value="CIVICCATALYST_DB")
    
    # Advanced Configuration
    with st.expander("Advanced Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_limit = st.slider("CPU Limit (cores)", min_value=0.25, max_value=4.0, value=0.5, step=0.25)
            memory_limit = st.slider("Memory Limit (GB)", min_value=0.5, max_value=8.0, value=1.0, step=0.5)
            storage_size = st.slider("Storage Size (GB)", min_value=5, max_value=100, value=20, step=5)
        
        with col2:
            enable_autoscaling = st.checkbox("Enable Autoscaling", value=True)
            if enable_autoscaling:
                min_replicas = st.number_input("Minimum Replicas", min_value=1, max_value=5, value=1)
                max_replicas = st.number_input("Maximum Replicas", min_value=min_replicas, max_value=20, value=10)
                cpu_threshold = st.slider("CPU Threshold for Scaling (%)", min_value=50, max_value=90, value=70)
            
            enable_ssl = st.checkbox("Enable SSL/TLS", value=True)
            custom_domain = st.text_input("Custom Domain (optional)", placeholder="app.yourdomain.com")
    
    # Save Configuration Button
    if st.button("Save Deployment Configuration", type="primary", use_container_width=True):
        # Build configuration object based on selected provider
        config_data = {
            "deployment_target": deployment_target,
            "app_name": app_name,
            "environment": environment,
            "region": region,
            "replicas": int(replicas),
            "with_db": with_db,
            "with_monitoring": with_monitoring,
            "cpu_limit": cpu_limit,
            "memory_limit": memory_limit,
            "storage_size": storage_size,
            "enable_autoscaling": enable_autoscaling,
            "enable_ssl": enable_ssl,
            "custom_domain": custom_domain,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if enable_autoscaling:
            config_data.update({
                "min_replicas": min_replicas,
                "max_replicas": max_replicas,
                "cpu_threshold": cpu_threshold
            })
        
        # Add provider-specific settings
        if deployment_target == "Google Cloud":
            config_data.update({
                "gcloud_project": gcloud_project,
                "gcloud_credentials": gcloud_credentials,
                "gke_cluster": gke_cluster,
                "use_cloud_run": use_cloud_run,
                "use_cloud_sql": use_cloud_sql
            })
        elif deployment_target == "AWS":
            config_data.update({
                "aws_access_key": aws_access_key,
                "aws_secret_key": aws_secret_key,
                "aws_region": aws_region,
                "eks_cluster": eks_cluster,
                "use_fargate": use_fargate,
                "use_rds": use_rds
            })
        elif deployment_target == "Azure":
            config_data.update({
                "azure_subscription_id": azure_subscription_id,
                "azure_resource_group": azure_resource_group,
                "azure_credentials": azure_credentials,
                "aks_cluster": aks_cluster,
                "use_app_service": use_app_service,
                "use_cosmos_db": use_cosmos_db
            })
        elif deployment_target == "Docker Cloud":
            config_data.update({
                "docker_username": docker_username,
                "docker_token": docker_token,
                "docker_image": docker_image,
                "use_docker_compose": use_docker_compose
            })
        elif deployment_target == "Snowflake":
            config_data.update({
                "snowflake_account": snowflake_account,
                "snowflake_user": snowflake_user,
                "snowflake_password": snowflake_password,
                "snowflake_warehouse": snowflake_warehouse,
                "snowflake_database": snowflake_database
            })
        
        # Store configuration in MongoDB
        if store_deployment_config(config_data):
            st.success("Configuration saved successfully!")
        else:
            st.error("Failed to save configuration.")
    
    # Deploy Button
    st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div style='background: rgba(255,65,108,0.1); border-radius: 8px; padding: 10px; margin-bottom: 10px;'>
            ⚠️ <b>Deployment Notice:</b> Deploying will create resources in your selected cloud provider that may incur costs.
            Make sure your credentials are correct and you have the necessary permissions.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚀 Deploy Now", use_container_width=True):
            # Check if we have the necessary credentials
            can_deploy = False
            missing_fields = []
            
            if deployment_target == "Google Cloud" and gcloud_project and gcloud_credentials:
                can_deploy = True
            elif deployment_target == "AWS" and aws_access_key and aws_secret_key:
                can_deploy = True
            elif deployment_target == "Azure" and azure_subscription_id and azure_credentials:
                can_deploy = True
            elif deployment_target == "Docker Cloud" and docker_username and docker_token:
                can_deploy = True
            elif deployment_target == "Snowflake" and snowflake_account and snowflake_user and snowflake_password:
                can_deploy = True
            else:
                # Identify missing fields
                if deployment_target == "Google Cloud":
                    if not gcloud_project: missing_fields.append("GCP Project ID")
                    if not gcloud_credentials: missing_fields.append("Service Account JSON")
                elif deployment_target == "AWS":
                    if not aws_access_key: missing_fields.append("AWS Access Key")
                    if not aws_secret_key: missing_fields.append("AWS Secret Key")
                elif deployment_target == "Azure":
                    if not azure_subscription_id: missing_fields.append("Azure Subscription ID")
                    if not azure_credentials: missing_fields.append("Azure Credentials")
                elif deployment_target == "Docker Cloud":
                    if not docker_username: missing_fields.append("Docker Hub Username")
                    if not docker_token: missing_fields.append("Docker Hub Token")
                elif deployment_target == "Snowflake":
                    if not snowflake_account: missing_fields.append("Snowflake Account")
                    if not snowflake_user: missing_fields.append("Snowflake Username")
                    if not snowflake_password: missing_fields.append("Snowflake Password")
            
            if can_deploy:
                # Simulate deployment process
                st.markdown("<div class='loader'><div class='loader-dot'></div><div class='loader-dot'></div><div class='loader-dot'></div></div>", unsafe_allow_html=True)
                
                success = simulate_deployment(deployment_target)
                
                if success:
                    st.balloons()
                    st.success(f"🎉 Successfully deployed to {deployment_target}!")
                    
                    # Show monitoring link
                    if deployment_target == "Google Cloud":
                        st.markdown(f"[View in Google Cloud Console](https://console.cloud.google.com/kubernetes/list?project={gcloud_project})")
                    elif deployment_target == "AWS":
                        st.markdown(f"[View in AWS Console](https://console.aws.amazon.com/eks/home?region={aws_region}#/clusters/{eks_cluster})")
                    elif deployment_target == "Azure":
                        st.markdown(f"[View in Azure Portal](https://portal.azure.com/#resource/subscriptions/{azure_subscription_id}/resourceGroups/{azure_resource_group}/providers/Microsoft.ContainerService/managedClusters/{aks_cluster})")
                    elif deployment_target == "Docker Cloud":
                        st.markdown("[View in Docker Hub](https://hub.docker.com/)")
                    elif deployment_target == "Snowflake":
                        st.markdown("[View in Snowflake Console](https://app.snowflake.com/)")
            else:
                missing = ", ".join(missing_fields)
                st.error(f"Missing required fields: {missing}")
                st.info("Please fill in all required fields before deploying.")

# -----------------------------------------------------------------------------
# TAB 3: CONFIGURATION FILES
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<div class='section-header'>Configuration Files</div>", unsafe_allow_html=True)
    st.write("Generate configuration files for your deployment. These files can be customized and used to deploy your application.")
    
    # Configuration options
    file_cols = st.columns(2)
    
    with file_cols[0]:
        python_version = st.selectbox("Python Version", ["3.9", "3.10", "3.11"], index=1)
        include_comments = st.checkbox("Include Comments", value=True, key="include_comments_checkbox")
        with_caching = st.checkbox("Optimize for Caching", value=True, key="with_caching_checkbox")
    
    with file_cols[1]:
        k8s_replicas = st.slider("Kubernetes Replicas", min_value=1, max_value=10, value=3)
        enable_k8s_autoscaling = st.checkbox("Enable Kubernetes Autoscaling", value=True, key="enable_k8s_autoscaling_checkbox")
        enable_monitoring = st.checkbox("Include Monitoring Configuration", value=True, key="enable_monitoring_checkbox")
    
    # Configuration file tabs
    config_tabs = st.tabs(["Dockerfile", "Docker Compose", "Kubernetes"])
    
    with config_tabs[0]:
        st.subheader("📄 Dockerfile")
        dockerfile_content = generate_dockerfile(include_comments, python_version, with_caching)
        
        # Add annotations to explain important parts
        st.markdown("<div class='code-annotation'>This Dockerfile is optimized for Streamlit applications with health checks and proper caching.</div>", unsafe_allow_html=True)
        
        st.code(dockerfile_content, language="dockerfile")
        st.download_button(
            label="📥 Download Dockerfile",
            data=dockerfile_content,
            file_name="Dockerfile",
            mime="text/plain"
        )
    
    with config_tabs[1]:
        st.subheader("📄 Docker Compose Configuration")
        docker_compose_content = generate_docker_compose()
        
        st.markdown("<div class='code-annotation'>This Docker Compose file includes Streamlit, MongoDB, and Nginx for a complete deployment stack.</div>", unsafe_allow_html=True)
        
        st.code(docker_compose_content, language="yaml")
        st.download_button(
            label="📥 Download docker-compose.yml",
            data=docker_compose_content,
            file_name="docker-compose.yml",
            mime="text/yaml"
        )
    
    with config_tabs[2]:
        st.subheader("📄 Kubernetes Deployment YAML")
        k8s_yaml_content = generate_k8s_yaml(k8s_replicas, enable_k8s_autoscaling, enable_monitoring)
        
        st.markdown("<div class='code-annotation'>This Kubernetes configuration includes deployment, service, autoscaling, and monitoring setup.</div>", unsafe_allow_html=True)
        
        st.code(k8s_yaml_content, language="yaml")
        st.download_button(
            label="📥 Download Kubernetes YAML",
            data=k8s_yaml_content,
            file_name="kubernetes-deployment.yaml",
            mime="text/yaml"
        )
    
    # Additional configuration generators
    st.markdown("<div class='subsection-header'>Additional Configuration Templates</div>", unsafe_allow_html=True)
    
    additional_configs = st.multiselect(
        "Select additional configuration files to generate",
        ["Nginx Config", "GitHub Actions CI/CD", "Terraform", "Helm Chart", "env file"]
    )
    
    if "Nginx Config" in additional_configs:
        nginx_conf = """server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://streamlit:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}"""
        
        st.subheader("Nginx Configuration")
        st.code(nginx_conf, language="nginx")
        st.download_button(
            label="📥 Download nginx.conf",
            data=nginx_conf,
            file_name="nginx.conf",
            mime="text/plain"
        )
    
    if "GitHub Actions CI/CD" in additional_configs:
        github_actions = """name: Build and Deploy

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        
    - name: Run tests
      run: |
        pytest
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ secrets.DOCKER_HUB_USERNAME }}/civiccatalyst:latest
        
  deploy:
    needs: build
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      
    - name: Set Kubernetes context
      uses: azure/k8s-set-context@v3
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
        
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f kubernetes-deployment.yaml
        kubectl rollout status deployment/civiccatalyst-app-deployment
"""
        
        st.subheader("GitHub Actions Workflow")
        st.code(github_actions, language="yaml")
        st.download_button(
            label="📥 Download github-workflow.yml",
            data=github_actions,
            file_name="github-workflow.yml",
            mime="text/yaml"
        )
    
    if "Terraform" in additional_configs:
        terraform_config = """provider "aws" {
  region = "us-east-1"
}

resource "aws_eks_cluster" "civiccatalyst" {
  name     = "civiccatalyst-cluster"
  role_arn = aws_iam_role.eks_cluster.arn

  vpc_config {
    subnet_ids = [aws_subnet.example1.id, aws_subnet.example2.id]
  }

  depends_on = [
    aws_iam_role_policy_attachment.example-AmazonEKSClusterPolicy,
  ]
}

resource "aws_eks_node_group" "civiccatalyst" {
  cluster_name    = aws_eks_cluster.civiccatalyst.name
  node_group_name = "civiccatalyst-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = [aws_subnet.example1.id, aws_subnet.example2.id]

  scaling_config {
    desired_size = 3
    max_size     = 5
    min_size     = 1
  }

  instance_types = ["t3.medium"]

  depends_on = [
    aws_iam_role_policy_attachment.example-AmazonEKSWorkerNodePolicy,
  ]
}

# IAM roles and other resources would be defined here

resource "aws_db_instance" "civiccatalyst_db" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "13.4"
  instance_class       = "db.t3.micro"
  name                 = "civiccatalyst"
  username             = "postgres"
  password             = "changeme"  # Use secrets manager in production
  parameter_group_name = "default.postgres13"
  skip_final_snapshot  = true
}

output "cluster_endpoint" {
  value = aws_eks_cluster.civiccatalyst.endpoint
}
"""
        
        st.subheader("Terraform Configuration")
        st.code(terraform_config, language="hcl")
        st.download_button(
            label="📥 Download main.tf",
            data=terraform_config,
            file_name="main.tf",
            mime="text/plain"
        )
    
    if "Helm Chart" in additional_configs:
        helm_values = """# Default values for civiccatalyst-chart.
# This is a YAML-formatted file.

replicaCount: 3

image:
  repository: your-registry/civiccatalyst
  tag: latest
  pullPolicy: Always

nameOverride: ""
fullnameOverride: ""

service:
  type: ClusterIP
  port: 80
  targetPort: 8501

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: app.example.com
      paths: ["/"]
  tls:
    - secretName: civiccatalyst-tls
      hosts:
        - app.example.com

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

mongodb:
  enabled: true
  auth:
    rootPassword: changeme  # Use secrets in production
    username: civiccatalyst
    password: changeme      # Use secrets in production
    database: civiccatalyst
  persistence:
    size: 8Gi

"""
        
        st.subheader("Helm Chart Values")
        st.code(helm_values, language="yaml")
        st.download_button(
            label="📥 Download values.yaml",
            data=helm_values,
            file_name="values.yaml",
            mime="text/yaml"
        )
    
    if "env file" in additional_configs:
        env_file = """# Environment variables for Civic Catalyst
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_THEME_BASE=dark

# MongoDB connection
MONGO_URI=mongodb://username:password@mongodb:27017/civiccatalyst?authSource=admin

# Security settings
JWT_SECRET=change_this_to_a_secure_random_string
PASSWORD_SALT=change_this_to_a_secure_random_string

# API keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# Logging
LOG_LEVEL=INFO
"""
        
        st.subheader("Environment Variables (.env)")
        st.code(env_file, language="bash")
        st.download_button(
            label="📥 Download .env",
            data=env_file,
            file_name=".env",
            mime="text/plain"
        )

# -----------------------------------------------------------------------------
# TAB 4: COST ESTIMATION
# -----------------------------------------------------------------------------
with tabs[3]:
    st.markdown("<div class='section-header'>Deployment Cost Estimation</div>", unsafe_allow_html=True)
    st.write("Estimate the monthly cost of your deployment based on selected provider and resources.")
    
    # Cost estimation inputs
    est_cols = st.columns(2)
    
    with est_cols[0]:
        est_provider = st.selectbox("Cloud Provider", cloud_providers, index=cloud_providers.index(deployment_target))
        est_region_map = {
            "Google Cloud": ["us-east", "us-west", "europe", "asia"],
            "AWS": ["us-east", "us-west", "europe", "asia"],
            "Azure": ["us-east", "us-west", "europe", "asia"],
            "Docker Cloud": ["default"],
            "Snowflake": ["us-east", "europe", "asia"]
        }
        est_region = st.selectbox("Region", est_region_map.get(est_provider, ["default"]))
    
    with est_cols[1]:
        est_replicas = st.number_input("Application Instances", min_value=1, max_value=20, value=3)
        est_storage = st.number_input("Storage (GB)", min_value=5, max_value=1000, value=50)
        est_db = st.checkbox("Include Database", value=True, key="est_db_checkbox")
    
    # Calculate estimated costs
    cost_estimates = estimate_deployment_cost(
        est_provider, 
        est_replicas, 
        est_storage, 
        est_db, 
        est_region
    )
    
    # Display cost breakdown
    st.markdown("<div class='subsection-header'>Monthly Cost Breakdown</div>", unsafe_allow_html=True)
    
    cost_cols = st.columns(4)
    
    with cost_cols[0]:
        st.metric("Compute", f"${cost_estimates['compute_cost']}")
    
    with cost_cols[1]:
        st.metric("Storage", f"${cost_estimates['storage_cost']}")
    
    with cost_cols[2]:
        st.metric("Database", f"${cost_estimates['db_cost']}")
    
    with cost_cols[3]:
        st.metric("Total", f"${cost_estimates['total_cost']}", delta=None)
    
    # Cost comparison across providers
    st.markdown("<div class='subsection-header'>Cost Comparison Across Providers</div>", unsafe_allow_html=True)
    
    # Calculate costs for all providers
    all_provider_costs = []
    for provider in cloud_providers:
        costs = estimate_deployment_cost(provider, est_replicas, est_storage, est_db, est_region)
        all_provider_costs.append({
            "Provider": provider,
            "Compute": costs["compute_cost"],
            "Storage": costs["storage_cost"],
            "Database": costs["db_cost"],
            "Total": costs["total_cost"]
        })
    
    # Create a DataFrame for visualization
    cost_df = pd.DataFrame(all_provider_costs)
    
    # Plot stacked bar chart
    fig = go.Figure()
    
    # Add bars for each cost component
    fig.add_trace(go.Bar(
        x=cost_df["Provider"],
        y=cost_df["Compute"],
        name="Compute",
        marker_color="#FF416C"
    ))
    
    fig.add_trace(go.Bar(
        x=cost_df["Provider"],
        y=cost_df["Storage"],
        name="Storage",
        marker_color="#FF9F76"
    ))
    
    fig.add_trace(go.Bar(
        x=cost_df["Provider"],
        y=cost_df["Database"],
        name="Database",
        marker_color="#FFC371"
    ))
    
    # Update layout
    fig.update_layout(
        barmode="stack",
        title="Monthly Cost Comparison (USD)",
        plot_bgcolor="rgba(40, 44, 52, 0.8)",
        paper_bgcolor="rgba(40, 44, 52, 0)",
        font_color="#E0E0E0",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
        height=400,
        margin=dict(t=60, b=80),
        xaxis=dict(
            title="Provider",
            titlefont_size=14,
            tickfont_size=12
        ),
        yaxis=dict(
            title="Cost (USD)",
            titlefont_size=14,
            tickfont_size=12,
            gridcolor="rgba(255, 255, 255, 0.1)"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost optimization tips
    st.markdown("<div class='subsection-header'>Cost Optimization Tips</div>", unsafe_allow_html=True)
    
    tips_cols = st.columns(3)
    
    with tips_cols[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 💰 Use Spot Instances")
        st.markdown("""
        Spot instances can reduce compute costs by up to 90%. Consider using spot instances for non-critical workloads or stateless applications.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tips_cols[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Right-size Resources")
        st.markdown("""
        Monitor resource utilization and adjust instance sizes accordingly. Many applications use less than 20% of provisioned resources.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tips_cols[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔄 Auto-scaling")
        st.markdown("""
        Implement auto-scaling to adjust resources based on demand. Scale down during off-peak hours to reduce costs.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ROI Calculator
    st.markdown("<div class='subsection-header'>ROI Calculator</div>", unsafe_allow_html=True)
    
    roi_cols = st.columns(2)
    
    with roi_cols[0]:
        monthly_users = st.number_input("Expected Monthly Users", min_value=100, max_value=1000000, value=10000, step=1000)
        revenue_per_user = st.number_input("Revenue per User ($)", min_value=0.01, max_value=100.0, value=0.50, step=0.1)
        monthly_cost = cost_estimates["total_cost"]
    
    with roi_cols[1]:
        monthly_revenue = monthly_users * revenue_per_user
        monthly_profit = monthly_revenue - monthly_cost
        annual_profit = monthly_profit * 12
        roi = (monthly_profit / monthly_cost) * 100 if monthly_cost > 0 else 0
        
        st.metric("Monthly Revenue", f"${monthly_revenue:.2f}")
        st.metric("Monthly Profit", f"${monthly_profit:.2f}")
        st.metric("Annual Profit", f"${annual_profit:.2f}")
        st.metric("ROI", f"{roi:.1f}%")

# -----------------------------------------------------------------------------
# TAB 5: MONITORING
# -----------------------------------------------------------------------------
with tabs[4]:
    st.markdown("<div class='section-header'>Resource Monitoring</div>", unsafe_allow_html=True)
    st.write("Monitor resource usage and performance metrics for your deployed application.")
    
    # Timeframe selector
    timeframe = st.selectbox("Timeframe", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"], index=1)
    
    if timeframe == "Last 24 Hours":
        days = 1
    elif timeframe == "Last 7 Days":
        days = 7
    else:
        days = 30
    
    # Generate simulated metrics data
    metrics_data = generate_resource_metrics(days)
    
    # Display resource usage trend chart
    st.markdown("<div class='subsection-header'>Resource Usage Trend</div>", unsafe_allow_html=True)
    fig = plot_resource_metrics(metrics_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display current resource usage
    st.markdown("<div class='subsection-header'>Current Resource Usage</div>", unsafe_allow_html=True)
    
    # Generate current metrics (use the latest values from our simulated data)
    latest_data = metrics_data.loc[metrics_data.groupby('metric')['timestamp'].idxmax()]
    latest_dict = {row['metric']: row['value'] for _, row in latest_data.iterrows()}
    
    gauge_cols = st.columns(3)
    
    with gauge_cols[0]:
        cpu_value = latest_dict.get('CPU Usage (%)', 45)
        st.plotly_chart(create_gauge_chart(cpu_value, "CPU Usage (%)"), use_container_width=True)
    
    with gauge_cols[1]:
        memory_value = latest_dict.get('Memory Usage (%)', 60)
        st.plotly_chart(create_gauge_chart(memory_value, "Memory Usage (%)"), use_container_width=True)
    
    with gauge_cols[2]:
        network_value = latest_dict.get('Network Traffic (MB/s)', 25)
        # Scale to percentage for the gauge (assuming 150 MB/s is max)
        network_percent = min(100, (network_value / 150) * 100)
        st.plotly_chart(create_gauge_chart(network_percent, f"Network: {network_value} MB/s", threshold_good=40, threshold_warning=80), use_container_width=True)
    
    # Application health metrics
    st.markdown("<div class='subsection-header'>Application Health</div>", unsafe_allow_html=True)
    
    health_cols = st.columns(4)
    
    with health_cols[0]:
        response_time = round(np.random.uniform(0.1, 0.9), 2)
        st.metric("Avg Response Time", f"{response_time}s", delta=-0.05, delta_color="normal")
    
    with health_cols[1]:
        uptime = 99.95 + np.random.uniform(-0.1, 0.05)
        st.metric("Uptime", f"{uptime:.2f}%", delta=0.01, delta_color="normal")
    
    with health_cols[2]:
        error_rate = round(np.random.uniform(0.01, 0.5), 2)
        st.metric("Error Rate", f"{error_rate}%", delta="-0.1%", delta_color="inverse")
    
    with health_cols[3]:
        active_users = int(np.random.uniform(50, 200))
        st.metric("Active Users", active_users, delta=12, delta_color="normal")
    
    # System health checks
    st.markdown("<div class='subsection-header'>System Health Checks</div>", unsafe_allow_html=True)
    
    # Simulate some health check data
    health_checks = [
        {"name": "API Gateway", "status": "Healthy", "latency": f"{np.random.uniform(0.05, 0.2):.2f}s"},
        {"name": "Database", "status": "Healthy", "latency": f"{np.random.uniform(0.1, 0.3):.2f}s"},
        {"name": "Authentication Service", "status": "Healthy", "latency": f"{np.random.uniform(0.1, 0.4):.2f}s"},
        {"name": "Object Storage", "status": "Healthy", "latency": f"{np.random.uniform(0.2, 0.5):.2f}s"},
        {"name": "Worker Nodes", "status": "Degraded", "latency": f"{np.random.uniform(0.5, 1.2):.2f}s"}
    ]
    
    # Display health checks in a fancy table
    st.markdown("""
    <table style="width: 100%; border-collapse: collapse; background: rgba(40, 44, 52, 0.8); border-radius: 8px; overflow: hidden;">
        <thead>
            <tr style="background: rgba(255, 65, 108, 0.2);">
                <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Service</th>
                <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Status</th>
                <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Latency</th>
                <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Action</th>
            </tr>
        </thead>
        <tbody>
    """, unsafe_allow_html=True)
    
    for check in health_checks:
        status_color = "green" if check["status"] == "Healthy" else "orange" if check["status"] == "Degraded" else "red"
        st.markdown(f"""
        <tr style="border-bottom: 1px solid #444;">
            <td style="padding: 12px;">{check["name"]}</td>
            <td style="padding: 12px;"><span style="color: {status_color}; font-weight: bold;">{check["status"]}</span></td>
            <td style="padding: 12px;">{check["latency"]}</td>
            <td style="padding: 12px;"><a href="#" style="color: #FF416C; text-decoration: none;">Details</a></td>
        </tr>
        """, unsafe_allow_html=True)
    
    st.markdown("</tbody></table>", unsafe_allow_html=True)
    
    # Alert settings
    st.markdown("<div class='subsection-header'>Alert Settings</div>", unsafe_allow_html=True)
    
    alert_cols = st.columns(2)
    
    with alert_cols[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔔 Alert Thresholds")
        
        cpu_alert = st.slider("CPU Alert Threshold (%)", min_value=50, max_value=95, value=80)
        memory_alert = st.slider("Memory Alert Threshold (%)", min_value=50, max_value=95, value=85)
        error_alert = st.slider("Error Rate Threshold (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        if st.button("Save Alert Thresholds"):
            st.success("Alert thresholds saved successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with alert_cols[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📧 Notification Channels")
        
        email_notify = st.checkbox("Email Notifications", value=True)
        if email_notify:
            email_recipients = st.text_input("Email Recipients (comma-separated)", value="admin@example.com")
        
        slack_notify = st.checkbox("Slack Notifications", value=True)
        if slack_notify:
            slack_channel = st.text_input("Slack Channel", value="#deployments")
        
        sms_notify = st.checkbox("SMS Notifications", value=False)
        if sms_notify:
            phone_numbers = st.text_input("Phone Numbers (comma-separated)")
        
        if st.button("Save Notification Settings"):
            st.success("Notification settings saved successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 6: DEPLOYMENT HISTORY
# -----------------------------------------------------------------------------
with tabs[5]:
    st.markdown("<div class='section-header'>Deployment History</div>", unsafe_allow_html=True)
    st.write("View and manage your deployment history, including rollbacks and configuration changes.")
    
    # Refresh button for deployment history
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_history"):
            st.rerun()
    
    # Load deployment history
    deployment_history = get_deployment_history()
    
    if not deployment_history:
        st.info("No deployment history found. Deploy your application to see history here.")
    else:
        # Display deployment history in a fancy table
        st.markdown("""
        <table style="width: 100%; border-collapse: collapse; background: rgba(40, 44, 52, 0.8); border-radius: 8px; overflow: hidden;">
            <thead>
                <tr style="background: rgba(255, 65, 108, 0.2);">
                    <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Timestamp</th>
                    <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Provider</th>
                    <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Environment</th>
                    <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Region</th>
                    <th style="padding: 12px; text-align: left; border-bottom: 1px solid #444;">Actions</th>
                </tr>
            </thead>
            <tbody>
        """, unsafe_allow_html=True)
        
        for i, deployment in enumerate(deployment_history):
            timestamp = deployment.get("timestamp", "N/A")
            if timestamp != "N/A":
                timestamp = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
            
            provider = deployment.get("deployment_target", "Unknown")
            environment = deployment.get("environment", "Production")
            region = deployment.get("region", "us-east")
            
            st.markdown(f"""
            <tr style="border-bottom: 1px solid #444;">
                <td style="padding: 12px;">{timestamp}</td>
                <td style="padding: 12px;">{provider}</td>
                <td style="padding: 12px;">{environment}</td>
                <td style="padding: 12px;">{region}</td>
                <td style="padding: 12px;">
                    <a href="#" onclick="viewDetails('{i}')" style="color: #FF416C; text-decoration: none; margin-right: 10px;">Details</a>
                    <a href="#" onclick="rollback('{i}')" style="color: #FF416C; text-decoration: none;">Rollback</a>
                </td>
            </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("</tbody></table>", unsafe_allow_html=True)
        
        # Add JavaScript for actions
        st.markdown("""
        <script>
        function viewDetails(index) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: index,
                dataType: 'str',
                componentInstance: 'view_deployment_details'
            }, '*');
        }
        
        function rollback(index) {
            if (confirm('Are you sure you want to roll back to this deployment?')) {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: index,
                    dataType: 'str',
                    componentInstance: 'rollback_deployment'
                }, '*');
            }
        }
        </script>
        """, unsafe_allow_html=True)
        
        # Create placeholders for the actions
        if 'view_deployment_details' not in st.session_state:
            st.session_state.view_deployment_details = None
        
        if 'rollback_deployment' not in st.session_state:
            st.session_state.rollback_deployment = None
        
        # Handle view details action
        if st.session_state.view_deployment_details is not None:
            index = int(st.session_state.view_deployment_details)
            st.session_state.view_deployment_details = None  # Reset
            
            if 0 <= index < len(deployment_history):
                deployment = deployment_history[index]
                
                st.markdown("<div class='subsection-header'>Deployment Details</div>", unsafe_allow_html=True)
                
                # Format the deployment details
                details_str = json.dumps(deployment, indent=2)
                
                # Show the details
                st.code(details_str, language="json")
        
        # Handle rollback action
        if st.session_state.rollback_deployment is not None:
            index = int(st.session_state.rollback_deployment)
            st.session_state.rollback_deployment = None  # Reset
            
            if 0 <= index < len(deployment_history):
                deployment = deployment_history[index]
                
                # Simulate rollback
                with st.spinner(f"Rolling back to deployment from {deployment.get('timestamp', 'N/A')}..."):
                    time.sleep(2)  # Simulate processing time
                    
                    # Store the rolled back configuration
                    if store_deployment_config(deployment):
                        st.success("Successfully rolled back to previous deployment!")
                    else:
                        st.error("Failed to roll back deployment. Please try again.")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 Civic Catalyst. All rights reserved.<br>Built with ❤️ by the Civic Catalyst Team</div>", unsafe_allow_html=True)