import streamlit as st
import sys
import platform
import os
import uuid
from datetime import datetime
from pymongo import MongoClient
import json

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
#st.set_page_config(page_title="Scaling Up & Deployment", layout="wide", initial_sidebar_state="expanded")
if st.session_state.get("role") != "admin":
    st.error("Access Denied. Only Admins can access this Page.")
    st.stop()
# -----------------------------------------------------------------------------
# CUSTOM CSS FOR THE PAGE
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      body {
          background: #1e1e1e;
          font-family: 'Helvetica Neue', sans-serif;
          color: #E0E0E0;
      }
      .stApp {
          background-color: #1e1e1e;
      }
      .main-title {
          text-align: center;
          font-size: 3.5rem;
          font-weight: 900;
          color: #FF416C;
          margin-bottom: 1.5rem;
      }
      .section-header {
          font-size: 2.2rem;
          font-weight: 700;
          margin-top: 2rem;
          margin-bottom: 1rem;
          color: #FF4B2B;
      }
      .dockerfile-box, .k8s-yaml-box {
          background: #2E2E2E;
          border-radius: 8px;
          padding: 1.5rem;
          margin: 1rem 0;
          font-family: monospace;
          white-space: pre-wrap;
          word-wrap: break-word;
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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'><i class='fas fa-rocket icon'></i>Scaling Up & Deployment</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR MONGODB & CONFIGURATION
# -----------------------------------------------------------------------------
def get_mongo_client():
    connection_string = st.secrets["mongodb"].get("connection_string", "mongodb://localhost:27017")
    return MongoClient(connection_string)

def store_deployment_config(config_data):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        config_collection = db["deployment_config"]
        # Use a fixed ID to update the default configuration document.
        config_collection.update_one({"_id": "default_config"}, {"$set": config_data}, upsert=True)
        return True
    except Exception as e:
        st.error(f"Error storing deployment configuration: {e}")
        return False
    finally:
        client.close()

# -----------------------------------------------------------------------------
# DOCKERFILE GENERATION
# -----------------------------------------------------------------------------
def generate_dockerfile():
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as req_file:
            req_content = req_file.read()
    else:
        req_content = "# No requirements.txt found. Please add your dependencies here."

    dockerfile_content = f"""# Generated Dockerfile
FROM python:{py_version}-slim

# Set work directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire application
COPY . .

# Expose port (Streamlit default)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
"""
    return dockerfile_content

def generate_k8s_yaml():
    # Generates a simple Kubernetes deployment YAML for the app.
    app_name = "civiccatalyst-app"
    k8s_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: {app_name}
        image: YOUR_IMAGE_NAME_HERE
        ports:
        - containerPort: 8501
        env:
        - name: STREAMLIT_SERVER_ENABLECORS
          value: "false"
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: {app_name}
"""
    return k8s_yaml

# -----------------------------------------------------------------------------
# SIDEBAR: DEPLOYMENT OPTIONS
# -----------------------------------------------------------------------------
st.sidebar.header("Deployment Settings")

# Extended deployment targets
deployment_target = st.sidebar.selectbox("Select Deployment Target", 
                                         ["Google Cloud", "Docker Cloud", "Snowflake", "AWS", "Azure"])

# Collect credentials & connection options based on selection
if deployment_target == "Google Cloud":
    st.sidebar.subheader("Google Cloud Settings")
    gcloud_project = st.sidebar.text_input("GCP Project ID")
    gcloud_credentials = st.sidebar.text_area("Service Account JSON (paste here)", height=200)
elif deployment_target == "Docker Cloud":
    st.sidebar.subheader("Docker Cloud Settings")
    docker_username = st.sidebar.text_input("Docker Hub Username")
    docker_token = st.sidebar.text_input("Docker Hub Token", type="password")
elif deployment_target == "Snowflake":
    st.sidebar.subheader("Snowflake Settings")
    snowflake_account = st.sidebar.text_input("Snowflake Account")
    snowflake_user = st.sidebar.text_input("Snowflake Username")
    snowflake_password = st.sidebar.text_input("Snowflake Password", type="password")
    snowflake_warehouse = st.sidebar.text_input("Snowflake Warehouse")
elif deployment_target == "AWS":
    st.sidebar.subheader("AWS Settings")
    aws_access_key = st.sidebar.text_input("AWS Access Key")
    aws_secret_key = st.sidebar.text_input("AWS Secret Key", type="password")
    aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")
elif deployment_target == "Azure":
    st.sidebar.subheader("Azure Settings")
    azure_subscription_id = st.sidebar.text_input("Azure Subscription ID")
    azure_resource_group = st.sidebar.text_input("Resource Group")
    azure_credentials = st.sidebar.text_area("Azure Credentials JSON (paste here)", height=200)

# Option to save configuration
if st.sidebar.button("Save Deployment Configuration"):
    config_data = {"deployment_target": deployment_target, "timestamp": datetime.utcnow().isoformat()}
    if deployment_target == "Google Cloud":
        config_data.update({"gcloud_project": gcloud_project, "gcloud_credentials": gcloud_credentials})
    elif deployment_target == "Docker Cloud":
        config_data.update({"docker_username": docker_username, "docker_token": docker_token})
    elif deployment_target == "Snowflake":
        config_data.update({
            "snowflake_account": snowflake_account,
            "snowflake_user": snowflake_user,
            "snowflake_password": snowflake_password,
            "snowflake_warehouse": snowflake_warehouse
        })
    elif deployment_target == "AWS":
        config_data.update({
            "aws_access_key": aws_access_key,
            "aws_secret_key": aws_secret_key,
            "aws_region": aws_region
        })
    elif deployment_target == "Azure":
        config_data.update({
            "azure_subscription_id": azure_subscription_id,
            "azure_resource_group": azure_resource_group,
            "azure_credentials": azure_credentials
        })
    if store_deployment_config(config_data):
        st.sidebar.success("Configuration saved successfully!")

# -----------------------------------------------------------------------------
# MAIN PAGE CONTENT: DOCKERFILE & KUBERNETES YAML GENERATION
# -----------------------------------------------------------------------------
st.subheader("Dockerfile Generation")
st.write("The Dockerfile below is generated based on your current Python version and the contents of your requirements.txt file.")
dockerfile_content = generate_dockerfile()
st.code(dockerfile_content, language="docker")
st.download_button(
    label="Download Dockerfile",
    data=dockerfile_content,
    file_name="Dockerfile",
    mime="text/plain"
)

st.subheader("Kubernetes Deployment YAML")
st.write("Generate a basic Kubernetes YAML file for deploying your app. Modify the image name as needed.")
k8s_yaml_content = generate_k8s_yaml()
st.code(k8s_yaml_content, language="yaml")
st.download_button(
    label="Download Kubernetes YAML",
    data=k8s_yaml_content,
    file_name="deployment.yaml",
    mime="text/yaml"
)

# -----------------------------------------------------------------------------
# DEPLOYMENT ACTIONS
# -----------------------------------------------------------------------------
st.subheader("Deploy Your App")
st.write("Click the button below to deploy your app. Credentials must be provided to proceed.")

if st.button("Deploy Now"):
    # Simulated deployment integration logic
    if deployment_target == "Google Cloud":
        if not gcloud_project or not gcloud_credentials:
            st.error("Please provide your Google Cloud credentials.")
        else:
            st.success("Deployment to Google Cloud initiated!")
            st.info("Redirecting to Google Cloud Console...")
            st.markdown("[Go to Google Cloud Console](https://console.cloud.google.com/)")
    elif deployment_target == "Docker Cloud":
        if not docker_username or not docker_token:
            st.error("Please provide your Docker Cloud credentials.")
        else:
            st.success("Deployment to Docker Cloud initiated!")
            st.info("Visit Docker Hub to manage your container.")
            st.markdown("[Go to Docker Hub](https://hub.docker.com/)")
    elif deployment_target == "Snowflake":
        if not (snowflake_account and snowflake_user and snowflake_password and snowflake_warehouse):
            st.error("Please provide your complete Snowflake credentials.")
        else:
            st.success("Deployment to Snowflake initiated!")
            st.info("Visit Snowflake to manage your data pipelines.")
            st.markdown("[Go to Snowflake](https://app.snowflake.com/)")
    elif deployment_target == "AWS":
        if not (aws_access_key and aws_secret_key and aws_region):
            st.error("Please provide your complete AWS credentials.")
        else:
            st.success("Deployment to AWS initiated!")
            st.info("Visit AWS Console to manage your services.")
            st.markdown("[Go to AWS Console](https://aws.amazon.com/console/)")
    elif deployment_target == "Azure":
        if not (azure_subscription_id and azure_resource_group and azure_credentials):
            st.error("Please provide your complete Azure credentials.")
        else:
            st.success("Deployment to Azure initiated!")
            st.info("Visit Azure Portal to manage your resources.")
            st.markdown("[Go to Azure Portal](https://portal.azure.com/)")
    else:
        st.error("Unsupported deployment target.")

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<div class='footer'>© 2025 Civic Catalyst. All rights reserved.</div>", unsafe_allow_html=True)
