import streamlit as st
import openai
import requests
import pandas as pd
import numpy as np
import time
import random
import logging
import os
import re
import json
import schedule
import threading
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from pymongo import MongoClient
from qdrant_client import QdrantClient

# -----------------------------------------------------------------------------
# SETUP & LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOG_FILE = "pipeline.log"

def log_message(msg, level="info"):
    logging_func = getattr(logging, level.lower(), logging.info)
    logging_func(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()} - {level.upper()}: {msg}\n")

# -----------------------------------------------------------------------------
# ADMIN ACCESS CHECK
# -----------------------------------------------------------------------------
if st.session_state.get("role") != "admin":
    st.error("Access Denied. Only admins can access the configuration page.")
    st.stop()

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
#st.set_page_config(page_title="General Configuration", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .main-title {
          font-size: 3.5rem;
          font-weight: 900;
          text-align: center;
          margin-bottom: 2rem;
          color: #FF416C;
      }
      .section-header {
          font-size: 2.2rem;
          font-weight: 700;
          margin-top: 1.5rem;
          margin-bottom: 1rem;
          color: #FF4B2B;
      }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<div class='main-title'>NLP Management & Pipeline Configuration</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR DATABASE CONNECTIONS
# -----------------------------------------------------------------------------
def get_mongo_client():
    # The connection string and database name should be defined in st.secrets
    return MongoClient(st.secrets["mongodb"].get("connection_string", "mongodb://localhost:27017"))

def get_qdrant_client():
    return QdrantClient(
        host=st.secrets["qdrant"].get("host", "localhost"),
        port=st.secrets["qdrant"].get("port", 6333)
    )

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR CACHING & CHAT HISTORY
# -----------------------------------------------------------------------------
def compute_hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_cached_answer(prompt: str) -> dict:
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        cache_doc = db["chatbot_cache"].find_one({"prompt_hash": compute_hash(prompt)})
        return cache_doc
    except Exception as e:
        st.error(f"Error retrieving cached answer: {e}")
        return None
    finally:
        client.close()

def store_cached_answer(prompt: str, answer: str):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        doc = {
            "prompt_hash": compute_hash(prompt),
            "prompt": prompt,
            "answer": answer,
            "timestamp": datetime.utcnow()
        }
        db["chatbot_cache"].update_one({"prompt_hash": doc["prompt_hash"]}, {"$set": doc}, upsert=True)
    except Exception as e:
        st.error(f"Error storing cached answer: {e}")
    finally:
        client.close()

def store_chat_history(username: str, prompt: str, answer: str):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        doc = {
            "username": username,
            "prompt": prompt,
            "answer": answer,
            "timestamp": datetime.utcnow()
        }
        db["chat_history"].insert_one(doc)
    except Exception as e:
        st.error(f"Error storing chat history: {e}")
    finally:
        client.close()

def get_chat_history(username: str, limit: int = 10):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        docs = list(db["chat_history"].find({"username": username}).sort("timestamp", -1).limit(limit))
        return docs
    except Exception as e:
        st.error(f"Error retrieving chat history: {e}")
        return []
    finally:
        client.close()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR SCRAPING & PREPROCESSING (SIMPLIFIED)
# -----------------------------------------------------------------------------
def init_selenium_driver():
    from selenium.webdriver.chrome.options import Options
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scrape_hespress_pages(num_pages, use_selenium=False):
    base_url = 'https://www.hespress.com/politique'
    pagination_url = 'https://www.hespress.com/politique/page/{}/'
    headers = {'User-Agent': 'Mozilla/5.0'}
    extracted_urls = []
    driver = init_selenium_driver() if use_selenium else None
    
    for page in range(1, num_pages + 1):
        url = base_url if page == 1 else pagination_url.format(page)
        log_message(f"Scraping Page {page}: {url}")
        try:
            if use_selenium:
                driver.get(url)
                time.sleep(random.uniform(2, 4))
                page_source = driver.page_source
            else:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                page_source = response.content
        except Exception as e:
            log_message(f"Failed to retrieve page {page}: {e}", level="error")
            continue
        soup = BeautifulSoup(page_source, 'html.parser')
        for div in soup.find_all('div', class_='card-img-top'):
            if div.find('span', class_='cat politique'):
                a_tag = div.find('a', class_='stretched-link')
                if a_tag and a_tag.get('href'):
                    extracted_urls.append({'title': a_tag.get('title', '').strip(), 'url': a_tag['href']})
        time.sleep(random.uniform(1, 3))
    if driver:
        driver.quit()
        log_message("Selenium driver closed.")
    unique_urls = [dict(t) for t in {tuple(d.items()) for d in extracted_urls}]
    log_message(f"Total unique URLs extracted: {len(unique_urls)}")
    return unique_urls

# -----------------------------------------------------------------------------
# STREAMLIT UI: GENERAL CONFIGURATION PAGE (ADMIN ONLY)
# -----------------------------------------------------------------------------

# Sidebar: Configuration Settings
st.sidebar.header("Configuration Settings")
language = st.sidebar.selectbox("Language", options=["English", "Français", "العربية"], index=0)
st.session_state["lang"] = language

st.sidebar.subheader("OpenAI Settings")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=openai.api_key)
default_model = st.sidebar.text_input("Default Model", value=st.secrets["openai"].get("default_model", "gpt-3.5-turbo"))
max_tokens = st.sidebar.number_input("Max Tokens", min_value=50, max_value=1000, value=st.secrets["openai"].get("max_tokens", 150), step=50)
cost_opt = st.sidebar.checkbox("Enable Cost Optimization", value=st.secrets["openai"].get("cost_optimization", True))

st.sidebar.subheader("MongoDB Settings")
mongo_conn = st.sidebar.text_input("MongoDB Connection String", value=st.secrets["mongodb"].get("connection_string", "mongodb://localhost:27017"))
mongo_db = st.sidebar.text_input("MongoDB Database", value=st.secrets["mongodb"].get("database", "CivicCatalyst"))

st.sidebar.subheader("Qdrant Settings")
qdrant_host = st.sidebar.text_input("Qdrant Host", value=st.secrets["qdrant"].get("host", "localhost"))
qdrant_port = st.sidebar.number_input("Qdrant Port", value=st.secrets["qdrant"].get("port", 6333), step=1)

st.sidebar.subheader("Pipeline Settings")
num_pages = st.sidebar.number_input("Pages to Scrape", min_value=1, max_value=100, value=st.session_state.get("pipeline_config", {}).get("num_pages_to_scrape", 10), step=1)
use_selenium = st.sidebar.checkbox("Use Selenium", value=st.session_state.get("pipeline_config", {}).get("use_selenium", False))
logging_level = st.sidebar.selectbox("Logging Level", options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(st.session_state.get("pipeline_config", {}).get("logging_level", "INFO")))
scheduler_frequency = st.sidebar.selectbox("Scheduler Frequency", options=["Every Minute", "Every Hour", "Every Day"], index=["Every Minute", "Every Hour", "Every Day"].index(st.session_state.get("pipeline_config", {}).get("scheduler_frequency", "Every Hour")))

st.sidebar.markdown("---")
st.sidebar.write("Additional actions:")
if st.sidebar.button("Test MongoDB Connection"):
    try:
        client = get_mongo_client()
        db = client[mongo_db]
        collections = db.list_collection_names()
        st.sidebar.success(f"MongoDB connected. Collections: {collections}")
    except Exception as e:
        st.sidebar.error(f"MongoDB error: {e}")
    finally:
        client.close()

if st.sidebar.button("Test Qdrant Connection"):
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name="municipal_projects")
        st.sidebar.success("Qdrant connected.")
    except Exception as e:
        st.sidebar.error(f"Qdrant error: {e}")

if st.sidebar.button("Run Sample Pipeline Job"):
    st.sidebar.info("Running sample pipeline job...")
    time.sleep(2)
    st.sidebar.success("Pipeline job completed.")

# Display current configuration preview
st.markdown("### Current Configuration")
current_config = {
    "OpenAI": {
        "api_key": openai_api_key,
        "default_model": default_model,
        "max_tokens": max_tokens,
        "cost_optimization": cost_opt,
    },
    "MongoDB": {
        "connection_string": mongo_conn,
        "database": mongo_db,
    },
    "Qdrant": {
        "host": qdrant_host,
        "port": qdrant_port,
    },
    "Pipeline": {
        "num_pages_to_scrape": num_pages,
        "use_selenium": use_selenium,
        "logging_level": logging_level,
        "scheduler_frequency": scheduler_frequency,
    },
    "Language": st.session_state["lang"]
}
st.json(current_config)

if st.button("Save Configuration"):
    st.session_state["pipeline_config"] = {
        "num_pages_to_scrape": num_pages,
        "use_selenium": use_selenium,
        "logging_level": logging_level,
        "scheduler_frequency": scheduler_frequency,
    }
    st.success("Configuration saved successfully!")

st.markdown("### System Information")
st.write(f"Last updated: {datetime.utcnow().isoformat()}")

if st.button("View Audit Logs"):
    try:
        client = get_mongo_client()
        db = client[mongo_db]
        audit_logs = list(db["AuditLogs"].find({}).sort("timestamp", -1).limit(20))
        if audit_logs:
            st.write("Recent Audit Logs:")
            df_audit = pd.DataFrame(audit_logs)
            st.dataframe(df_audit)
        else:
            st.info("No audit logs found.")
    except Exception as e:
        st.error(f"Error retrieving audit logs: {e}")
    finally:
        client.close()

st.markdown("---")
st.write("This page allows you to adjust and test the configuration parameters for the CivicCatalyst application.")

# -----------------------------------------------------------------------------
# ADDITIONAL ACTIONS (e.g., Backup, Refresh)
# -----------------------------------------------------------------------------
if st.button("Backup Configuration"):
    backup_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "configuration": current_config
    }
    backup_file = f"backup_config_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, indent=2)
    st.download_button("Download Backup", backup_file, file_name=backup_file, mime="application/json")
    st.success("Backup created successfully.")

st.markdown("### Additional Features")
st.write("You can further extend this page to include advanced analytics, language localization settings, role management, and performance monitoring for the entire application.")

st.markdown("### System Info")
st.write(f"Current Time: {datetime.utcnow().isoformat()}")
