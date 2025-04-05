import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

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
# OPENAI CONFIGURATION (Assume API key is stored in st.secrets)
# -----------------------------------------------------------------------------
openai.api_key = st.secrets["openai"]["api_key"]

# -----------------------------------------------------------------------------
# ACCESS CONTROL: Admins Only
# -----------------------------------------------------------------------------
if st.session_state.get("role") != "admin":
    st.error("Access Denied. Only admins can access the configuration page.")
    st.stop()

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="NLP Management", layout="wide", initial_sidebar_state="expanded")
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
# HELPER FUNCTIONS FOR DATABASE & QDRANT CONNECTIONS
# -----------------------------------------------------------------------------
def get_mongo_client():
    return MongoClient(st.secrets["mongodb"].get("connection_string", "mongodb://localhost:27017"))

def get_qdrant_client():
    return QdrantClient(
        host=st.secrets["qdrant"].get("host", "localhost"),
        port=st.secrets["qdrant"].get("port", 6333)
    )

def load_qdrant_documents(collection_name: str, vector_dim: int):
    client = get_qdrant_client()
    all_docs = []
    points, next_page_offset = client.scroll(collection_name=collection_name, limit=100)
    all_docs.extend([pt.payload for pt in points])
    while next_page_offset is not None:
        points, next_page_offset = client.scroll(collection_name=collection_name, offset=next_page_offset, limit=100)
        all_docs.extend([pt.payload for pt in points])
    client.close()
    return all_docs

def search_qdrant(collection_name: str, query_vector: list, top: int = 5):
    client = get_qdrant_client()
    results = client.search(collection_name=collection_name, query_vector=query_vector, limit=top, with_payload=True)
    client.close()
    return results

def compute_hash(text: str) -> str:
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
        doc = {"prompt_hash": compute_hash(prompt), "prompt": prompt, "answer": answer, "timestamp": datetime.utcnow()}
        db["chatbot_cache"].update_one({"prompt_hash": doc["prompt_hash"]}, {"$set": doc}, upsert=True)
    except Exception as e:
        st.error(f"Error storing cached answer: {e}")
    finally:
        client.close()

def store_chat_history(username: str, prompt: str, answer: str):
    try:
        client = get_mongo_client()
        db = client["CivicCatalyst"]
        doc = {"username": username, "prompt": prompt, "answer": answer, "timestamp": datetime.utcnow()}
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
# HELPER FUNCTIONS FOR SCRAPING & PREPROCESSING
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
                    log_message(f"Extracted URL: {a_tag['href']}")
        time.sleep(random.uniform(1, 3))
    if driver:
        driver.quit()
        log_message("Selenium driver closed.")
    unique_urls = [dict(t) for t in {tuple(d.items()) for d in extracted_urls}]
    log_message(f"Total unique URLs extracted: {len(unique_urls)}")
    return unique_urls

def extract_article_details(url, use_selenium=False):
    details = {"title": None, "publication_date": None, "content": None, "images": []}
    try:
        if use_selenium:
            driver = init_selenium_driver()
            driver.get(url)
            time.sleep(2)
            page_source = driver.page_source
            driver.quit()
        else:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            page_source = response.content
        soup = BeautifulSoup(page_source, 'html.parser')
        details["title"] = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Untitled"
        pub_date_elem = soup.find('time')
        if pub_date_elem:
            details["publication_date"] = pub_date_elem.get_text(strip=True)
        content_div = soup.find('div', class_='content-article') or soup.find('div', class_='article-content')
        if content_div:
            paragraphs = content_div.find_all('p')
            details["content"] = "\n".join([para.get_text(strip=True) for para in paragraphs])
        images = content_div.find_all('img') if content_div else []
        details["images"] = [img['src'] for img in images if img.get('src')]
    except Exception as e:
        log_message(f"Error extracting details from {url}: {e}", level="error")
    return details

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    return text.strip()

def run_scraping_pipeline(num_pages, use_selenium=False):
    log_message("Starting scraping pipeline...")
    urls = scrape_hespress_pages(num_pages, use_selenium)
    log_message("Scraping URLs completed.")
    articles = []
    for item in urls:
        details = extract_article_details(item["url"], use_selenium)
        if details["content"]:
            details["content_clean"] = preprocess_text(details["content"])
        else:
            details["content_clean"] = ""
        details["url"] = item["url"]
        details["scraped_title"] = item["title"]
        articles.append(details)
    df_articles = pd.DataFrame(articles)
    output_csv = f"hespress_articles_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df_articles.to_csv(output_csv, index=False, encoding='utf-8-sig')
    log_message(f"Scraping pipeline completed. Data saved to {output_csv}")
    return df_articles, output_csv

def run_preprocessing_pipeline(input_csv):
    try:
        df = pd.read_csv(input_csv, encoding='utf-8-sig')
        df["word_count"] = df["content_clean"].apply(lambda x: len(x.split()))
        log_message(f"Preprocessing completed on {len(df)} records.")
        return df
    except Exception as e:
        log_message(f"Error in preprocessing pipeline: {e}", level="error")
        return pd.DataFrame()

def scheduled_pipeline():
    num_pages = st.session_state.get("num_pages_to_scrape", 10)
    use_selenium = st.session_state.get("use_selenium", False)
    df, csv_file = run_scraping_pipeline(num_pages, use_selenium)
    preprocessed_df = run_preprocessing_pipeline(csv_file)
    st.success("Pipeline run completed.")
    return preprocessed_df

# -----------------------------------------------------------------------------
# STREAMLIT UI: TABS FOR DIFFERENT PIPELINE FUNCTIONS
# -----------------------------------------------------------------------------
tabs = st.tabs(["Scraping", "Preprocessing", "Scheduling", "Logs", "Configuration", "NLP Test", "Additional Pipeline Actions"])

# -----------------------------------------------------------------------------
# TAB 1: Scraping
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Hespress Scraping")
    st.write("Configure parameters to scrape political articles from Hespress.")
    num_pages = st.number_input("Number of Pages to Scrape", min_value=1, max_value=100, value=10, step=1)
    method = st.radio("Scraping Method", options=["Requests/BeautifulSoup", "Selenium"], index=0)
    use_selenium = True if method == "Selenium" else False
    if st.button("Run Scraping Pipeline"):
        df_articles, output_csv = run_scraping_pipeline(num_pages, use_selenium)
        st.success(f"Scraping completed. Data saved to **{output_csv}**.")
        st.dataframe(df_articles)

# -----------------------------------------------------------------------------
# TAB 2: Preprocessing
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Preprocessing Pipeline")
    uploaded_csv = st.file_uploader("Upload CSV from Scraping (or use the last saved file)", type=["csv"])
    if uploaded_csv:
        df_uploaded = pd.read_csv(uploaded_csv, encoding='utf-8-sig')
        st.write("Original Data:")
        st.dataframe(df_uploaded.head())
        df_processed = run_preprocessing_pipeline(uploaded_csv)
        st.write("Preprocessed Data:")
        st.dataframe(df_processed.head())
        csv_processed = f"preprocessed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        df_processed.to_csv(csv_processed, index=False, encoding='utf-8-sig')
        st.download_button("Download Preprocessed CSV", csv_processed, "preprocessed_data.csv", "text/csv")

# -----------------------------------------------------------------------------
# TAB 3: Scheduling Pipeline
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Schedule Pipeline")
    st.write("Configure and run your pipeline on schedule. (Simulation)")
    frequency = st.selectbox("Select Frequency", options=["Every Minute", "Every Hour", "Every Day"])
    st.write(f"Selected Frequency: **{frequency}**")
    
    if st.button("Run Pipeline Now"):
        df_result = scheduled_pipeline()
        st.write("Pipeline Result:")
        st.dataframe(df_result)
    
    st.info("Note: Actual scheduling is to be implemented via production schedulers like Airflow or cron jobs.")

# -----------------------------------------------------------------------------
# TAB 4: Logs
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Pipeline Logs")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
        st.text_area("Logs", log_content, height=300)
    else:
        st.info("No logs available.")
    if st.button("Clear Logs"):
        try:
            os.remove(LOG_FILE)
            st.success("Logs cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing logs: {e}")

# -----------------------------------------------------------------------------
# TAB 5: Configuration
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Pipeline Configuration")
    st.write("This section allows you to adjust configuration parameters for the NLP pipeline.")
    num_pages_to_scrape = st.number_input("Default Number of Pages to Scrape", min_value=1, max_value=100, value=10, step=1)
    use_selenium_option = st.checkbox("Use Selenium for Scraping", value=False)
    st.session_state["num_pages_to_scrape"] = num_pages_to_scrape
    st.session_state["use_selenium"] = use_selenium_option
    st.write("Current Settings:")
    st.json({
        "num_pages_to_scrape": st.session_state["num_pages_to_scrape"],
        "use_selenium": st.session_state["use_selenium"],
        "OpenAI Model": st.secrets["openai"].get("default_model", "gpt-3.5-turbo"),
        "Max Tokens": st.secrets["openai"].get("max_tokens", 150)
    })
    if st.button("Save Configuration"):
        st.session_state["pipeline_config"] = {
            "num_pages_to_scrape": num_pages_to_scrape,
            "use_selenium": use_selenium_option,
        }
        st.success("Configuration saved successfully!")
    st.markdown("### Backup Configuration")
    if st.button("Backup Configuration"):
        backup_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "configuration": {
                "num_pages_to_scrape": st.session_state.get("num_pages_to_scrape"),
                "use_selenium": st.session_state.get("use_selenium"),
            }
        }
        backup_file = f"backup_config_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, indent=2)
        st.download_button("Download Backup", backup_file, file_name=backup_file, mime="application/json")
        st.success("Backup created successfully.")

# -----------------------------------------------------------------------------
# TAB 6: NLP Test
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("NLP Test")
    st.write("Enter some text to test the NLP pipeline (e.g., summarization, sentiment analysis, keyword extraction).")
    sample_text = st.text_area("Enter Text for NLP Testing", placeholder="Type some text here...")
    if st.button("Run NLP Test"):
        if not sample_text.strip():
            st.error("Please enter some text for testing.")
        else:
            test_prompt = (
                f"Analyze the following text and return a JSON object with keys:\n"
                f"  - summary: a brief summary\n"
                f"  - sentiment: overall sentiment (POS, NEG, or NEU)\n"
                f"  - polarity: a float representing sentiment polarity\n"
                f"  - keywords: a list of key terms\n\n"
                f"Text: {sample_text}\n\n"
                f"Only return the JSON object."
            )
            try:
                gpt_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an NLP analysis assistant."},
                        {"role": "user", "content": test_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.5,
                )
                gpt_output = gpt_response["choices"][0]["message"]["content"].strip()
                extracted_nlp = json.loads(gpt_output)
                st.write("NLP Test Output:")
                st.json(extracted_nlp)
            except Exception as e:
                st.error(f"Error during NLP test: {e}")

# -----------------------------------------------------------------------------
# TAB 7: Additional Pipeline Actions
# -----------------------------------------------------------------------------
with tabs[6]:
    st.header("Additional Pipeline Actions")
    st.write("Advanced actions for pipeline management and performance monitoring.")
    if st.button("Run Full Pipeline"):
        df_result = scheduled_pipeline()
        st.write("Pipeline Result:")
        st.dataframe(df_result)
    st.info("This section can be extended with more advanced pipeline controls and analytics.")

