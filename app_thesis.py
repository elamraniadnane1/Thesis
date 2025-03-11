import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import plotly.figure_factory as ff
import seaborn as sns
import io
from io import BytesIO
import base64
import time
import logging
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import openai
from streamlit import tabs
import re
import concurrent.futures
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

# Import the authentication system
from auth_system import init_auth, login_page, require_auth
# Initialize authentication
init_auth()
# Check authentication before showing the main app
if not login_page():
    st.stop()

# ------------------------------------------------------
# Define Logout Function
# ------------------------------------------------------
def logout():
    """
    Reset the authentication status and refresh the app to show the login page.
    """
    st.session_state['authentication_status'] = False  # Adjust the key based on your auth_system
    st.rerun()

# ------------------------------------------------------
# 1. Custom CSS (Optional)
# ------------------------------------------------------
# Load the SVG file
def load_svg():
    svg_path = Path("icon.svg")
    return svg_path.read_text()

st.set_page_config(
    page_title="Social Media Analysis Tool 2025, By: Adnane El Amrani",
    page_icon=load_svg(),
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------------
# Initialize Embedding Model and FAISS Index for RAG
# -----------------------------------------------------------------------------------
@st.cache_resource
def initialize_rag(csv_file_path: str):
    """
    Initializes the RAG system by loading comments, generating embeddings,
    and setting up the FAISS index for similarity search.
    """
    # Load comments
    df = pd.read_csv(csv_file_path)
    df.columns = ["article_url", "commenter", "comment_date", "comment"]
    df['processed_comment'] = df['comment'].apply(preprocess_text)
    
    # Initialize Sentence Transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient
    
    # Generate embeddings
    embeddings = embedding_model.encode(df['processed_comment'].tolist(), convert_to_numpy=True)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Using Inner Product for cosine similarity
    index.add(embeddings)
    
    return df, embedding_model, index

# Preprocessing function (as defined earlier)
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(text.split())  # Remove extra spaces
    return text


# Create a ThreadPoolExecutor for potential background tasks (scraping, heavy analysis).
# This allows the scraping/analysis to run in a background thread so the UI remains responsive.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# We'll also create some session_state placeholders to avoid re-running heavy tasks:
if "df_main" not in st.session_state:
    st.session_state.df_main = None   # Will store our main DataFrame (comments) once loaded
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False  # Flag to track if sentiment analysis is done
if "scraping_in_progress" not in st.session_state:
    st.session_state.scraping_in_progress = False

theme = st.get_option("theme.base")  # Returns "light" or "dark"

if theme == "dark":
    # Use dark-friendly CSS
    dark_mode_css = """
    <style>
    body {
        font-family: 'Poppins', sans-serif;
        /* Let Streamlit or user override handle background in dark mode */
        background: none !important; 
        color: #f0f0f0 !important;  /* Lighter text for dark background */
    }
    </style>
    """
    st.markdown(dark_mode_css, unsafe_allow_html=True)
else:
    # Use your original gradient for light mode
    light_mode_css = """
    <style>
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f0f4f8 0%, #d7e3fc 100%) !important;
        color: #1a365d !important; 
    }
    </style>
    """
    st.markdown(light_mode_css, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Import Lucide icons CSS */
    @import url('https://cdn.jsdelivr.net/npm/lucide-static@0.16.29/font/lucide.min.css');
    
    /* Main layout and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    
    /* Previous styles remain the same until tabs... */
    
    /* Enhanced Tabs styling with icons */
    .stTabs {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 2.5rem;
        position: relative;
    }

    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        gap: 10px;
        padding: 0.75rem;
        overflow-x: auto;
        scrollbar-width: thin;
        white-space: nowrap;
        -ms-overflow-style: -ms-autohiding-scrollbar;
        position: relative;
        max-width: calc(5 * 200px);
        margin: 0 auto;
    }

    .stTabs [data-baseweb="tab"] {
        min-width: 180px;
        height: 50px;
        background: rgba(243, 244, 246, 0.7);
        border-radius: 10px;
        color: #2B3E50;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem;
        margin: 0;
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 8px;
    }

    /* Tab icons */
    .stTabs [data-baseweb="tab"]::before {
        font-family: 'lucide';
        font-size: 1.2rem;
        margin-right: 0.5rem;
        opacity: 0.8;
        transition: all 0.3s ease;
    }

    /* Custom icons for each tab - adjust content based on your tab names */
    .stTabs [data-baseweb="tab"][aria-label*="Dashboard"]::before {
        content: '\\e900';  /* dashboard icon */
    }

    .stTabs [data-baseweb="tab"][aria-label*="Analytics"]::before {
        content: '\\e901';  /* chart icon */
    }

    .stTabs [data-baseweb="tab"][aria-label*="Posts"]::before {
        content: '\\e902';  /* message-square icon */
    }

    .stTabs [data-baseweb="tab"][aria-label*="Users"]::before {
        content: '\\e903';  /* users icon */
    }

    .stTabs [data-baseweb="tab"][aria-label*="Settings"]::before {
        content: '\\e904';  /* settings icon */
    }

    /* Icon hover effect */
    .stTabs [data-baseweb="tab"]:hover::before {
        transform: scale(1.1);
        opacity: 1;
    }

    /* Active tab icon styling */
    .stTabs [aria-selected="true"]::before {
        opacity: 1;
        transform: scale(1.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3AAFA9 0%, #2B7A78 100%) !important;
        color: white !important;
        font-weight: 600;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(43, 122, 120, 0.2);
    }

    /* Gradient fade for overflow indication */
    .stTabs [data-baseweb="tab-list"]::after {
        content: '';
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        width: 50px;
        background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,1));
        pointer-events: none;
    }

    /* Tab hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(58, 175, 169, 0.1);
        transform: translateY(-2px);
    }

    /* Rest of the styles remain the same... */

    /* Enhanced animations for tabs */
    .stTabs [data-baseweb="tab"] {
        animation: slideIn 0.3s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Responsive adjustments for tabs with icons */
    @media screen and (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            min-width: 160px;
            height: 45px;
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
        }

        .stTabs [data-baseweb="tab"]::before {
            font-size: 1rem;
            margin-right: 0.3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True)

# ------------------------------------------------------
# 2. Initialize the sentiment analysis pipeline
# ------------------------------------------------------
@st.cache_resource
def load_sentiment_pipeline():
    """Load the Arabic BERT model for sentiment analysis (cached)."""
    sentiment_pipe = pipeline(
        "text-classification",
        model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    )
    return sentiment_pipe

# ------------------------------------------------------
# 3. Load data
# ------------------------------------------------------
@st.cache_data
def load_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load the comments CSV into a pandas DataFrame.
    Expected CSV format:
    article_url, commenter, comment_date, comment
    """
    df = pd.read_csv(csv_file_path)
    df.columns = ["article_url", "commenter", "comment_date", "comment"]
    return df

# ------------------------------------------------------
# 4. Perform sentiment analysis (Standard)
# ------------------------------------------------------
def analyze_sentiment(df: pd.DataFrame, sentiment_pipe):
    """Run the sentiment analysis pipeline on each comment without chunking."""
    sentiments = []
    for comment in df['comment']:
        # If pipeline returns a single dict per comment
        result = sentiment_pipe(comment)  
        # result might look like: {'label': 'POS', 'score': 0.99}
        sentiments.append(result['label'])  # "POS", "NEG", or "NEU"
    df['sentiment'] = sentiments
    return df

# ------------------------------------------------------
# 4a. Perform sentiment analysis in chunks
# ------------------------------------------------------
def analyze_sentiment_in_chunks(df: pd.DataFrame, sentiment_pipe, chunk_size: int = 100):
    """
    Run the sentiment analysis pipeline on each comment in chunks to handle large data more efficiently.
    This helps avoid memory overload and provides real-time progress updates for the user.
    """
    total_comments = len(df)
    if total_comments == 0:
        st.warning("No comments found in the dataset.")
        df['sentiment'] = []
        return df

    sentiments = []
    num_chunks = (total_comments - 1) // chunk_size + 1  # Ceiling division
    
    progress_bar = st.progress(0)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_comments)
        chunk = df['comment'][start_idx:end_idx].tolist()
        
        # Inference on the chunk
        results = sentiment_pipe(chunk)
        # results might be a list of dicts:
        # [{'label': 'POS', 'score': 0.99}, {'label': 'NEG', 'score': 0.88}, ...]
        chunk_sentiments = [res['label'] for res in results]
        sentiments.extend(chunk_sentiments)
        
        # Update progress bar
        progress_val = int((i + 1) / num_chunks * 100)
        progress_bar.progress(progress_val)
    
    df['sentiment'] = sentiments
    return df


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def init_driver():
    """
    Initialize a headless Chrome WebDriver using webdriver_manager.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def load_all_comments(driver):
    """
    Continuously click 'Load More' buttons to retrieve all comments, if available.
    """
    while True:
        try:
            load_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More') or contains(text(), 'تحميل المزيد')]"))
            )
            load_more_button.click()
            logging.info("Clicked 'Load More' button to load more comments.")
            time.sleep(random.uniform(1, 3))
        except Exception:
            logging.info("No more 'Load More' buttons found. All comments loaded.")
            break

def extract_article_data(driver, url):
    """
    Extract details of a single article (title, publication_date, content, images)
    and its associated comments from the provided URL.
    """
    article_data = {
        'title': None,
        'url': url,
        'publication_date': None,
        'content': None,
        'images': [],
    }
    comments_data = []

    try:
        driver.get(url)
        logging.info(f"Opened URL: {url}")

        # Wait for main content (e.g. h1) to appear
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'h1')))

        # Attempt to load all comments
        load_all_comments(driver)

        # Parse the rendered HTML
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 1. Article Title
        try:
            title_tag = soup.find('h1')
            if title_tag:
                article_data['title'] = title_tag.get_text(strip=True)
                logging.info(f"Title: {article_data['title']}")
            else:
                logging.error("Could not find article title.")
        except AttributeError:
            logging.error("Failed to retrieve article title.")

        # 2. Publication Date (multiple selectors)
        try:
            pub_date = None
            possible_selectors = [
                {'tag': 'span', 'class': 'publication-date'},
                {'tag': 'div', 'class': 'article-meta'},
                {'tag': 'time', 'class': 'published'},
                {'tag': 'div', 'class': 'article-date'},
            ]
            for sel in possible_selectors:
                candidate = soup.find(sel['tag'], class_=sel['class'])
                if candidate:
                    pub_date = candidate.get_text(strip=True)
                    break

            if pub_date:
                article_data['publication_date'] = pub_date
                logging.info(f"Publication Date: {pub_date}")
            else:
                logging.error("Could not find the publication date.")
        except AttributeError:
            logging.error("Failed to retrieve publication date.")

        # 3. Article Content
        try:
            content_div = soup.find('div', class_='content-article') or \
                          soup.find('div', class_='article-content')
            if content_div:
                paragraphs = content_div.find_all('p')
                combined_text = "\n".join([p.get_text(strip=True) for p in paragraphs])
                article_data['content'] = combined_text
                logging.info("Article content extracted.")
            else:
                logging.error("Could not find article content.")
        except AttributeError:
            logging.error("Failed to retrieve article content.")

        # 4. Images
        try:
            images = content_div.find_all('img') if content_div else []
            image_urls = [img['src'] for img in images if 'src' in img.attrs]
            article_data['images'] = image_urls
            logging.info(f"Found {len(image_urls)} images.")
        except AttributeError:
            logging.error("Failed to retrieve images.")

        # 5. Comments
        try:
            comments_section = soup.find('div', class_='comments-area')
            if comments_section:
                comment_list = comments_section.find('ul', class_='comment-list')
                if comment_list:
                    all_li = comment_list.find_all('li', class_='comment')
                    for idx, comment_li in enumerate(all_li, start=1): 
                            commenter = comment_li.find('span', class_='fn').get_text(strip=True)
                            comment_date = comment_li.find('div', class_='comment-date').get_text(strip=True)
                            comment_text = comment_li.find('div', class_='comment-text').get_text(strip=True)
                            comments_data.append({
                                'article_url': url,
                                'commenter': commenter,
                                'comment_date': comment_date,
                                'comment': comment_text
                            })
                            logging.info(f"Comment {idx} by {commenter}")
                    logging.info(f"Total comments extracted: {len(comments_data)}")
                else:
                    logging.info("No comment list found in the comments section.")
            else:
                logging.info("No comments section found.")
        except Exception as e:
            logging.error(f"Error extracting comments: {e}")

    except Exception as e:
        logging.error(f"Error processing URL {url}: {e}")

    return article_data, comments_data
import dateparser

# Function to parse Arabic date
def parse_arabic_date(date_str):
    try:
        # Split the date string
        # Example: "الجمعة 20 دجنبر 2024 - 19:11"
        parts = date_str.split()
        
        # Extract day, month, and year
        day = int(parts[1])  # 20
        month_name = parts[2]  # دجنبر
        year = int(parts[3])  # 2024
        
        # Map Arabic month names to numbers
        month_map = {
            'يناير': 1, 'فبراير': 2, 'مارس': 3, 'أبريل': 4,
            'ماي': 5, 'يونيو': 6, 'يوليوز': 7, 'غشت': 8,
            'شتنبر': 9, 'أكتوبر': 10, 'نونبر': 11, 'دجنبر': 12
        }
        
        month = month_map.get(month_name, 1)  # Default to 1 if month not found
        
        return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
    except Exception:
        # Return a default date if parsing fails
        return pd.to_datetime('2024-12-01')

def run_scraper():
    """
    Reads 'hespress_politics_urls.csv', scrapes data, and appends
    new articles/comments to the existing CSV files.
    """
    # Load URLs
    try:
        df_urls = pd.read_csv('hespress_politics_urls.csv')
        logging.info(f"Loaded {len(df_urls)} URLs from hespress_politics_urls.csv.")
    except FileNotFoundError:
        logging.error("File 'hespress_politics_urls.csv' not found.")
        return
    except Exception as e:
        logging.error(f"Error reading 'hespress_politics_urls.csv': {e}")
        return

    #Initialize driver
    driver = init_driver()

    all_articles = []
    all_comments = []

    #Loop over each URL
    for index, row in df_urls.iterrows():
        url_title = row.get('title', f"Article #{index+1}")
        url_link = row.get('url', None)
        if url_link:
            logging.info(f"Processing: {url_title} -> {url_link}")
            article_data, comments_data = extract_article_data(driver, url_link)
            all_articles.append(article_data)
            all_comments.extend(comments_data)
            # Sleep to mimic human-like behavior
            time.sleep(random.uniform(2, 5))
        else:
            logging.warning(f"No URL found for row {index+1}.")

    driver.quit()
    logging.info("Browser closed.")

    if all_articles:
        new_articles = pd.DataFrame(all_articles)
        try:
            existing_articles = pd.read_csv('hespress_politics_details.csv')
            combined_articles = pd.concat([existing_articles, new_articles], ignore_index=True)
            

            combined_articles.drop_duplicates(
                subset=["url", "title"],  # Adjust as needed
                keep="first",
                inplace=True
            )
        except FileNotFoundError:
            combined_articles = new_articles

        combined_articles.to_csv('hespress_politics_details.csv', index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(new_articles)} new articles to 'hespress_politics_details.csv'.")


    if all_comments:
        new_comments = pd.DataFrame(all_comments)
        try:
            existing_comments = pd.read_csv('hespress_politics_comments.csv')
            combined_comments = pd.concat([existing_comments, new_comments], ignore_index=True)
            
            # -----------------------------------------------------
            # ADD THIS DEDUPLICATION LINE HERE
            # -----------------------------------------------------
            # Use whichever columns uniquely identify a comment:
            combined_comments.drop_duplicates(
                subset=["article_url", "commenter", "comment_date", "comment"],
                keep="first",
                inplace=True
            )

        except FileNotFoundError:
            # If the CSV doesn't exist yet, no dedup step is needed initially
            combined_comments = new_comments

        combined_comments.to_csv('hespress_politics_comments.csv', index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(new_comments)} new comments to 'hespress_politics_comments.csv'.")

        st.rerun()

# Cache the results to minimize repeated API calls and control costs
@st.cache_data
def perform_gpt_topic_modeling(text_data: str, model_name: str, language_hint: str = "Arabic", max_topics: int = 5) -> list:
    """
    Use an OpenAI GPT model to generate a list of potential topics from the text_data.
    This function supports Arabic, Darija, French, and English, but language is
    indicated via the language_hint parameter for better GPT instructions.
    """
    if not text_data.strip():
        return []

    # Prompt to feed GPT - short and precise to reduce token usage
    prompt = (
        f"You are an expert in text analysis. You will read the following text in {language_hint} and "
        f"extract up to {max_topics} major political topics or themes. "
        f"Text:\n{text_data}\n\nReturn the topics in a concise list."
    )
    
    try:
        # Basic example call to GPT
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more deterministic results
            max_tokens=512    # Limit tokens to control cost
        )
        
        # Extract the GPT output (assistant's message)
        gpt_output = response.choices[0].message["content"].strip()
        
        # Naive approach: Split lines to form a list
        topics = gpt_output.split("\n")
        # Clean each topic line
        topics = [t.strip("- ") for t in topics if t.strip()]
        
        return topics
    except Exception as e:
        st.error(f"Error during GPT topic modeling: {e}")
        return []

def plot_sentiment_pie_chart(df: pd.DataFrame):
    """Create a pie chart of sentiment distribution using Plotly."""
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    fig = px.pie(
        sentiment_counts,
        names='sentiment',
        values='count',
        color='sentiment',
        color_discrete_map={'POS': 'green', 'NEG': 'red', 'NEU': 'blue'},
        title='Sentiment Distribution (Pie Chart)',
        hole=0.3  # For a donut chart; set to 0 for a regular pie.
    )
    st.plotly_chart(fig)

def plot_heatmap(df: pd.DataFrame):
    """
    Example: Plot a simple heatmap showing correlation 
    among numeric columns (for demonstration).
    """
    numeric_data = pd.DataFrame({
        'random_feature_1': np.random.randn(len(df)),
        'random_feature_2': np.random.randn(len(df))
    })
    corr = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax)
    ax.set_title("Sample Correlation Heatmap")
    st.pyplot(fig)

def top_commenters_bar_chart(df: pd.DataFrame, top_n: int = 10):
    """
    Show a bar chart of the top commenters (by comment count).
    """
    top_commenters = df['commenter'].value_counts().head(top_n).reset_index()
    top_commenters.columns = ['commenter', 'count']
    
    fig = px.bar(
        top_commenters, 
        x='commenter', 
        y='count', 
        title=f"Top {top_n} Commenters",
        color='count',
        color_continuous_scale='Tealgrn'
    )
    st.plotly_chart(fig)

def download_csv(df: pd.DataFrame, filename='analyzed_comments.csv'):
    """
    Provide a link to download the DataFrame as a CSV.
    """
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv_buffer.getvalue()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Add this with your other helper functions, before the main() function
def highlight_sentiment(row):
    """
    Returns a background color for each row based on the sentiment value.
    Colors: green for positive, red for negative, yellow for neutral
    """
    color = "white"
    if row["sentiment"] == "POS":
        color = "#d8f5d8"  # light green
    elif row["sentiment"] == "NEG":
        color = "#fddddd"  # light red
    elif row["sentiment"] == "NEU":
        color = "#f9f9c5"  # light yellow
    return [f"background-color: {color}"] * len(row)
@require_auth
def main():
    st.title("Social Media & Newspaper Analysis Tool")
    st.write(
        """
        This dashboard uses an **Arabic BERT** model 
        ([`CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment)) 
        to analyze the sentiment of comments. 
        Upload your CSV file to get started.
        """
    )
    
    # Sidebar options
    with st.sidebar:
        st.markdown("---")
        if st.button("Logout"):
            logout()
            st.success("You have been logged out.")

        st.write("---")
        st.header("App Configuration")
        show_heatmap = st.checkbox("Show Sample Heatmap", value=False)
        show_top_commenters = st.checkbox("Show Top Commenters Bar Chart", value=True)
        top_n_commenters = st.slider("Number of Top Commenters to Display", 5, 30, 10)
        
        st.markdown("---")

        st.write("GPT Model Selection (Political Topic Modeling)")
        gpt_model = st.selectbox(
            "Choose GPT Model:",
            options=["gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="Select a GPT model for topic modeling. GPT-3.5-turbo is cheaper, GPT-4 is more powerful."
        )

        st.write("**Choose Analysis Method**")
        analysis_method = st.radio(
            "Select how to run sentiment analysis:",
            ("Standard", "Chunked (Recommended for large datasets)")
        )
        chunk_size = st.slider("Chunk Size (if using chunked analysis)", 50, 1000, 100, step=50)

        st.write("---")
        st.write("**Data Upload**")

    uploaded_file = st.file_uploader(
        label="Upload a CSV file (with columns: article_url, commenter, comment_date, comment):",
        type="csv"
    )

    # Default path (update if needed)
    csv_file_path = "hespress_politics_comments.csv"



    # Load pipeline once
    sentiment_pipe = load_sentiment_pipeline()

    # Initialize RAG
    csv_file_path = "hespress_politics_comments.csv"
    if os.path.exists(csv_file_path):
        df_main, embedding_model, faiss_index = initialize_rag(csv_file_path)
        st.session_state.df_main = df_main
        st.session_state.embedding_model = embedding_model
        st.session_state.faiss_index = faiss_index
        st.success(f"Loaded and indexed data from `{csv_file_path}`.")
    else:
        st.error(f"The file `{csv_file_path}` does not exist. Please upload the CSV file.")
        st.stop()

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = ["article_url", "commenter", "comment_date", "comment"]
        st.session_state.df_main = df
        st.success("File uploaded successfully!")
    else:
        st.info("No file uploaded. Using the Hespress CSV file stored in our Database.")
        df = load_data(csv_file_path)
        st.session_state.df_main = df
    

    if df is not None:
        st.subheader("Custom Comment Analysis")
        user_comment = st.text_area("Enter an Arabic/Darija comment to analyze its sentiment:")
        analyze_btn = st.button("Analyze This Comment")

        if analyze_btn:
            if user_comment.strip():
                with st.spinner("Analyzing..."):
                    user_result = sentiment_pipe(user_comment)
                    # Correctly access the first element of the list and then the 'label'
                    user_sentiment = user_result[0]['label']
                st.write(f"**Predicted Sentiment**: {user_sentiment}")
            else:
                st.warning("Please enter a non-empty comment.") 
   
        # Create Streamlit tabs for organized navigation
        # First, add the tabs with icons
        tabs = st.tabs([
            "🔄 Data Preprocessing",
            "😊 Sentiment Analysis",
            "📊 Data Visualizations",
            "🔍 Topic Modeling",
            "⚠️ Terrorism Detection",
            "🚫 Offensive Language Detection",
            "🌐 Scraping",
            "🤖 Chatbot",
            "📈 Evaluation",
            "⚡ Performance",
            "📡 Monitoring",
            "⬇️ Download"
        ]) 

        with tabs[0]:
            st.subheader("Data Overview")
            
            # Data statistics in an expandable section
            with st.expander("📊 Dataset Statistics", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Comments", len(df))
                with col2:
                    st.metric("Unique Commenters", df['commenter'].nunique())
                with col3:
                    st.metric("Articles Covered", df['article_url'].nunique())
                
                # Calculate missing values
                missing_data = df.isnull().sum()
                if missing_data.any():
                    st.write("Missing Values per Column:")
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Percentage': (missing_data.values / len(df) * 100).round(2)
                    })
                    st.dataframe(missing_df)

            # Text preprocessing options
            st.subheader("🔄 Text Preprocessing Options")
            
            preprocess_col1, preprocess_col2 = st.columns(2)
            
            with preprocess_col1:
                remove_urls = st.checkbox("Remove URLs", value=True)
                remove_numbers = st.checkbox("Remove Numbers", value=False)
                remove_punctuation = st.checkbox("Remove Punctuation", value=False)
                
            with preprocess_col2:
                remove_extra_spaces = st.checkbox("Remove Extra Spaces", value=True)
                normalize_arabic = st.checkbox("Normalize Arabic Text", value=True)
                remove_emoji = st.checkbox("Remove Emojis", value=False)

            if st.button("Apply Text Preprocessing"):
                with st.spinner("Preprocessing text..."):
                    # Create a copy of the dataframe
                    processed_df = df.copy()
                    
                    def preprocess_text(text):
                        if pd.isna(text):
                            return text
                            
                        # Convert to string if not already
                        text = str(text)
                        
                        if remove_urls:
                            text = re.sub(r'http\S+|www.\S+', '', text)
                            
                        if remove_numbers:
                            text = re.sub(r'\d+', '', text)
                            
                        if remove_punctuation:
                            text = re.sub(r'[^\w\s]', '', text)
                            
                        if remove_extra_spaces:
                            text = ' '.join(text.split())
                            
                        if normalize_arabic:
                            # Basic Arabic normalization
                            text = re.sub("[إأٱآا]", "ا", text)
                            text = re.sub("ى", "ي", text)
                            text = re.sub("ة", "ه", text)
                            
                        if remove_emoji:
                            text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
                            
                        return text
                    
                    processed_df['processed_comment'] = processed_df['comment'].apply(preprocess_text)
                    df = processed_df
                    st.success("Text preprocessing completed!")

            # Data cleaning options
            st.subheader("🧹 Data Cleaning")
            
            clean_col1, clean_col2 = st.columns(2)
            
            with clean_col1:
                if st.button("Remove Duplicate Comments"):
                    initial_count = len(df)
                    df.drop_duplicates(subset=["article_url", "commenter", "comment_date", "comment"], 
                                     keep="first", inplace=True)
                    st.success(f"Removed duplicates. Row count went from {initial_count} to {len(df)}.")
                    
                min_length = st.number_input("Minimum Comment Length", min_value=1, value=5)
                if st.button("Remove Short Comments"):
                    initial_count = len(df)
                    df = df[df['comment'].str.len() >= min_length]
                    st.success(f"Removed comments shorter than {min_length} characters. Rows: {initial_count} → {len(df)}")

            with clean_col2:
                if st.button("Drop Missing Values"):
                    initial_count = len(df)
                    df.dropna(subset=["comment"], inplace=True)
                    st.success(f"Dropped missing comment rows. Rows: {initial_count} → {len(df)}")
                
                # Add option to remove comments with specific patterns
                pattern = st.text_input("Remove comments containing pattern (regex):")
                if st.button("Remove Pattern") and pattern:
                    initial_count = len(df)
                    df = df[~df['comment'].str.contains(pattern, na=False, regex=True)]
                    st.success(f"Removed comments with pattern. Rows: {initial_count} → {len(df)}")

            # Data preview
            st.subheader("👀 Data Preview")
            
            preview_options = st.radio(
                "Choose preview type:",
                ["Sample Data", "Filtered View", "Statistics"],
                horizontal=True
            )
            
            if preview_options == "Sample Data":
                sample_size = st.slider("Sample size", 5, 50, 10)
                st.dataframe(df.sample(n=sample_size))
                
            elif preview_options == "Filtered View":
                col1, col2 = st.columns(2)
                with col1:
                    selected_commenter = st.selectbox(
                        "Filter by commenter:",
                        ["All"] + list(df['commenter'].unique())
                    )
                with col2:
                    

                    # Convert comment_dates to datetime
                    df['parsed_date'] = df['comment_date'].apply(parse_arabic_date)
                    
                    # Get min and max dates for the date input
                    min_date = df['parsed_date'].min().date()
                    max_date = df['parsed_date'].max().date()
                    
                    date_range = st.date_input(
                        "Select date range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                
                filtered_df = df.copy()
                if selected_commenter != "All":
                    filtered_df = filtered_df[filtered_df['commenter'] == selected_commenter]
                st.dataframe(filtered_df)
                
            else:  # Statistics
                st.write("Comment Length Statistics:")
                df['comment_length'] = df['comment'].str.len()
                stats_df = df['comment_length'].describe()
                st.dataframe(stats_df)
                
                fig = px.histogram(
                    df,
                    x='comment_length',
                    nbins=50,
                    title='Distribution of Comment Lengths'
                )
                st.plotly_chart(fig)

            # Export processed data
            st.subheader("💾 Export Processed Data")
            if st.button("Download Processed Dataset"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='processed_comments.csv',
                    mime='text/csv'
                )

        with tabs[1]:
            st.subheader("Sentiment Analysis")
            if st.button("Run Sentiment Analysis"):
                with st.spinner("Analyzing sentiment..."):
                    if analysis_method == "Standard":
                        st.session_state.df_main = analyze_sentiment_in_chunks(df, sentiment_pipe, chunk_size=1000)
                    else:
                        st.session_state.df_main = analyze_sentiment_in_chunks(df, sentiment_pipe, chunk_size=chunk_size)
                st.success("Sentiment Analysis Complete!")

                # Show first 10 results with highlighting
                st.write("Sentiment Analysis Results (first 10 rows):")
                st.dataframe(df.head(10).style.apply(highlight_sentiment, axis=1))
            else:
                st.info("Click the button above to run sentiment analysis.")
            
            

            if "sentiment" in df.columns:
                # Create a new column for comment length
                df["comment_length"] = df["comment"].astype(str).apply(len)

                # Plot using Plotly histogram
                fig_len = px.histogram(
                    df,
                    x="comment_length",
                    nbins=30,
                    title="Distribution of Comment Lengths",
                    labels={"comment_length": "Comment Length (characters)"}
                )
                st.plotly_chart(fig_len)
                # Sentiment Distribution
                st.subheader("Sentiment Distribution")
                sentiment_counts = df['sentiment'].value_counts()

                # Traditional Matplotlib bar chart
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Count")
                ax.set_title("Bar Chart: Distribution of Sentiments in Comments")
                st.pyplot(fig)


                # Comment Length Distribution
                st.subheader("Comment Length Distribution")

                #Top Commenters
                if show_top_commenters:
                    st.subheader("Top Commenters")
                    top_commenters_bar_chart(df, top_n=top_n_commenters)

                #Filter by sentiment
                st.subheader("Filter / Search by Sentiment")
                selected_sentiment = st.selectbox("Choose a sentiment to filter:", ["ALL"] + list(sentiment_counts.index))

                if selected_sentiment != "ALL":
                    filtered_df = df[df['sentiment'] == selected_sentiment]
                else:
                    filtered_df = df

                st.write(f"Showing comments for sentiment: **{selected_sentiment}**")
                st.dataframe(
                    filtered_df[['commenter', 'comment_date', 'comment', 'sentiment']].style.apply(highlight_sentiment, axis=1)
                )

                #Observations / Insights
                st.subheader("Observations / Insights")
                st.write(
                    """
                    - Use the above tables and charts to identify common themes 
                    or patterns in citizen comments.
                    - Focus on negative or neutral sentiments to understand areas 
                    of dissatisfaction or potential improvements.
                    - The government can prioritize policies or decisions 
                    that address the most prominent concerns.
                    - Compare sentiment trends over time to see if reactions 
                    change based on events, announcements, or policy changes.
                    """
                )
                #Time-based sentiment analysis
                if "sentiment" in df.columns:
                    st.subheader("Time-Based Sentiment Analysis")
                    try:
                        import dateparser

                        # Function to correct month names
                        def correct_month_names(date_str, mapping):
                            for wrong, correct in mapping.items():
                                if wrong in date_str:
                                    return date_str.replace(wrong, correct)
                            return date_str  # Return unchanged if no match found

                        # Define month mapping
                        month_mapping = {
                            'دجنبر': 'ديسمبر',
                            'كانون الثاني': 'كانون الثاني',  # January
                            'شباط': 'شباط',                # February
                            'آذار': 'آذار',                 # March
                            'نيسان': 'نيسان',                # April
                            'أيار': 'أيار',                  # May
                            'حزيران': 'حزيران',              # June
                            'تموز': 'تموز',                  # July
                            'آب': 'آب',                      # August
                            'أيلول': 'أيلول',                # September
                            'تشرين الأول': 'تشرين الأول',      # October
                            'تشرين الثاني': 'تشرين الثاني',      # November
                            'كانون الأول': 'كانون الأول',        # December
                            # Add any other necessary mappings
                        }

                        

                        # Correct month names
                        st.write("Correcting month names in dates...")
                        df['comment_date_corrected'] = df['comment_date'].apply(lambda x: correct_month_names(x, month_mapping))

                        # Parse corrected dates
                        st.write("Parsing corrected dates. This may take a moment...")
                        df['parsed_date'] = df['comment_date_corrected'].apply(parse_arabic_date)

                        # Count successfully parsed dates
                        num_parsed = df['parsed_date'].notna().sum()
                        total = len(df)
                        st.write(f"Successfully parsed {num_parsed} out of {total} dates.")

                        if num_parsed == 0:
                            st.warning("No dates were successfully parsed. Please check the date formats and month mappings.")
                        else:
                            # Drop rows where parsing failed
                            if 'parsed_date' in df.columns:
                                df = df.dropna(subset=['parsed_date'])

                            # Update 'comment_date' with parsed dates
                            df['comment_date'] = df['parsed_date']
                            df = df.drop(columns=['parsed_date', 'comment_date_corrected'])

                            # Ensure 'comment_date' is datetime
                            df['comment_date'] = pd.to_datetime(df['comment_date'])

                            df['date'] = df['comment_date']

                            # Group by date and sentiment
                            daily_counts = df.groupby([df['comment_date'].dt.date, 'sentiment']).size().reset_index(name='count')
                            pivoted = daily_counts.pivot(index='comment_date', columns='sentiment', values='count').fillna(0)

                            st.write("### Daily Counts of Each Sentiment")
                            st.dataframe(pivoted.style.highlight_max(color='lightgreen', axis=0))

                            # Plot trends over time using Plotly
                            pivoted_df = pivoted.reset_index().melt(
                                id_vars='comment_date', var_name='sentiment', value_name='count'
                            )
                            fig2 = px.line(
                                pivoted_df,
                                x='comment_date',
                                y='count',
                                color='sentiment',
                                title="Sentiment Counts Over Time (Interactive)",
                                markers=True
                            )
                            fig2.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Count",
                                legend_title="Sentiment",
                                template="plotly_dark"  # Optional: Choose a template that fits your app's theme
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Could not parse dates. Error: {e}")

            
        with tabs[2]:
            st.subheader("Sentiment Analysis")
            if st.button("Run Analysis"):
                with st.spinner("Analyzing sentiment..."):
                    if analysis_method == "Standard":
                        df = analyze_sentiment_in_chunks(df, sentiment_pipe, chunk_size=1000)
                    else:
                        df = analyze_sentiment_in_chunks(df, sentiment_pipe, chunk_size=chunk_size)
                st.success("Sentiment Analysis Complete!")

                # Show first 10 results with highlighting
                st.write("Sentiment Analysis Results (first 10 rows):")
                st.dataframe(df.head(10).style.apply(highlight_sentiment, axis=1))
            else:
                st.info("Click the button above to run sentiment analysis.")
            
            import dateparser

            # Function to correct month names
            def correct_month_names(date_str, mapping):
                for wrong, correct in mapping.items():
                    if wrong in date_str:
                        return date_str.replace(wrong, correct)
                return date_str  # Return unchanged if no match found

            # Define month mapping
            month_mapping = {
                'دجنبر': 'ديسمبر',
                'كانون الثاني': 'كانون الثاني',  # January
                'شباط': 'شباط',                # February
                'آذار': 'آذار',                 # March
                'نيسان': 'نيسان',                # April
                'أيار': 'أيار',                  # May
                'حزيران': 'حزيران',              # June
                'تموز': 'تموز',                  # July
                'آب': 'آب',                      # August
                'أيلول': 'أيلول',                # September
                'تشرين الأول': 'تشرين الأول',      # October
                'تشرين الثاني': 'تشرين الثاني',      # November
                'كانون الأول': 'كانون الأول',        # December
                # Add any other necessary mappings
            }

            

            # Correct month names
            st.write("Correcting month names in dates...")
            df['comment_date_corrected'] = df['comment_date'].apply(lambda x: correct_month_names(x, month_mapping))

            # Parse corrected dates
            st.write("Parsing corrected dates. This may take a moment...")
            df['parsed_date'] = df['comment_date_corrected'].apply(parse_arabic_date)
        
            # Count successfully parsed dates
            num_parsed = df['parsed_date'].notna().sum()
            total = len(df)
            st.write(f"Successfully parsed {num_parsed} out of {total} dates.")

            if num_parsed == 0:
                st.warning("No dates were successfully parsed. Please check the date formats and month mappings.")
            else:
                # Drop rows where parsing failed
                if 'parsed_date' in df.columns:
                    df = df.dropna(subset=['parsed_date'])

                # Update 'comment_date' with parsed dates
                df['comment_date'] = df['parsed_date']
                df = df.drop(columns=['parsed_date', 'comment_date_corrected'])

                # Ensure 'comment_date' is datetime
                df['comment_date'] = pd.to_datetime(df['comment_date'])

                df['date'] = df['comment_date']

            


            if "sentiment" in df.columns:
                # Create tabs within the sentiment analysis section
                sentiment_subtabs = st.tabs([
                    "📊 Basic Analysis",
                    "📈 Advanced Metrics",
                    "🔍 Pattern Analysis",
                    "⏱️ Temporal Analysis",
                    "🔗 Cross Analysis"
                ])

                with sentiment_subtabs[0]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment Distribution Pie Chart
                        plot_sentiment_pie_chart(df)
                        
                        # Add sentiment percentages
                        sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100
                        st.write("### Sentiment Distribution (%)")
                        for sent, pct in sentiment_percentages.items():
                            st.metric(sent, f"{pct:.1f}%")
                    
                    with col2:
                        # Comment Length vs Sentiment Box Plot
                        df['comment_length'] = df['comment'].str.len()
                        fig = px.box(df, x='sentiment', y='comment_length',
                                title='Comment Length Distribution by Sentiment',
                                color='sentiment')
                        st.plotly_chart(fig)

                with sentiment_subtabs[1]:
                    # Advanced Metrics Section
                    col1, col2 = st.columns(2)
                    
                    try:
                        # Convert comment_date to datetime if not already
                        #df['date'] = pd.to_datetime(df['comment_date'].apply(lambda x: x.split('-')[0]), format='%Y')
                        df['date'] = pd.to_datetime(df['comment_date'], errors="coerce")
                        with col1:
                            # Sentiment Ratio Over Time
                            sentiment_ratio = df.groupby('date')['sentiment'].value_counts(normalize=True).unstack()
                            fig = px.line(sentiment_ratio, title='Sentiment Ratio Evolution',
                                        labels={'value': 'Ratio', 'date': 'Date'})
                            st.plotly_chart(fig)

                        with col2:
                            # Sentiment by Day of Week
                            df['day_of_week'] = df['date'].dt.day_name()
                            dow_sentiment = df.groupby('day_of_week')['sentiment'].value_counts(normalize=True).unstack()
                            fig = px.bar(dow_sentiment, title='Sentiment Distribution by Day of Week',
                                        barmode='stack')
                            st.plotly_chart(fig)

                        # Hour of Day Analysis
                        df['hour'] = df['date'].dt.hour
                        hourly_sentiment = df.groupby('hour')['sentiment'].value_counts(normalize=True).unstack()
                        fig = px.line(hourly_sentiment, title='Sentiment Distribution by Hour',
                                    labels={'hour': 'Hour of Day', 'value': 'Ratio'})
                        st.plotly_chart(fig)
                        
                    except Exception as e:
                        st.warning(f"Could not process temporal data: {str(e)}")

                with sentiment_subtabs[2]:
                    # Pattern Analysis Section
                    st.write("### Comment Pattern Analysis")
                    
                    # Word Count Distribution
                    df['word_count'] = df['comment'].str.split().str.len()
                    fig = px.histogram(df, x='word_count', color='sentiment',
                                    title='Word Count Distribution by Sentiment',
                                    marginal='box')
                    st.plotly_chart(fig)

                    # Add average words per comment metric
                    avg_words = df['word_count'].mean()
                    st.metric("Average Words per Comment", f"{avg_words:.1f}")

                    # Most active times
                    st.write("### Comment Activity Patterns")
                    try:
                        df['hour'] = pd.to_datetime(df['comment_date']).dt.hour
                        hourly_counts = df.groupby('hour').size()
                        fig = px.bar(x=hourly_counts.index, y=hourly_counts.values,
                                title='Comments by Hour of Day',
                                labels={'x': 'Hour', 'y': 'Number of Comments'})
                        st.plotly_chart(fig)
                    except:
                        st.warning("Could not process time-based patterns")

                with sentiment_subtabs[3]:
                    # Temporal Analysis Section
                    st.write("### Temporal Patterns")
                    # Drop rows where parsing failed
                    if 'parsed_date' in df.columns:
                        df = df.dropna(subset=['parsed_date'])

                            
                    df['comment_date'] = pd.to_datetime(df['comment_date'])
                    df['date'] = df['comment_date']
                    try:
                        # Monthly Trend
                        df['month'] = df['date'].dt.month
                        monthly_sentiment = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
                        fig = px.line(monthly_sentiment, title='Monthly Sentiment Trends',
                                    labels={'month': 'Month', 'value': 'Count'})
                        st.plotly_chart(fig)

                        # Seasonal Analysis
                        df['season'] = df['month'].map({12:'Winter', 1:'Winter', 2:'Winter',
                                                    3:'Spring', 4:'Spring', 5:'Spring',
                                                    6:'Summer', 7:'Summer', 8:'Summer',
                                                    9:'Fall', 10:'Fall', 11:'Fall'})
                        seasonal = df.groupby(['season', 'sentiment']).size().unstack(fill_value=0)
                        fig = px.bar(seasonal, title='Seasonal Sentiment Distribution',
                                    barmode='group')
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.warning(f"Could not process seasonal data: {str(e)}")

                with sentiment_subtabs[4]:
                    # Cross Analysis Section
                    st.write("### Cross-Variable Analysis")
                    
                    # Comment Length vs Engagement
                    df['engagement_score'] = df.groupby('commenter')['comment'].transform('count')
                    fig = px.scatter(df, x='comment_length', y='engagement_score',
                                color='sentiment', title='Comment Length vs User Engagement',
                                trendline="lowess")
                    st.plotly_chart(fig)

                    # User Behavior Analysis
                    user_sentiment = df.groupby('commenter')['sentiment'].value_counts().unstack(fill_value=0)
                    user_sentiment['total'] = user_sentiment.sum(axis=1)
                    top_users = user_sentiment.nlargest(10, 'total')
                    fig = px.bar(top_users, title='Top 10 Users Sentiment Distribution',
                                barmode='stack')
                    st.plotly_chart(fig)

                    # User Engagement Stats
                    st.write("### User Engagement Statistics")
                    engagement_stats = df.groupby('commenter').agg({
                        'comment': 'count',
                        'comment_length': 'mean'
                    }).round(2)
                    engagement_stats.columns = ['Number of Comments', 'Avg Comment Length']
                    st.dataframe(engagement_stats.head(10))

                # Interactive Filtering
                st.write("### 🔍 Interactive Comment Explorer")
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_sentiment = st.selectbox(
                        "Filter by Sentiment:",
                        ["All"] + list(df['sentiment'].unique())
                    )
                    
                with col2:
                    min_length = st.slider(
                        "Minimum Comment Length:",
                        min_value=0,
                        max_value=int(df['comment_length'].max()),
                        value=0
                    )

                # Apply filters
                filtered_df = df.copy()
                if selected_sentiment != "All":
                    filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
                filtered_df = filtered_df[filtered_df['comment_length'] >= min_length]

                # Show filtered results
                st.dataframe(
                    filtered_df[['commenter', 'comment_date', 'comment', 'sentiment', 'comment_length']]
                    .style.apply(highlight_sentiment, axis=1)
                )

                # Export Options
                st.write("### 📥 Export Analysis")
                if st.button("Generate Analysis Report"):
                    report = f"""
                    Sentiment Analysis Report
                    ========================
                    Total Comments: {len(df)}
                    Sentiment Distribution:
                    {sentiment_percentages.to_string()}
                    
                    Average Comment Length by Sentiment:
                    {df.groupby('sentiment')['comment_length'].mean().to_string()}
                    
                    Most Active Users:
                    {df['commenter'].value_counts().head().to_string()}
                    """
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="sentiment_analysis_report.txt",
                        mime="text/plain"
                    )
                    
            # Add enhanced analysis options with more controls
            analysis_options = st.expander("Analysis Options", expanded=False)
            with analysis_options:
                col1, col2, col3 = st.columns(3)
                with col1:
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                        help="Filter results based on model confidence"
                    )
                    
                    sentiment_filter = st.multiselect(
                        "Filter Sentiments",
                        ["POS", "NEG", "NEU"],
                        default=["POS", "NEG", "NEU"],
                        help="Select specific sentiments to analyze"
                    )
                    
                with col2:
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        help="Number of comments to process at once"
                    )
                    
                    min_comment_length = st.number_input(
                        "Minimum Comment Length",
                        min_value=0,
                        value=5,
                        help="Filter out short comments"
                    )
                    
                with col3:
                    time_period = st.selectbox(
                        "Analysis Time Period",
                        ["All Time", "Last Week", "Last Month", "Last Quarter", "Last Year", "Custom Range"],
                        help="Filter analysis by time period"
                    )
                    
                    if time_period == "Custom Range":
                        start_date = st.date_input("Start Date")
                        end_date = st.date_input("End Date")

            # Advanced date processing function
            def detect_and_preprocess_dates(date_value):
                """
                Convert Arabic/French string dates to datetime objects with comprehensive error handling.
                If already a Timestamp (or not a string), return as-is or convert safely.
                """
                # 1. If the value is already a Timestamp or NaN, return it (or NaT if missing)
                if pd.isna(date_value):
                    return pd.NaT
                if isinstance(date_value, pd.Timestamp):
                    return date_value  # It's already parsed
                
                # 2. If it's not a string, try to convert it
                if not isinstance(date_value, str):
                    try:
                        return pd.to_datetime(date_value, errors="coerce")
                    except Exception:
                        return pd.NaT
                
                # 3. Otherwise, it's a string. Proceed with custom parsing logic:
                try:
                    # Arabic month mappings with variations
                    arabic_months = {
                        'يناير': '01', 'فبراير': '02', 'مارس': '03', 'أبريل': '04', 'ابريل': '04',
                        'ماي': '05', 'مايو': '05', 'يونيو': '06', 'يوليوز': '07', 'يوليو': '07',
                        'غشت': '08', 'اغسطس': '08', 'شتنبر': '09', 'سبتمبر': '09',
                        'أكتوبر': '10', 'اكتوبر': '10', 'نونبر': '11', 'نوفمبر': '11',
                        'دجنبر': '12', 'ديسمبر': '12'
                    }
                    
                    # Arabic weekday mappings (for optional validation)
                    arabic_weekdays = {
                        'الاثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت', 'الأحد'
                    }
                    
                    parts = date_value.split()
                    if not parts:  # Empty string
                        return pd.NaT
                    
                    # Optional: If you expect something like "الجمعة 20 دجنبر 2024 - 19:11"
                    #  - parts[0] => weekday
                    #  - parts[1] => day
                    #  - parts[2] => monthName
                    #  - parts[3] => year
                    #  - parts[4] => '-' or similar
                    #  - parts[5] => time
                    
                    # Check if the first token is a known weekday (this is optional)
                    if parts[0] in arabic_weekdays:
                        # Then the day is likely parts[1]
                        day_str = parts[1]
                        month_str = parts[2] if len(parts) > 2 else ''
                        year_str = parts[3] if len(parts) > 3 else ''
                        # If there's a time, it may be parts[5]
                        time_str = parts[5] if len(parts) > 5 else "00:00"
                    else:
                        # If there's no weekday, you might have a different format
                        # Adjust logic as needed for your data
                        return pd.to_datetime(date_value, errors='coerce')
                    
                    # Map Arabic month name
                    month = arabic_months.get(month_str, '01')  # Default to '01' if not found
                    
                    # Validate day/year
                    if not (day_str.isdigit() and year_str.isdigit()):
                        return pd.NaT
                    
                    # Build final date string
                    datetime_str = f"{year_str}-{month}-{day_str} {time_str}"
                    return pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M', errors='coerce')
                
                except Exception as e:
                    st.warning(f"Date parsing error: {str(e)} for date: {date_value}")
                    return pd.NaT



            if st.button("Run Advanced Sentiment Analysis"):
                with st.spinner("Performing comprehensive sentiment analysis..."):
                    try:
                        # Process sentiment analysis
                        if analysis_method == "Standard":
                            results = analyze_sentiment_in_chunks(df, sentiment_pipe, chunk_size=1000)
                        else:
                            results = analyze_sentiment_in_chunks(df, sentiment_pipe, chunk_size=batch_size)
                        
                        if not isinstance(results, pd.DataFrame):
                            st.error("Sentiment analysis failed to return proper results.")
                            st.stop()

                        df = results.copy()  # Ensure we work on a fresh copy

                        # -------------------------------------------------------------------
                        # FIX FOR KEYERROR: 'confidence'
                        # If the model/pipeline doesn't return confidence scores,
                        # create a dummy 'confidence' column so the filter step won't fail.
                        # -------------------------------------------------------------------
                        if 'confidence' not in df.columns:
                            st.warning("Confidence scores not available in the dataset. Creating a dummy column with 1.0")
                            df['confidence'] = 1.0

                        # Process dates
                        df['processed_date'] = df['comment_date'].apply(detect_and_preprocess_dates)
                        
                        # Remove rows with no valid date if needed
                        df.dropna(subset=['processed_date'], inplace=True)

                        # Apply filters
                        df = df[df['comment'].astype(str).str.len() >= min_comment_length]
                        if sentiment_filter:
                            df = df[df['sentiment'].isin(sentiment_filter)]
                            
                        # Confidence-based filtering
                        df = df[df['confidence'] >= confidence_threshold]
                        
                        # Time period filtering
                        if time_period != "All Time":
                            now = pd.Timestamp.now()
                            if time_period == "Custom Range":
                                df = df[
                                    (df['processed_date'].dt.date >= start_date) & 
                                    (df['processed_date'].dt.date <= end_date)
                                ]
                            else:
                                time_deltas = {
                                    "Last Week": pd.Timedelta(days=7),
                                    "Last Month": pd.Timedelta(days=30),
                                    "Last Quarter": pd.Timedelta(days=90),
                                    "Last Year": pd.Timedelta(days=365)
                                }
                                df = df[df['processed_date'] >= now - time_deltas.get(time_period, pd.Timedelta(days=365))]

                        # If no rows remain, warn and stop
                        if df.empty:
                            st.warning("No data available after applying the current filters/time range.")
                            st.stop()

                        # Create analysis tabs
                        analysis_tabs = st.tabs(["📊 Basic Stats", "📈 Temporal Analysis", 
                                                "🔍 Detailed Analysis", "💭 Content Analysis"])

                        # ------------------- TAB 1: Basic Stats --------------------
                        with analysis_tabs[0]:
                            st.subheader("Statistical Overview")
                            
                            total_comments = len(df)
                            avg_confidence = df['confidence'].mean()
                            avg_length = df['comment'].astype(str).str.len().mean()
                            unique_commenters = df['commenter'].nunique()
                            
                            stats_cols = st.columns(4)
                            with stats_cols[0]:
                                st.metric("Total Comments", f"{total_comments:,}")
                                
                            with stats_cols[1]:
                                if not np.isnan(avg_confidence):
                                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                                else:
                                    st.metric("Average Confidence", "N/A")
                                
                            with stats_cols[2]:
                                if not np.isnan(avg_length):
                                    st.metric("Avg Comment Length", f"{avg_length:.0f} chars")
                                else:
                                    st.metric("Avg Comment Length", "N/A")
                                
                            with stats_cols[3]:
                                st.metric("Unique Commenters", f"{unique_commenters:,}")

                            # Sentiment distribution
                            sentiment_dist = df['sentiment'].value_counts()
                            fig_sentiment = px.pie(
                                values=sentiment_dist.values,
                                names=sentiment_dist.index,
                                title="Sentiment Distribution",
                                color_discrete_map={'POS': '#90EE90', 'NEG': '#FFB6C1', 'NEU': '#F0F8FF'}
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True)

                            # Confidence distribution
                            fig_confidence = px.histogram(
                                df,
                                x='confidence',
                                color='sentiment',
                                nbins=50,
                                title='Sentiment Confidence Distribution',
                                color_discrete_map={'POS': '#90EE90', 'NEG': '#FFB6C1', 'NEU': '#F0F8FF'}
                            )
                            st.plotly_chart(fig_confidence, use_container_width=True)

                        # ------------------- TAB 2: Temporal Analysis --------------------
                        with analysis_tabs[1]:
                            st.subheader("Temporal Patterns")

                            # Prepare daily sentiments
                            df['date'] = df['processed_date'].dt.date
                            daily_counts = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
                            if daily_counts.empty:
                                st.warning("No daily sentiment data to display.")
                            else:
                                # Pivot to get columns = sentiments
                                pivoted = daily_counts.pivot(index='date', columns='sentiment', values='count').fillna(0)

                                # Rolling average
                                window_size = st.slider("Rolling Average Window (days)", 1, 30, 7)
                                rolling_sentiments = pivoted.rolling(window=window_size).mean().reset_index()

                                # Melt for Plotly
                                melted_rolling = rolling_sentiments.melt(
                                    id_vars='date', 
                                    var_name='sentiment', 
                                    value_name='count'
                                )

                                fig_time = px.line(
                                    melted_rolling,
                                    x='date',
                                    y='count',
                                    color='sentiment',
                                    title=f"{window_size}-Day Rolling Average of Sentiment Distribution",
                                    labels={"count": "Number of Comments", "date": "Date"}
                                )
                                st.plotly_chart(fig_time, use_container_width=True)
                            
                            # Hourly distribution
                            df['hour'] = df['processed_date'].dt.hour
                            hourly_dist = df.groupby(['hour', 'sentiment']).size().reset_index(name='count')

                            if hourly_dist.empty:
                                st.warning("No hourly sentiment data to display.")
                            else:
                                fig_hourly = px.bar(
                                    hourly_dist,
                                    x='hour',
                                    y='count',
                                    color='sentiment',
                                    title="Hourly Distribution of Sentiments",
                                    labels={"count": "Number of Comments", "hour": "Hour of Day"},
                                    barmode="group"
                                )
                                st.plotly_chart(fig_hourly, use_container_width=True)

                        # ------------------- TAB 3: Detailed Analysis --------------------
                        with analysis_tabs[2]:
                            st.subheader("Detailed Comment Analysis")
                            
                            # Advanced filtering options (within the result set)
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                selected_sentiment = st.selectbox(
                                    "Filter by Sentiment",
                                    ["All"] + list(df['sentiment'].unique())
                                )
                                
                            with col2:
                                confidence_range = st.slider(
                                    "Confidence Range",
                                    0.0, 1.0, (0.5, 1.0)
                                )
                                
                            with col3:
                                sort_by = st.selectbox(
                                    "Sort By",
                                    ["Date", "Confidence", "Comment Length"]
                                )
                            
                            # Apply subfilters
                            filtered_df = df.copy()
                            if selected_sentiment != "All":
                                filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
                            
                            filtered_df = filtered_df[
                                (filtered_df['confidence'] >= confidence_range[0]) & 
                                (filtered_df['confidence'] <= confidence_range[1])
                            ]
                            
                            # Sorting
                            if sort_by == "Date":
                                filtered_df = filtered_df.sort_values('processed_date', ascending=False)
                            elif sort_by == "Confidence":
                                filtered_df = filtered_df.sort_values('confidence', ascending=False)
                            else:  # Comment Length
                                filtered_df['comment_length'] = filtered_df['comment'].astype(str).str.len()
                                filtered_df = filtered_df.sort_values('comment_length', ascending=False)
                            
                            if filtered_df.empty:
                                st.warning("No comments match your subfilters.")
                            else:
                                # Display results
                                st.dataframe(
                                    filtered_df[['commenter', 'processed_date', 'comment', 'sentiment', 'confidence']]
                                    .style.apply(highlight_sentiment, axis=1)
                                    .format({
                                        'processed_date': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                                        'confidence': '{:.2%}'
                                    })
                                    .set_properties(**{'text-align': 'left'})
                                    .hide_index(),
                                    height=400
                                )

                        # ------------------- TAB 4: Content Analysis --------------------
                        with analysis_tabs[3]:
                            st.subheader("Content Analysis")
                            
                            # Comment length distribution
                            length_series = df['comment'].astype(str).apply(len)
                            if length_series.empty:
                                st.warning("No valid comment lengths to plot.")
                            else:
                                fig_lengths = px.histogram(
                                    x=length_series,
                                    nbins=30,
                                    title="Comment Length Distribution",
                                    labels={'x': 'Comment Length (characters)', 'y': 'Count'}
                                )
                                st.plotly_chart(fig_lengths, use_container_width=True)
                            
                            # Most active commenters
                            st.subheader("Most Active Commenters")
                            top_commenters = df['commenter'].value_counts().head(10)
                            if top_commenters.empty:
                                st.warning("No commenters found in the filtered dataset.")
                            else:
                                fig_commenters = px.bar(
                                    x=top_commenters.index,
                                    y=top_commenters.values,
                                    title="Top 10 Most Active Commenters",
                                    labels={'x': 'Commenter', 'y': 'Number of Comments'}
                                )
                                st.plotly_chart(fig_commenters, use_container_width=True)

                        # ------------------- Export Options -------------------
                        st.subheader("Export Results")
                        export_col1, export_col2 = st.columns(2)
                        
                        with export_col1:
                            if st.button("Export to CSV"):
                                csv_data = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download CSV",
                                    csv_data,
                                    "sentiment_analysis_results.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                                
                        with export_col2:
                            if st.button("Export Analysis Report"):
                                # Example of a simple "report" string builder
                                report_str = (
                                    f"Sentiment Analysis Report\n"
                                    f"Total Comments: {len(df)}\n"
                                    f"Average Confidence: {df['confidence'].mean():.2%}\n"
                                    f"Sentiments:\n{df['sentiment'].value_counts()}\n"
                                    "\n---\n"
                                    "Further details could be added here..."
                                )
                                st.download_button(
                                    "Download Report",
                                    report_str,
                                    "sentiment_analysis_report.txt",
                                    "text/plain",
                                    key='download-report'
                                )

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        st.stop()

            # -------------------------------------------------------------------
            # Enhanced Feedback System with Storage
            # -------------------------------------------------------------------
            st.divider()
            st.subheader("Analysis Feedback & Quality Control")
            
            # Initialize feedback storage
            if 'feedback_file' not in st.session_state:
                st.session_state.feedback_file = 'feedback_data.json'
                if not os.path.exists(st.session_state.feedback_file):
                    with open(st.session_state.feedback_file, 'w', encoding='utf-8') as f:
                        json.dump([], f)

            feedback_container = st.container()
            with feedback_container:
                feedback_tabs = st.tabs([
                    "📝 Quick Feedback",
                    "📊 Detailed Feedback",
                    "❌ Error Reports",
                    "📈 Feedback Analytics"
                ])
                
                with feedback_tabs[0]:
                    col1, col2 = st.columns(2)
                    with col1:
                        quick_rating = st.slider(
                            "Rate the analysis quality",
                            1, 5, 3,
                            help="1 = Poor, 5 = Excellent"
                        )
                        
                        analysis_speed = st.select_slider(
                            "Analysis Speed",
                            options=["Very Slow", "Slow", "Average", "Fast", "Very Fast"],
                            value="Average"
                        )
                        
                    with col2:
                        satisfaction = st.select_slider(
                            "Overall Satisfaction",
                            options=["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
                            value="Neutral"
                        )
                        
                        would_recommend = st.radio(
                            "Would you recommend this tool?",
                            ["Yes", "Maybe", "No"]
                        )
                    
                    quick_comment = st.text_area(
                        "Quick Comment (Optional)",
                        height=100,
                        placeholder="Any immediate thoughts or suggestions?"
                    )
                
                with feedback_tabs[1]:
                    st.subheader("Detailed Analysis Feedback")
                    
                    # Specific aspect ratings
                    aspect_col1, aspect_col2 = st.columns(2)
                    
                    with aspect_col1:
                        sentiment_accuracy = st.slider(
                            "Sentiment Analysis Accuracy",
                            1, 5, 3,
                            help="How accurate were the sentiment predictions?"
                        )
                        
                        visualization_quality = st.slider(
                            "Visualization Quality",
                            1, 5, 3,
                            help="How helpful were the visualizations?"
                        )
                        
                        ui_experience = st.slider(
                            "User Interface Experience",
                            1, 5, 3,
                            help="How user-friendly was the interface?"
                        )
                    
                    with aspect_col2:
                        data_processing = st.slider(
                            "Data Processing Quality",
                            1, 5, 3,
                            help="How well was the data processed and analyzed?"
                        )
                        
                        feature_completeness = st.slider(
                            "Feature Completeness",
                            1, 5, 3,
                            help="How complete are the available features?"
                        )
                        
                        documentation_clarity = st.slider(
                            "Documentation Clarity",
                            1, 5, 3,
                            help="How clear were the instructions and documentation?"
                        )
                    
                    # Positive aspects
                    st.subheader("Strengths and Areas for Improvement")
                    
                    positive_aspects = st.multiselect(
                        "What worked well?",
                        [
                            "Sentiment Accuracy",
                            "Date Processing",
                            "Visualization Quality",
                            "Processing Speed",
                            "User Interface",
                            "Analysis Depth",
                            "Export Options",
                            "Error Handling",
                            "Documentation",
                            "Feature Set"
                        ],
                        default=["Sentiment Accuracy"]
                    )
                    
                    improvement_needed = st.multiselect(
                        "What needs improvement?",
                        [
                            "Sentiment Accuracy",
                            "Date Processing",
                            "Visualization Quality",
                            "Processing Speed",
                            "User Interface",
                            "Analysis Depth",
                            "Export Options",
                            "Error Handling",
                            "Documentation",
                            "Feature Set"
                        ]
                    )
                    
                    # Detailed feedback
                    st.subheader("Additional Feedback")
                    
                    detailed_feedback = st.text_area(
                        "Detailed comments or suggestions",
                        height=150,
                        placeholder="Please provide any specific feedback, suggestions, or feature requests..."
                    )
                    
                    # Feature requests
                    feature_requests = st.text_area(
                        "Feature Requests",
                        height=100,
                        placeholder="What additional features would you like to see?"
                    )
                
                with feedback_tabs[2]:
                    st.subheader("Error Reporting")
                    
                    error_type = st.selectbox(
                        "Type of Error",
                        [
                            "None",
                            "Sentiment Misclassification",
                            "Date Processing Error",
                            "Visualization Error",
                            "Performance Issue",
                            "Data Processing Error",
                            "UI/UX Issue",
                            "Export Error",
                            "Other"
                        ]
                    )
                    
                    if error_type != "None":
                        error_severity = st.select_slider(
                            "Error Severity",
                            options=["Low", "Medium", "High", "Critical"],
                            value="Medium"
                        )
                        
                        error_frequency = st.select_slider(
                            "Error Frequency",
                            options=["One-time", "Occasional", "Frequent", "Consistent"],
                            value="One-time"
                        )
                        
                        error_description = st.text_area(
                            "Error Description",
                            height=150,
                            help="Please provide specific examples and steps to reproduce the error"
                        )
                        
                        reproducible = st.checkbox("Is this error consistently reproducible?")
                        
                        if reproducible:
                            steps_to_reproduce = st.text_area(
                                "Steps to Reproduce",
                                height=150,
                                placeholder="1. Step one\n2. Step two\n3. Step three..."
                            )
                        
                        # File upload for error evidence
                        st.subheader("Error Evidence")
                        
                        evidence_type = st.multiselect(
                            "Type of Evidence",
                            ["Screenshot", "Log File", "Data Sample", "Other"]
                        )
                        
                        if "Screenshot" in evidence_type:
                            error_screenshot = st.file_uploader(
                                "Upload Screenshot",
                                type=['png', 'jpg', 'jpeg']
                            )
                        
                        if "Log File" in evidence_type:
                            error_log = st.file_uploader(
                                "Upload Log File",
                                type=['txt', 'log']
                            )
                        
                        if "Data Sample" in evidence_type:
                            data_sample = st.file_uploader(
                                "Upload Data Sample",
                                type=['csv', 'xlsx', 'json']
                            )
                
                with feedback_tabs[3]:
                    st.subheader("Feedback Analytics")
                    
                    # Load existing feedback data
                    try:
                        with open(st.session_state.feedback_file, 'r', encoding='utf-8') as f:
                            feedback_data = json.load(f)
                        
                        if feedback_data:
                            # Convert to DataFrame for analysis
                            feedback_df = pd.DataFrame(feedback_data)
                            
                            # Display overall metrics
                            metric_cols = st.columns(4)
                            
                            with metric_cols[0]:
                                avg_rating = feedback_df['quick_rating'].mean()
                                st.metric("Average Rating", f"{avg_rating:.2f}/5")
                            
                            with metric_cols[1]:
                                satisfaction_counts = feedback_df['satisfaction'].value_counts()
                                most_common = satisfaction_counts.index[0]
                                st.metric("Most Common Satisfaction", most_common)
                            
                            with metric_cols[2]:
                                recommend_pct = (feedback_df['would_recommend'] == 'Yes').mean() * 100
                                st.metric("Would Recommend", f"{recommend_pct:.1f}%")
                            
                            with metric_cols[3]:
                                total_feedback = len(feedback_df)
                                st.metric("Total Feedback Count", total_feedback)
                            
                            # Ratings over time
                            st.subheader("Ratings Trend")
                            feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
                            ratings_time = px.line(
                                feedback_df,
                                x='timestamp',
                                y='quick_rating',
                                title="Rating Trends Over Time"
                            )
                            st.plotly_chart(ratings_time, use_container_width=True)
                            
                            # Common issues wordcloud (placeholder logic)
                            if 'detailed_feedback' in feedback_df.columns:
                                st.subheader("Common Feedback Themes")
                                all_feedback = ' '.join(feedback_df['detailed_feedback'].dropna())
                                # (Implement a wordcloud or text analysis here if desired)
                            
                            # Error distribution
                            if 'error_type' in feedback_df.columns:
                                error_counts = feedback_df['error_type'].value_counts()
                                fig_errors = px.pie(
                                    values=error_counts.values,
                                    names=error_counts.index,
                                    title="Distribution of Reported Issues"
                                )
                                st.plotly_chart(fig_errors, use_container_width=True)
                        
                        else:
                            st.info("No feedback data available yet.")
                        
                    except Exception as e:
                        st.error(f"Error loading feedback data: {str(e)}")

            # Submit feedback button with improved validation and storage
            if st.button("Submit Comprehensive Feedback", type="primary"):
                if quick_rating > 0 and satisfaction:
                    try:
                        # Prepare feedback data
                        feedback_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "quick_rating": quick_rating,
                            "satisfaction": satisfaction,
                            "analysis_speed": analysis_speed,
                            "would_recommend": would_recommend,
                            "quick_comment": quick_comment,
                            "sentiment_accuracy": sentiment_accuracy if 'sentiment_accuracy' in locals() else None,
                            "visualization_quality": visualization_quality if 'visualization_quality' in locals() else None,
                            "ui_experience": ui_experience if 'ui_experience' in locals() else None,
                            "data_processing": data_processing if 'data_processing' in locals() else None,
                            "feature_completeness": feature_completeness if 'feature_completeness' in locals() else None,
                            "documentation_clarity": documentation_clarity if 'documentation_clarity' in locals() else None,
                            "positive_aspects": positive_aspects if 'positive_aspects' in locals() else [],
                            "improvement_needed": improvement_needed if 'improvement_needed' in locals() else [],
                            "detailed_feedback": detailed_feedback if 'detailed_feedback' in locals() else "",
                            "feature_requests": feature_requests if 'feature_requests' in locals() else "",
                            "error_type": error_type if 'error_type' in locals() else "None",
                            "error_severity": error_severity if 'error_severity' in locals() and error_type != "None" else None,
                            "error_frequency": error_frequency if 'error_frequency' in locals() and error_type != "None" else None,
                            "error_description": error_description if 'error_description' in locals() and error_type != "None" else None,
                            "reproducible": reproducible if 'reproducible' in locals() and error_type != "None" else None,
                            "steps_to_reproduce": steps_to_reproduce if 'steps_to_reproduce' in locals() and 'reproducible' in locals() and reproducible else None
                        }
                        
                        # Load existing feedback
                        with open(st.session_state.feedback_file, 'r', encoding='utf-8') as f:
                            existing_feedback = json.load(f)
                        
                        # Append new feedback
                        existing_feedback.append(feedback_entry)
                        
                        # Save updated feedback
                        with open(st.session_state.feedback_file, 'w', encoding='utf-8') as f:
                            json.dump(existing_feedback, f, ensure_ascii=False, indent=2)
                        
                        st.success("Thank you for your comprehensive feedback! Your input helps us improve the analysis system.")
                        
                        # Show feedback summary
                        st.write("Feedback Summary:")
                        summary_cols = st.columns(3)
                        with summary_cols[0]:
                            st.metric("Quality Rating", f"{quick_rating}/5")
                        with summary_cols[1]:
                            st.metric("Satisfaction", satisfaction)
                        with summary_cols[2]:
                            st.metric("Would Recommend", would_recommend)
                        
                    except Exception as e:
                        st.error(f"Error saving feedback: {str(e)}")
                else:
                    st.warning("Please provide at least a quality rating and satisfaction level.")


          
            
        with tabs[3]:
            #Political Topic Modeling with GPT
            st.subheader("Political Topic Modeling")
            st.write(
                """
                Enter text data (e.g., combined comments in Arabic, Darija, French, or English)
                to extract major political topics. This may incur OpenAI API costs.
                """
            )

            # Text area to enter or combine text data
            text_for_topics = st.text_area("Paste or summarize the text you want to analyze for topics:", "")

            if st.button("Identify Political Topics with GPT"):
                if text_for_topics.strip():
                    with st.spinner("GPT-based Topic Modeling in progress..."):
                        # Perform GPT topic modeling
                        topics = perform_gpt_topic_modeling(
                            text_data=text_for_topics,
                            model_name=gpt_model,  # from the sidebar
                            language_hint="Arabic/Darija/French/English",  # or a more specific guess
                            max_topics=5
                        )
                    if topics:
                        st.success("Topics Identified!")
                        for i, topic in enumerate(topics, start=1):
                            st.write(f"{i}. {topic}")
                    else:
                        st.warning("No topics found, or GPT returned an empty result.")
                else:
                    st.warning("Please enter or paste some text data.")
        with tabs[6]:
        
            st.subheader("Scrape More Data from Hespress")
            if st.button("Scrape"):
                if st.session_state.scraping_in_progress:
                    st.warning("Scraping is already in progress. Please wait.")
                else:
                    st.session_state.scraping_in_progress = True
                    st.info("Scraping in progress... This may take a few minutes.")
                
                
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Simulate incremental progress (e.g., 5 steps total)
                    # If your `run_scraper()` can provide real-time progress,
                    # integrate that logic instead of this static loop.
                    import time
                    total_steps = 5
                    for i in range(total_steps):
                        # Simulate some wait time
                        time.sleep(1)
                        progress_bar.progress(int((i + 1) * (100 / total_steps)))
                    
                    # Actually run your scraper function
                    future = executor.submit(run_scraper)
                    
                    # Clear the progress bar once done
                    progress_bar.empty()
                    
                    st.success("Scraping complete! The new data has been appended to the existing CSV files.")

                # OPTIONAL: Immediately reload the updated CSV if you want to reflect the new data in this session:
                try:
                    df = pd.read_csv("hespress_politics_comments.csv")
                    st.rerun()  # Force a rerun to refresh your dashboard with new data
                except Exception as e:
                    st.warning(f"Could not reload new data automatically: {e}")
        
        # Place chat input outside of restricted UI elements
        prompt = st.chat_input("Ask about Moroccan politics...")
        with tabs[7]:
            st.subheader("🤖 Interactive Politics Chatbot")
            
            # --------------------------------------------------------------------------------
            # 1. SIDEBAR MODEL CONFIGURATION
            #    (This is the ONLY place the user can change model or temperature)
            # --------------------------------------------------------------------------------
            st.sidebar.markdown("### Chatbot Configuration")
            model = st.sidebar.selectbox(
                "Select GPT Model",
                ["gpt-3.5-turbo", "gpt-4"],
                help="GPT-3.5 is faster and cheaper, GPT-4 is more capable"
            )
            
            temperature = st.sidebar.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher = more creative, Lower = more focused"
            )
            
            max_tokens = st.sidebar.number_input(
                "Max Response Length",
                min_value=50,
                max_value=2000,
                value=500,
                step=50,
                help="Maximum number of tokens in the response"
            )

            # Save the current config in session_state (used by both React & Native Chat)
            st.session_state["selected_model"] = model
            st.session_state["temperature"] = temperature
            st.session_state["max_tokens"] = max_tokens

            # --------------------------------------------------------------------------------
            # 2. ADD OPTIONAL STYLES FOR THE PAGE LAYOUT
            # --------------------------------------------------------------------------------
            st.markdown(
                """
                <style>
                .stApp {
                    max-width: 100%;
                    margin: 0 auto;
                }
                .chat-container {
                    max-width: 800px;
                    margin: 0 auto;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            from streamlit.components.v1 import html

            # --------------------------------------------------------------------------------
            # 3. REACT-BASED CHATBOT CODE
            # --------------------------------------------------------------------------------
         

            # --------------------------------------------------------------------------------
            # 5. STREAMLIT-SIDE PYTHON LOGIC TO HANDLE MESSAGES FROM THE REACT FRONT-END
            # --------------------------------------------------------------------------------
            if "openai_messages" not in st.session_state:
                st.session_state.openai_messages = []
            def preprocess_text(text):
                if pd.isna(text):
                    return ""
                text = str(text).lower()
                text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
                text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
                text = re.sub(r'\d+', '', text)  # Remove numbers
                text = ' '.join(text.split())  # Remove extra spaces
                return text
            # This function calls OpenAI, injecting relevant context from your df_main using RAG
            def handle_message(user_query):
                try:
                    # 1) Preprocess the user query
                    processed_query = preprocess_text(user_query)
                    
                    # 2) Generate embedding for the query
                    query_embedding = st.session_state.embedding_model.encode([processed_query], convert_to_numpy=True)
                    faiss.normalize_L2(query_embedding)
                    
                    # 3) Search for top 5 relevant comments using FAISS
                    top_k = 5
                    distances, indices = st.session_state.faiss_index.search(query_embedding, top_k)
                    
                    # 4) Retrieve the relevant comments
                    relevant_comments = st.session_state.df_main.iloc[indices[0]]['comment'].tolist()
                    
                    if not relevant_comments:
                        st.warning("No relevant comments found to provide context.")
                        context_str = "There are no relevant comments available to provide context for your query."
                    else:
                        context_str = "\n".join(relevant_comments)
                    
                    # 5) Build a system prompt incorporating the retrieved context
                    system_prompt = f"""
                    You are a knowledgeable assistant specialized in Moroccan politics.
                    The user asked: {user_query}

                    Here are some relevant user comments to draw context from:
                    {context_str}

                    Provide a thoughtful, balanced response that:
                    - Analyzes main themes/opinions
                    - Uses relevant examples from these comments
                    - Remains neutral and explains different viewpoints
                    """
                    
                    # 6) Construct messages for the OpenAI ChatCompletion
                    messages_for_api = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ]
                    
                    # 7) Call OpenAI
                    response = openai.ChatCompletion.create(
                        model=st.session_state.get("selected_model", "gpt-3.5-turbo"),
                        messages=messages_for_api,
                        temperature=st.session_state.get("temperature", 0.7),
                        max_tokens=st.session_state.get("max_tokens", 500)
                    )
                    
                    return response.choices[0].message["content"]
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return "Sorry, an error occurred."

            # --------------------------------------------------------------------------------
            # 6. (OPTIONAL) NATIVE STREAMLIT CHAT UI
            #    You can remove this if you only want the React front-end.
            # --------------------------------------------------------------------------------
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display existing conversation
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
    
            # Chat input
            if prompt :
                # Show user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Use the same handle_message logic for the native chat
                assistant_response = handle_message(prompt)

                # Show assistant reply
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)



 
        with tabs[11]:
            if "sentiment" in df.columns:
                st.subheader("Download Results")
                st.write("Click the button below to download the analyzed dataset as a CSV.")
                download_csv(df, filename="hespress_analyzed_comments.csv")


if __name__ == "__main__":
    main()
