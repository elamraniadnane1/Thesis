import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
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



# ------------------------------------------------------
# 1. Custom CSS (Optional)
# ------------------------------------------------------
st.set_page_config(
    page_title="Social Media Analysis Tool 2025, By: Adnane El Amrani",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Import Lucide icons CSS */
    @import url('https://cdn.jsdelivr.net/npm/lucide-static@0.16.29/font/lucide.min.css');
    
    /* Main layout and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f0f4f8 0%, #d7e3fc 100%);
        color: #1a365d;
    }
    
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

# ------------------------------------------------------
# 5. Helper for table highlighting based on sentiment
# ------------------------------------------------------
def highlight_sentiment(row):
    """
    Returns a background color for each row based on the sentiment value.
    Adjust the colors to your preference.
    """
    color = "white"
    if row["sentiment"] == "POS":
        color = "#d8f5d8"  # light green
    elif row["sentiment"] == "NEG":
        color = "#fddddd"  # light red
    elif row["sentiment"] == "NEU":
        color = "#f9f9c5"  # light yellow
    return [f"background-color: {color}"] * len(row)

# --- ADD AFTER LINE 110 ---

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


def main():
    st.title("Social Media Analysis Tool 2025, By : Adnane El Amrani")
    st.write(
        """
        This dashboard uses an **Arabic BERT** model 
        ([`CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment)) 
        to analyze the sentiment of comments. 
        Upload your CSV file or use the default example to get started.
        """
    )

    # Sidebar options
    with st.sidebar:
        st.header("App Configuration")
        show_heatmap = st.checkbox("Show Sample Heatmap", value=False)
        show_top_commenters = st.checkbox("Show Top Commenters Bar Chart", value=True)
        top_n_commenters = st.slider("Number of Top Commenters to Display", 5, 30, 10)
        
        st.markdown("---")

        st.write("GPT Model Selection")
        gpt_model = st.selectbox(
            "Choose GPT Model:",
            options=["gpt-3.5-turbo", "gpt-4", "text-davinci-003"],
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

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = ["article_url", "commenter", "comment_date", "comment"]
        st.success("File uploaded successfully!")
    else:
        st.info("No file uploaded. Using the Hespress CSV file stored in our Database.")
        df = load_data(csv_file_path)
    

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
                            return pd.to_datetime('2024-01-01')

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
                        df = analyze_sentiment_in_chunks(df, sentiment_pipe, chunk_size=1000)
                    else:
                        df = analyze_sentiment_in_chunks(df, sentiment_pipe, chunk_size=chunk_size)
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

                # Plotly Pie Chart
                plot_sentiment_pie_chart(df)

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

                        # Function to parse Arabic dates
                        def parse_arabic_date(date_str):
                            parsed_date = dateparser.parse(
                                date_str,
                                languages=['ar'],
                                settings={
                                    'DATE_ORDER': 'DMY',
                                    'TIMEZONE': 'UTC',
                                    'RETURN_AS_TIMEZONE_AWARE': False
                                }
                            )
                            return parsed_date

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
                            df = df.dropna(subset=['parsed_date'])

                            # Update 'comment_date' with parsed dates
                            df['comment_date'] = df['parsed_date']
                            df = df.drop(columns=['parsed_date', 'comment_date_corrected'])

                            # Ensure 'comment_date' is datetime
                            df['comment_date'] = pd.to_datetime(df['comment_date'])

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
                run_scraper()
                
                # Clear the progress bar once done
                progress_bar.empty()
                
                st.success("Scraping complete! The new data has been appended to the existing CSV files.")

                # OPTIONAL: Immediately reload the updated CSV if you want to reflect the new data in this session:
                try:
                    df = pd.read_csv("hespress_politics_comments.csv")
                    st.experimental_rerun()  # Force a rerun to refresh your dashboard with new data
                except Exception as e:
                    st.warning(f"Could not reload new data automatically: {e}")

 
        with tabs[11]:
            if "sentiment" in df.columns:
                st.subheader("Download Results")
                st.write("Click the button below to download the analyzed dataset as a CSV.")
                download_csv(df, filename="hespress_analyzed_comments.csv")


if __name__ == "__main__":
    main()
