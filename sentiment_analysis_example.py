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

# ------------------------------------------------------
# 1. Custom CSS (Optional)
# ------------------------------------------------------
st.set_page_config(page_title="Hespress Sentiment Analysis", layout="wide")
st.markdown(
    """
    <style>
    /* Custom fonts and background */
    body {
        background-color: #F8F9FA;
    }
    .title h1, .title h2, .title h3, .title h4, .title h5, .title h6 {
        color: #2B3E50;
    }
    .css-18e3th9 {
        padding: 1rem 2rem 2rem 2rem; 
    }
    /* Table hover and header color */
    table.dataframe tbody tr:hover {
        background-color: #EEE !important;
    }
    table.dataframe thead {
        background-color: #395B64;
        color: #F8F9FA;
    }
    .stSpinner > div > div {
        border-top-color: #5CDB94 !important;
    }
    /* Custom button styling */
    div.stButton > button {
        color: white;
        background: #3AAFA9;
        border-radius: 0.5rem;
        height: 3rem;
        font-size: 1rem;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        color: white;
        background: #2B7A78;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# ------------------------------------------------------
# 6. Additional Visualizations / Analyses
# ------------------------------------------------------
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

# ------------------------------------------------------
# 7. Main Streamlit App
# ------------------------------------------------------
def main():
    st.title("Hespress Sentiment Analysis Dashboard")
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
        
        st.write("**Choose Analysis Method**")
        analysis_method = st.radio(
            "Select how to run sentiment analysis:",
            ("Standard", "Chunked (Recommended for large datasets)")
        )
        chunk_size = st.slider("Chunk Size (if using chunked analysis)", 50, 1000, 100, step=50)

        st.write("---")
        st.write("**Data Upload**")

    # ------------------------------------------------------
    # 7a. File upload
    # ------------------------------------------------------
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
        st.info("No file uploaded. Using the default example CSV file.")
        df = load_data(csv_file_path)

    if df is not None:
        # ------------------------------------------------------
        # 7b. Data Preview
        # ------------------------------------------------------
        st.subheader("1. Preview of the Data")
        st.write("Here is a quick glance at the first 10 rows:")
        st.dataframe(df.head(10))

        # ------------------------------------------------------
        # 7c. Sentiment Analysis
        # ------------------------------------------------------
        st.subheader("2. Sentiment Analysis")
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

        # ------------------------------------------------------
        # 7d. Post-Analysis Visualizations
        # ------------------------------------------------------
        if "sentiment" in df.columns:
            # 3. Sentiment Distribution
            st.subheader("3. Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()

            # 3a. Traditional Matplotlib bar chart
            fig, ax = plt.subplots()
            sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            ax.set_title("Bar Chart: Distribution of Sentiments in Comments")
            st.pyplot(fig)

            # 3b. Plotly Pie Chart
            plot_sentiment_pie_chart(df)

            # 4. Top Commenters
            if show_top_commenters:
                st.subheader("4. Top Commenters")
                top_commenters_bar_chart(df, top_n=top_n_commenters)

            # 5. Filter by sentiment
            st.subheader("5. Filter / Search by Sentiment")
            selected_sentiment = st.selectbox("Choose a sentiment to filter:", ["ALL"] + list(sentiment_counts.index))

            if selected_sentiment != "ALL":
                filtered_df = df[df['sentiment'] == selected_sentiment]
            else:
                filtered_df = df

            st.write(f"Showing comments for sentiment: **{selected_sentiment}**")
            st.dataframe(
                filtered_df[['commenter', 'comment_date', 'comment', 'sentiment']].style.apply(highlight_sentiment, axis=1)
            )

        # 6. Observations / Insights
        st.subheader("6. Observations / Insights")
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

        # 7. Optional Heatmap
        if show_heatmap:
            st.subheader("Sample Correlation Heatmap")
            st.write(
                "This is an example heatmap for demonstration, based on random numeric data."
            )
            plot_heatmap(df)

        # 8. Custom Comment Analysis
        st.subheader("7. Try It Yourself: Custom Comment Analysis")
        user_comment = st.text_area("Enter an Arabic comment to analyze its sentiment:")
        analyze_btn = st.button("Analyze This Comment")
        if analyze_btn:
            if user_comment.strip():
                with st.spinner("Analyzing..."):
                    user_result = sentiment_pipe(user_comment)
                    # If pipeline returns a single dict like: {'label': 'POS', 'score': 0.99}
                    user_sentiment = user_result['label']
                st.write(f"**Predicted Sentiment**: {user_sentiment}")
            else:
                st.warning("Please enter a non-empty comment.")

        # 9. Time-based sentiment analysis
        if "sentiment" in df.columns:
            st.subheader("8. Time-Based Sentiment Analysis")
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



        # ------------------------------------------------------
        # 10. Word Cloud Visualization
        # ------------------------------------------------------
        if "sentiment" in df.columns:
            st.subheader("9. Word Cloud")
            
            # Retrieve unique sentiments and sort them for better UX
            sentiment_options = sorted(df['sentiment'].unique())
            
            # Multi-select widget for choosing multiple sentiments
            selected_sentiments = st.multiselect(
                "Generate word cloud for which sentiment(s)?",
                options=sentiment_options,
                default=sentiment_options  # Default to all sentiments selected
            )
            
            # Determine the text data based on user selection
            if selected_sentiments:
                text_data = " ".join(
                    df[df['sentiment'].isin(selected_sentiments)]['comment'].astype(str)
                )
            else:
                st.warning("Please select at least one sentiment to generate a word cloud.")
                text_data = ""
            
            # Button to generate the word cloud
            if st.button("Generate Word Cloud"):
                if text_data.strip():
                    # Generate the word cloud
                    wordcloud = WordCloud(
                        background_color='white',
                        width=800,
                        height=600,
                        font_path='arial.ttf',  # Specify a font path if needed
                        collocations=False,     # Avoid duplicate words
                        stopwords=set([])       # Define stopwords if necessary
                    ).generate(text_data)
                    
                    # Display the word cloud using Matplotlib
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    ax3.imshow(wordcloud, interpolation='bilinear')
                    ax3.axis("off")  # Hide the axes
                    st.pyplot(fig3)
                else:
                    st.warning("No text available to generate a word cloud.")


        # 11. Downloading the Results
        if "sentiment" in df.columns:
            st.subheader("10. Download Results")
            st.write("Click the button below to download the analyzed dataset as a CSV.")
            download_csv(df, filename="hespress_analyzed_comments.csv")


if __name__ == "__main__":
    main()
