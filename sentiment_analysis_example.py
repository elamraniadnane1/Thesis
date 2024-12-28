import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Initialize the sentiment analysis pipeline
# ------------------------------------------------------
@st.cache_resource
def load_sentiment_pipeline():
    # This model performs sentiment classification on Arabic text
    sentiment_pipe = pipeline(
        "text-classification", 
        model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    )
    return sentiment_pipe

# ------------------------------------------------------
# 2. Load data
# ------------------------------------------------------
@st.cache_data
def load_data(csv_file_path: str):
    """
    Load the comments CSV into a pandas DataFrame.
    Expected CSV format:
    article_url, commenter, comment_date, comment
    """
    df = pd.read_csv(csv_file_path)
    # Clean up or rename columns if needed
    df.columns = ["article_url", "commenter", "comment_date", "comment"]
    return df

# ------------------------------------------------------
# 3. Perform sentiment analysis
# ------------------------------------------------------
def analyze_sentiment(df: pd.DataFrame, sentiment_pipe):
    sentiments = []
    for comment in df['comment']:
        # The pipeline returns a list of dicts, e.g. [{'label': 'POSITIVE', 'score': 0.954...}]
        # For the CAMeL model, labels might be "POS", "NEG", or "NEU" (depending on the model).
        # Check the model documentation or do a quick test to see the exact label outputs.
        result = sentiment_pipe(comment)
        sentiments.append(result[0]['label'])
    
    df['sentiment'] = sentiments
    return df

# ------------------------------------------------------
# 4. Main Streamlit App
# ------------------------------------------------------
def main():
    st.set_page_config(page_title="Hespress Sentiment Analysis", layout="wide")

    st.title("Arabic Sentiment Analysis Dashboard")
    st.write(
        """
        This dashboard uses an Arabic BERT model 
        ([`CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment)) 
        to analyze the sentiment of comments. 
        Upload your CSV file or use the default example to get started.
        """
    )

    # ------------------------------------------------------
    # 4a. File upload
    # ------------------------------------------------------
    uploaded_file = st.file_uploader("Upload a CSV file with columns (article_url, commenter, comment_date, comment):", type="csv")

    # Default path (update if needed)
    csv_file_path = "C:\\Users\\DELL\\OneDrive\\Desktop\\Thesis\\hespress_politics_comments.csv"

    # Load pipeline once
    sentiment_pipe = load_sentiment_pipeline()

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = ["article_url", "commenter", "comment_date", "comment"]
        st.success("File uploaded successfully!")
    else:
        # Use the default CSV file if no file is uploaded
        st.info("No file uploaded. Using the default CSV file.")
        df = load_data(csv_file_path)

    # ------------------------------------------------------
    # 4b. Show raw data
    # ------------------------------------------------------
    if df is not None:
        st.subheader("1. Preview of the Data")
        st.dataframe(df.head(10))

        # ------------------------------------------------------
        # 4c. Run sentiment analysis
        # ------------------------------------------------------
        with st.spinner("Analyzing sentiment..."):
            df = analyze_sentiment(df, sentiment_pipe)

        st.subheader("2. Sentiment Analysis Results")
        st.dataframe(df.head(10))

        # ------------------------------------------------------
        # 4d. Visualization
        # ------------------------------------------------------
        st.subheader("3. Sentiment Distribution")

        sentiment_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        sentiment_counts.plot(
            kind='bar',
            color=['green', 'red', 'blue'],
            ax=ax
        )
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Sentiments in Comments")
        st.pyplot(fig)

        # ------------------------------------------------------
        # 4e. Possible grouping or filtering
        # ------------------------------------------------------
        st.subheader("4. Filter by Sentiment")
        selected_sentiment = st.selectbox("Choose a sentiment to filter:", ["ALL"] + list(sentiment_counts.index))

        if selected_sentiment != "ALL":
            filtered_df = df[df['sentiment'] == selected_sentiment]
        else:
            filtered_df = df

        st.write(f"Showing comments for sentiment: **{selected_sentiment}**")
        st.dataframe(filtered_df[['commenter', 'comment_date', 'comment', 'sentiment']])

        # ------------------------------------------------------
        # 4f. Conclusion / Insights
        # ------------------------------------------------------
        st.subheader("5. Observations / Insights")
        st.write(
            """
            - Use the above table and chart to identify common themes 
              or patterns in citizen comments.
            - Focus on negative or neutral sentiments to understand areas 
              of dissatisfaction or potential improvements.
            - The government can prioritize policies or decisions 
              that address the most prominent concerns.
            """
        )

if __name__ == "__main__":
    main()
