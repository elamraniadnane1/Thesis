import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # Make sure to install: pip install wordcloud

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
        # The CAMeL model might label them as "POS", "NEG", or "NEU".
        result = sentiment_pipe(comment)
        sentiments.append(result[0]['label'])
    
    df['sentiment'] = sentiments
    return df

# ------------------------------------------------------
# 4. Helper for table highlighting based on sentiment
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
# 5. Main Streamlit App
# ------------------------------------------------------
def main():
    st.set_page_config(page_title="Hespress Sentiment Analysis", layout="wide")

    st.title("Hespress Sentiment Analysis Dashboard")
    st.write(
        """
        This dashboard uses an Arabic BERT model 
        ([`CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment)) 
        to analyze the sentiment of comments. 
        Upload your CSV file or use the default example to get started.
        """
    )

    # ------------------------------------------------------
    # 5a. File upload
    # ------------------------------------------------------
    uploaded_file = st.file_uploader("Upload a CSV file with columns (article_url, commenter, comment_date, comment):", type="csv")

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
        # Use the default CSV file if no file is uploaded
        st.info("No file uploaded. Using the default CSV file.")
        df = load_data(csv_file_path)

    # ------------------------------------------------------
    # 5b. Show raw data
    # ------------------------------------------------------
    if df is not None:
        st.subheader("1. Preview of the Data")
        st.dataframe(df.head(10))

        # ------------------------------------------------------
        # 5c. Run sentiment analysis
        # ------------------------------------------------------
        with st.spinner("Analyzing sentiment..."):
            df = analyze_sentiment(df, sentiment_pipe)

        st.subheader("2. Sentiment Analysis Results")
        # Use a stylistic approach to highlight rows by sentiment
        st.dataframe(df.head(10).style.apply(highlight_sentiment, axis=1))

        # ------------------------------------------------------
        # 5d. Visualization - Overall sentiment distribution
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
        # 5e. Filter by sentiment
        # ------------------------------------------------------
        st.subheader("4. Filter by Sentiment")
        selected_sentiment = st.selectbox("Choose a sentiment to filter:", ["ALL"] + list(sentiment_counts.index))

        if selected_sentiment != "ALL":
            filtered_df = df[df['sentiment'] == selected_sentiment]
        else:
            filtered_df = df

        st.write(f"Showing comments for sentiment: **{selected_sentiment}**")
        st.dataframe(filtered_df[['commenter', 'comment_date', 'comment', 'sentiment']].style.apply(highlight_sentiment, axis=1))

        # ------------------------------------------------------
        # 5f. Observations / Insights
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

        # ------------------------------------------------------
        # 5g. Try it yourself: Analyze a custom comment
        # ------------------------------------------------------
        st.subheader("6. Try It Yourself: Custom Comment Analysis")
        user_comment = st.text_area("Enter an Arabic comment to analyze its sentiment:")
        if st.button("Analyze Comment"):
            if user_comment.strip():
                with st.spinner("Analyzing..."):
                    user_result = sentiment_pipe(user_comment)
                    user_sentiment = user_result[0]['label']
                st.write(f"**Predicted Sentiment**: {user_sentiment}")
            else:
                st.warning("Please enter a non-empty comment.")

        # ------------------------------------------------------
        # 5h. Time-based sentiment analysis
        # ------------------------------------------------------
        st.subheader("7. Time-Based Sentiment Analysis")
        # Convert comment_date to datetime (assuming YYYY-MM-DD or similar format)
        try:
            df['comment_date'] = pd.to_datetime(df['comment_date'], errors='coerce')
            # Drop NaT rows if any
            df = df.dropna(subset=['comment_date'])

            # Group by date and sentiment
            daily_counts = df.groupby([df['comment_date'].dt.date, 'sentiment']).size().reset_index(name='count')

            # Pivot so that each sentiment is a column
            pivoted = daily_counts.pivot(index='comment_date', columns='sentiment', values='count').fillna(0)

            st.write("Daily counts of each sentiment:")
            st.dataframe(pivoted.style.highlight_max(color='lightgreen', axis=0))

            # Plot the trends over time (stacked area or line)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            pivoted.plot(kind='line', ax=ax2, marker='o')
            ax2.set_title("Sentiment Counts Over Time")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Could not parse dates. Error: {e}")

        # ------------------------------------------------------
        # 5i. Word Cloud Visualization
        # ------------------------------------------------------
        st.subheader("8. Word Cloud")
        sentiment_options = ["ALL"] + list(sentiment_counts.index)
        sentiment_choice = st.selectbox("Generate word cloud for which sentiment?", sentiment_options)

        if sentiment_choice == "ALL":
            text_data = " ".join(df['comment'].astype(str))
        else:
            text_data = " ".join(df[df['sentiment'] == sentiment_choice]['comment'].astype(str))

        if st.button("Generate Word Cloud"):
            if text_data.strip():
                wordcloud = WordCloud(
                    background_color='white',
                    width=800,
                    height=600
                ).generate(text_data)

                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.imshow(wordcloud, interpolation='bilinear')
                ax3.axis("off")
                st.pyplot(fig3)
            else:
                st.warning("No text available to generate a word cloud.")

if __name__ == "__main__":
    main()
