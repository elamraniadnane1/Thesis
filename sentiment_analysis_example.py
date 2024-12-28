from transformers import pipeline

# Load the sentiment analysis pipeline with the specified model
pipe = pipeline("text-classification", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")

def analyze_sentiment(texts):
    """
    Analyze sentiment for a list of texts.

    Parameters:
        texts (list of str): List of text inputs to analyze.

    Returns:
        list of dict: List of sentiment analysis results for each text.
    """
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings.")

    results = pipe(texts)
    return results

if __name__ == "__main__":
    # Sample Arabic texts for sentiment analysis
    sample_texts = [
        "\u0635\u0648\u062a\u0643 \u062c\u0645\u064a\u0644 \u062c\u062f\u0627 \u0648\u0645\u0644\u0647\u0645.",  # "Your voice is very beautiful and inspiring."
        "\u0644\u0627 \u0623\u062d\u0628 \u0647\u0630\u0627 \u0627\u0644\u0645\u0643\u0627\u0646.",  # "I don't like this place."
        "\u0627\u0644\u062e\u062f\u0645\u0629 \u0645\u0645\u062a\u0627\u0632\u0629 \u0648\u0627\u0644\u0637\u0639\u0627\u0645 \u0644\u0630\u064a\u0630 \u062c\u062f\u064b\u0627."  # "The service is excellent and the food is very delicious."
    ]

    # Perform sentiment analysis
    sentiments = analyze_sentiment(sample_texts)

    # Print the results
    for text, sentiment in zip(sample_texts, sentiments):
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}\n")
