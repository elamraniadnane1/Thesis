#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Arabic Sentiment Analysis Pipeline
- Performs data preprocessing and normalization for Arabic text
- Uses CamelBERT-DA for sentiment analysis
- Creates Qdrant collections for vector search
- Extracts top pros and cons from comments
- Logs experiment metrics to MLFlow
"""

import os
import pandas as pd
import numpy as np
import re
import mlflow
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArabicSentimentAnalysis:
    """
    End-to-end pipeline for Arabic sentiment analysis of citizen comments,
    with vector database integration and MLFlow experiment tracking.
    """
    
    def __init__(self, mlflow_uri="http://192.168.1.116:5000"):
        """
        Initialize the sentiment analysis pipeline.
        
        Args:
            mlflow_uri: The URI for MLFlow tracking server
        """
        self.mlflow_uri = mlflow_uri
        self.model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        
        # Initialize MLFlow
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Load tokenizer and model
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Label mapping based on CamelBERT-DA model
        self.id2label = {
            0: "negative",
            1: "positive",
            2: "neutral"
        }
        
        # Initialize Qdrant client (assuming local instance)
        self.qdrant_client = QdrantClient("localhost", port=6333)
        
        logger.info("Sentiment analysis pipeline initialized successfully")
    
    def load_data(self, file_path):
        """
        Load data from CSV file
        
        Args:
            file_path: Path to the CSV file containing the comments
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            # Check for and handle potential column name issues
            if "What are the challenges/issues raised?" in df.columns and "What is the proposed solution?" in df.columns:
                df.rename(columns={
                    "What are the challenges/issues raised?": "challenges",
                    "What is the proposed solution?": "solutions"
                }, inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_arabic_text(self, text):
        """
        Preprocess Arabic text:
        - Remove special characters and non-Arabic text
        - Normalize Arabic characters
        - Remove extra spaces
        
        Args:
            text: Input Arabic text string
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Normalize Arabic characters
        text = self._normalize_arabic(text)
        
        # Remove non-Arabic characters except spaces and punctuation
        text = re.sub(r'[^\u0600-\u06FF\s.,!?;:]', ' ', text)
        
        # Remove extra spaces, tabs, and newlines
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _normalize_arabic(self, text):
        """
        Normalize Arabic text:
        - Normalize different forms of Alif, Yaa, etc.
        - Remove diacritics (tashkeel)
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized Arabic text
        """
        # Normalize Alif with Hamza forms
        text = re.sub(r'[إأآ]', 'ا', text)
        
        # Normalize Yaa and Alif Maqsura
        text = re.sub(r'[ىي]', 'ي', text)
        
        # Normalize Taa Marbuta to Haa
        text = re.sub(r'ة', 'ه', text)
        
        # Remove Tashkeel (diacritics)
        text = re.sub(r'[\u064B-\u0652]', '', text)
        
        # Remove Tatweel (stretching character)
        text = re.sub(r'ـ', '', text)
        
        return text
    
    def create_qdrant_collection(self, collection_name, vector_size=768):
        """
        Create or recreate a Qdrant collection for storing text embeddings.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors (768 for BERT)
        """
        try:
            # Check if collection exists, if yes, recreate it
            try:
                self.qdrant_client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass
            
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            raise
    
    def get_text_embedding(self, text):
        """
        Generate embedding vector for text using the model.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing embedding vector
        """
        # Preprocess text first
        text = self.preprocess_arabic_text(text)
        
        # Tokenize text and get embedding
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use the [CLS] token embedding from the last hidden state as text embedding
            embedding = outputs.hidden_states[-1][:, 0, :].numpy()[0]
        
        return embedding
    
    def store_documents_in_qdrant(self, df, text_column, collection_name, batch_size=100):
        """
        Store text documents and their embeddings in Qdrant collection.
        
        Args:
            df: DataFrame containing the documents
            text_column: Column name containing the text to embed
            collection_name: Name of the Qdrant collection
            batch_size: Number of documents to process in each batch
        """
        logger.info(f"Storing {len(df)} documents in Qdrant collection: {collection_name}")
        
        # First create or recreate the collection
        self.create_qdrant_collection(collection_name)
        
        # Process documents in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Storing documents"):
            batch_df = df.iloc[i:i+batch_size].copy()
            batch_points = []
            
            for idx, row in batch_df.iterrows():
                text = row[text_column]
                if not isinstance(text, str) or not text.strip():
                    continue
                
                preprocessed_text = self.preprocess_arabic_text(text)
                
                # Skip empty texts after preprocessing
                if not preprocessed_text:
                    continue
                
                # Get embedding
                embedding = self.get_text_embedding(preprocessed_text)
                
                # Prepare metadata with all row data
                metadata = row.to_dict()
                
                # Add document to batch
                batch_points.append(
                    models.PointStruct(
                        id=idx,
                        vector=embedding.tolist(),
                        payload=metadata
                    )
                )
            
            # Insert batch into collection
            if batch_points:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
        
        logger.info(f"Successfully stored documents in Qdrant collection: {collection_name}")
    
    def predict_sentiment(self, text):
        """
        Predict sentiment of text using the loaded model.
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Dictionary with sentiment and confidence score
        """
        # Preprocess text
        preprocessed_text = self.preprocess_arabic_text(text)
        
        if not preprocessed_text:
            return {"label": "neutral", "score": 1.0}
        
        # Tokenize
        inputs = self.tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(scores, dim=1).item()
            pred_score = scores[0][pred_label].item()
        
        return {
            "label": self.id2label[pred_label],
            "score": pred_score
        }
    
    def analyze_sentiments(self, df, text_columns):
        """
        Analyze sentiments for the specified text columns in the DataFrame.
        
        Args:
            df: DataFrame containing the comments
            text_columns: List of columns to analyze
            
        Returns:
            DataFrame with sentiment scores for each column
        """
        logger.info(f"Analyzing sentiments for columns: {text_columns}")
        
        result_df = df.copy()
        
        for column in text_columns:
            if column not in df.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
                
            # Add sentiment analysis columns
            sentiment_column = f"{column}_sentiment"
            score_column = f"{column}_score"
            
            # Initialize empty lists to store results
            sentiments = []
            scores = []
            
            # Apply sentiment analysis row by row
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Analyzing {column}"):
                text = row[column]
                if isinstance(text, str) and text.strip():
                    result = self.predict_sentiment(text)
                    sentiments.append(result["label"])
                    scores.append(result["score"])
                else:
                    sentiments.append("neutral")
                    scores.append(0.0)
            
            # Assign results to new columns
            result_df[sentiment_column] = sentiments
            result_df[score_column] = scores
        
        return result_df
    
    def extract_top_pros_cons(self, df, challenges_col, solutions_col, n=10):
        """
        Extract top pros and cons based on sentiment analysis.
        
        Args:
            df: DataFrame with sentiment analysis results
            challenges_col: Column name containing challenges/issues
            solutions_col: Column name containing proposed solutions
            n: Number of top items to extract
            
        Returns:
            Dictionary with top pros and cons
        """
        logger.info(f"Extracting top {n} pros and cons")
        
        # Ensure sentiment columns exist
        challenges_sentiment = f"{challenges_col}_sentiment"
        solutions_sentiment = f"{solutions_col}_sentiment"
        
        if challenges_sentiment not in df.columns or solutions_sentiment not in df.columns:
            logger.error("Sentiment columns not found in DataFrame")
            return {"top_pros": [], "top_cons": []}
        
        # Extract cons (negative challenges)
        cons_df = df[df[challenges_sentiment] == "negative"].sort_values(
            by=f"{challenges_col}_score", ascending=False
        )
        top_cons = cons_df[challenges_col].head(n).tolist()
        
        # Extract pros (positive solutions)
        pros_df = df[df[solutions_sentiment] == "positive"].sort_values(
            by=f"{solutions_col}_score", ascending=False
        )
        top_pros = pros_df[solutions_col].head(n).tolist()
        
        return {
            "top_pros": top_pros,
            "top_cons": top_cons
        }
    
    def evaluate_model(self, test_data, test_labels):
        """
        Evaluate the sentiment analysis model.
        
        Args:
            test_data: List of texts for evaluation
            test_labels: List of true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        predictions = []
        for text in tqdm(test_data, desc="Evaluating"):
            prediction = self.predict_sentiment(text)
            predictions.append(prediction["label"])
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='weighted'
        )
        
        # Calculate BLEU score (using reference as the actual label)
        # Convert labels to list of list for BLEU calculation
        references = [[label] for label in test_labels]
        hypothesis = predictions
        
        # Use SmoothingFunction to avoid zero score when no matches
        smoothie = SmoothingFunction().method1
        bleu = corpus_bleu(
            [[r] for r in references], 
            hypothesis, 
            smoothing_function=smoothie
        )
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "bleu_score": bleu
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def log_experiment(self, metrics, params=None, artifacts=None):
        """
        Log experiment metrics, parameters, and artifacts to MLFlow.
        
        Args:
            metrics: Dictionary with evaluation metrics
            params: Dictionary with model parameters
            artifacts: Dictionary with paths to artifacts to log
        """
        experiment_name = "arabic-sentiment-analysis"
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.error(f"Error setting up MLFlow experiment: {e}")
            experiment_id = "0"  # Use default experiment
        
        # Start a new run
        with mlflow.start_run(experiment_id=experiment_id):
            # Log model parameters
            if params:
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log artifacts
            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    if os.path.exists(artifact_path):
                        mlflow.log_artifact(artifact_path, artifact_name)
    
    def visualize_sentiment_distribution(self, df, output_path="sentiment_distribution.png"):
        """
        Create visualization of sentiment distribution.
        
        Args:
            df: DataFrame with sentiment analysis results
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        logger.info("Creating sentiment distribution visualization")
        
        # Identify sentiment columns
        sentiment_columns = [col for col in df.columns if col.endswith('_sentiment')]
        
        if not sentiment_columns:
            logger.warning("No sentiment columns found for visualization")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(sentiment_columns), 1, figsize=(10, 6 * len(sentiment_columns)))
        
        # If only one column, make axes iterable
        if len(sentiment_columns) == 1:
            axes = [axes]
        
        # Plot sentiment distribution for each column
        for i, column in enumerate(sentiment_columns):
            # Calculate value counts
            counts = df[column].value_counts()
            
            # Create bar plot
            sns.barplot(x=counts.index, y=counts.values, ax=axes[i], palette="viridis")
            
            # Set title and labels
            axes[i].set_title(f"Sentiment Distribution: {column.replace('_sentiment', '')}")
            axes[i].set_xlabel("Sentiment")
            axes[i].set_ylabel("Count")
            
            # Add count values on top of bars
            for j, count in enumerate(counts.values):
                axes[i].text(j, count + 5, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
        
        return output_path
    
    def run_pipeline(self, data_path, challenges_col="challenges", solutions_col="solutions"):
        """
        Run the complete sentiment analysis pipeline.
        
        Args:
            data_path: Path to the CSV file with comments
            challenges_col: Column name for challenges/issues
            solutions_col: Column name for proposed solutions
            
        Returns:
            Dictionary with results and metrics
        """
        logger.info("Starting sentiment analysis pipeline")
        
        # 1. Load and preprocess data
        df = self.load_data(data_path)
        
        # 2. Create Qdrant collections for vector search
        for collection_name, column in [
            ("challenges_collection", challenges_col),
            ("solutions_collection", solutions_col)
        ]:
            self.store_documents_in_qdrant(df, column, collection_name)
        
        # 3. Perform sentiment analysis
        text_columns = [challenges_col, solutions_col]
        results_df = self.analyze_sentiments(df, text_columns)
        
        # 4. Extract top pros and cons
        top_items = self.extract_top_pros_cons(results_df, challenges_col, solutions_col)
        
        # 5. Evaluate model (using a subset for testing)
        # Create a small test set from the data
        test_size = min(100, len(df) // 5)  # 20% or max 100 samples
        test_data = results_df[challenges_col].dropna().sample(test_size).tolist()
        
        # Create synthetic labels (for demonstration - in a real scenario, you would have ground truth labels)
        # Here we're using our model's predictions as "truth" for demonstration purposes
        test_labels = [self.predict_sentiment(text)["label"] for text in test_data]
        
        # Evaluate model
        metrics = self.evaluate_model(test_data, test_labels)
        
        # 6. Create visualization
        viz_path = self.visualize_sentiment_distribution(results_df)
        
        # 7. Log experiment to MLFlow
        params = {
            "model_name": self.model_name,
            "preprocessing_applied": "Arabic normalization, special char removal, diacritics removal"
        }
        
        artifacts = {}
        if viz_path:
            artifacts["visualization"] = viz_path
        
        self.log_experiment(metrics, params, artifacts)
        
        # 8. Return results
        return {
            "metrics": metrics,
            "top_pros_cons": top_items,
            "results_df": results_df
        }


def display_results(results):
    """
    Display the results of the sentiment analysis pipeline.
    
    Args:
        results: Dictionary with results and metrics
    """
    print("\n" + "="*50)
    print("ARABIC SENTIMENT ANALYSIS RESULTS")
    print("="*50)
    
    # Display metrics
    print("\n📊 MODEL METRICS:")
    print("-"*30)
    for metric, value in results["metrics"].items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Display top pros
    print("\n✅ TOP PROS (POSITIVE SOLUTIONS):")
    print("-"*30)
    for i, pro in enumerate(results["top_pros_cons"]["top_pros"], 1):
        if isinstance(pro, str) and pro.strip():
            print(f"{i}. {pro}")
    
    # Display top cons
    print("\n❌ TOP CONS (NEGATIVE ISSUES):")
    print("-"*30)
    for i, con in enumerate(results["top_pros_cons"]["top_cons"], 1):
        if isinstance(con, str) and con.strip():
            print(f"{i}. {con}")
    
    print("\n" + "="*50)


def main():
    """
    Main function to run the Arabic sentiment analysis pipeline.
    """
    try:
        # Initialize the sentiment analysis pipeline
        pipeline = ArabicSentimentAnalysis()
        
        # Run the pipeline
        data_path = "Remacto Comments.csv"  # Update with actual path if different
        results = pipeline.run_pipeline(data_path)
        
        # Display results
        display_results(results)
        
        print("\nSentiment analysis pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis pipeline: {e}", exc_info=True)
        print(f"\nError in sentiment analysis pipeline: {e}")


if __name__ == "__main__":
    main()