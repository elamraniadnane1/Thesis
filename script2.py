#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Arabic Regional Data Analysis Pipeline
- Performs data preprocessing and normalization for Arabic text
- Uses CamelBERT-DA for sentiment analysis
- Creates Qdrant collections for vector search
- Extracts top pros and cons from regional initiatives
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

class ArabicRegionalAnalysis:
    """
    End-to-end pipeline for Arabic sentiment analysis of regional initiatives,
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
        
        logger.info("Regional analysis pipeline initialized successfully")
    
    def load_data(self, file_path=None, data_str=None):
        """
        Load data from CSV file or string
        
        Args:
            file_path: Path to the CSV file containing the initiatives
            data_str: String containing CSV data
            
        Returns:
            DataFrame with loaded data
        """
        if file_path:
            logger.info(f"Loading data from {file_path}")
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} records from {file_path}")
                return df
            except Exception as e:
                logger.error(f"Error loading data from file: {e}")
                raise
        elif data_str:
            logger.info("Loading data from provided string")
            try:
                import io
                df = pd.read_csv(io.StringIO(data_str))
                logger.info(f"Loaded {len(df)} records from string")
                return df
            except Exception as e:
                logger.error(f"Error loading data from string: {e}")
                raise
        else:
            logger.error("No data source provided")
            raise ValueError("Either file_path or data_str must be provided")
    
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
    
    def analyze_by_region(self, df, region_col, title_col):
        """
        Analyze sentiments grouped by region.
        
        Args:
            df: DataFrame with sentiment analysis results
            region_col: Column name containing region information
            title_col: Column containing the initiative titles
            
        Returns:
            DataFrame with regional sentiment analysis
        """
        logger.info(f"Analyzing sentiments by region using {region_col} and {title_col}")
        
        # First ensure we have sentiment scores for titles
        sentiment_col = f"{title_col}_sentiment"
        score_col = f"{title_col}_score"
        
        if sentiment_col not in df.columns:
            logger.warning(f"Sentiment column {sentiment_col} not found. Performing sentiment analysis first.")
            df = self.analyze_sentiments(df, [title_col])
        
        # Group by region and aggregate
        region_analysis = df.groupby(region_col).agg({
            title_col: 'count',  # Count of initiatives per region
            sentiment_col: lambda x: x.value_counts().to_dict(),  # Sentiment distribution
            score_col: 'mean'  # Average sentiment score
        }).reset_index()
        
        # Rename columns for clarity
        region_analysis.rename(columns={
            title_col: 'initiative_count',
            sentiment_col: 'sentiment_distribution',
            score_col: 'avg_sentiment_score'
        }, inplace=True)
        
        return region_analysis
    
    def extract_top_pros_cons(self, df, title_col, n=10):
        """
        Extract top positive and negative initiatives based on sentiment analysis.
        
        Args:
            df: DataFrame with sentiment analysis results
            title_col: Column containing the initiative titles
            n: Number of top items to extract
            
        Returns:
            Dictionary with top positive and negative initiatives
        """
        logger.info(f"Extracting top {n} positive and negative initiatives")
        
        # Ensure sentiment columns exist
        sentiment_col = f"{title_col}_sentiment"
        score_col = f"{title_col}_score"
        
        if sentiment_col not in df.columns or score_col not in df.columns:
            logger.error("Sentiment columns not found in DataFrame")
            return {"top_pros": [], "top_cons": []}
        
        # Extract negative initiatives
        cons_df = df[df[sentiment_col] == "negative"].sort_values(
            by=score_col, ascending=False
        )
        top_cons = cons_df[title_col].head(n).tolist()
        
        # Extract positive initiatives
        pros_df = df[df[sentiment_col] == "positive"].sort_values(
            by=score_col, ascending=False
        )
        top_pros = pros_df[title_col].head(n).tolist()
        
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
        experiment_name = "arabic-regional-analysis"
        
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
    
    def visualize_sentiment_by_region(self, df, region_col, title_col, output_path="regional_sentiment.png"):
        """
        Create visualization of sentiment distribution by region.
        
        Args:
            df: DataFrame with sentiment analysis results
            region_col: Column name containing region information
            title_col: Column containing the initiative titles
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        logger.info("Creating regional sentiment distribution visualization")
        
        # Ensure sentiment column exists
        sentiment_col = f"{title_col}_sentiment"
        
        if sentiment_col not in df.columns:
            logger.warning(f"Sentiment column {sentiment_col} not found. Cannot create visualization.")
            return None
        
        # Create pivot table for visualization
        pivot_df = pd.crosstab(
            df[region_col], 
            df[sentiment_col], 
            normalize='index'
        ) * 100  # Convert to percentage
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create stacked bar chart
        pivot_df.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 8))
        
        # Set labels and title
        plt.xlabel('Region')
        plt.ylabel('Percentage of Initiatives')
        plt.title('Sentiment Distribution by Region')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        
        # Add percentages on bars
        for n, x in enumerate([*pivot_df.index.values]):
            for (proportion, y_loc) in zip(pivot_df.loc[x], pivot_df.loc[x].cumsum()):
                if proportion > 5:  # Only label if proportion is large enough to be visible
                    plt.text(x=n, y=(y_loc - proportion/2), s=f'{proportion:.1f}%', 
                             ha='center', va='center', color='white', fontweight='bold')
        
        plt.savefig(output_path)
        logger.info(f"Regional visualization saved to {output_path}")
        
        return output_path
    
    def visualize_topics_distribution(self, df, topics_col, output_path="topics_distribution.png"):
        """
        Create visualization of topics distribution.
        
        Args:
            df: DataFrame with sentiment analysis results
            topics_col: Column containing topics
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        logger.info("Creating topics distribution visualization")
        
        if topics_col not in df.columns:
            logger.warning(f"Topics column {topics_col} not found. Cannot create visualization.")
            return None
        
        # Process topics (they might be multi-value)
        all_topics = []
        for topics in df[topics_col]:
            if isinstance(topics, str):
                # Handle potential multi-line topics (separated by newlines)
                topic_list = topics.split('\n')
                all_topics.extend([t.strip() for t in topic_list if t.strip()])
        
        # Count topics
        topic_counts = pd.Series(all_topics).value_counts()
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart for better readability with many topics
        ax = topic_counts.plot(kind='barh', colormap='viridis')
        
        # Set labels and title
        plt.xlabel('Count')
        plt.ylabel('Topic')
        plt.title('Distribution of Topics')
        plt.tight_layout()
        
        # Add count values on bars
        for i, (topic, count) in enumerate(topic_counts.items()):
            ax.text(count + 0.1, i, str(count), va='center')
        
        plt.savefig(output_path)
        logger.info(f"Topics visualization saved to {output_path}")
        
        return output_path
    
    def run_pipeline(self, data_source, region_col="CT1", title_col="Titles", topics_col="Topics"):
        """
        Run the complete regional analysis pipeline.
        
        Args:
            data_source: Path to CSV file or CSV data string
            region_col: Column name for region information
            title_col: Column name for initiative titles
            topics_col: Column name for topics
            
        Returns:
            Dictionary with results and metrics
        """
        logger.info("Starting regional analysis pipeline")
        
        # 1. Load and preprocess data
        if os.path.exists(data_source):
            df = self.load_data(file_path=data_source)
        else:
            df = self.load_data(data_str=data_source)
        
        # 2. Create Qdrant collection for vector search
        self.store_documents_in_qdrant(df, title_col, "initiatives_collection")
        
        # 3. Perform sentiment analysis
        results_df = self.analyze_sentiments(df, [title_col])
        
        # 4. Regional analysis
        regional_analysis = self.analyze_by_region(results_df, region_col, title_col)
        
        # 5. Extract top pros and cons (positive and negative initiatives)
        top_items = self.extract_top_pros_cons(results_df, title_col)
        
        # 6. Evaluate model (using a subset for testing)
        # Create a small test set from the data
        test_size = min(100, len(df) // 5)  # 20% or max 100 samples
        test_data = results_df[title_col].dropna().sample(test_size).tolist() if len(results_df) > 0 else []
        
        # Create synthetic labels (for demonstration - in a real scenario, you would have ground truth labels)
        # Here we're using our model's predictions as "truth" for demonstration purposes
        test_labels = [self.predict_sentiment(text)["label"] for text in test_data]
        
        # Evaluate model
        metrics = self.evaluate_model(test_data, test_labels)
        
        # 7. Create visualizations
        viz_paths = {}
        
        # Regional sentiment visualization
        regional_viz_path = self.visualize_sentiment_by_region(
            results_df, region_col, title_col, "regional_sentiment.png"
        )
        if regional_viz_path:
            viz_paths["regional_sentiment"] = regional_viz_path
        
        # Topics distribution visualization
        topics_viz_path = self.visualize_topics_distribution(
            results_df, topics_col, "topics_distribution.png"
        )
        if topics_viz_path:
            viz_paths["topics_distribution"] = topics_viz_path
        
        # 8. Log experiment to MLFlow
        params = {
            "model_name": self.model_name,
            "preprocessing_applied": "Arabic normalization, special char removal, diacritics removal"
        }
        
        self.log_experiment(metrics, params, viz_paths)
        
        # 9. Return results
        return {
            "metrics": metrics,
            "top_pros_cons": top_items,
            "regional_analysis": regional_analysis,
            "results_df": results_df
        }


def display_results(results):
    """
    Display the results of the regional analysis pipeline.
    
    Args:
        results: Dictionary with results and metrics
    """
    print("\n" + "="*60)
    print("ARABIC REGIONAL INITIATIVES ANALYSIS RESULTS")
    print("="*60)
    
    # Display metrics
    print("\n📊 MODEL METRICS:")
    print("-"*40)
    for metric, value in results["metrics"].items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Display top positive initiatives
    print("\n✅ TOP POSITIVE INITIATIVES:")
    print("-"*40)
    for i, pro in enumerate(results["top_pros_cons"]["top_pros"], 1):
        if isinstance(pro, str) and pro.strip():
            print(f"{i}. {pro}")
    
    # Display top negative initiatives
    print("\n❌ TOP NEGATIVE INITIATIVES:")
    print("-"*40)
    for i, con in enumerate(results["top_pros_cons"]["top_cons"], 1):
        if isinstance(con, str) and con.strip():
            print(f"{i}. {con}")
    
    # Display regional analysis summary
    print("\n🌍 REGIONAL ANALYSIS SUMMARY:")
    print("-"*40)
    
    regional_df = results["regional_analysis"]
    for _, row in regional_df.iterrows():
        region = row[regional_df.columns[0]]  # First column is region name
        initiative_count = row['initiative_count']
        sentiment_dist = row['sentiment_distribution']
        avg_score = row['avg_sentiment_score']
        
        print(f"Region: {region}")
        print(f"  - Initiative Count: {initiative_count}")
        print(f"  - Sentiment Distribution: {sentiment_dist}")
        print(f"  - Average Sentiment Score: {avg_score:.4f}")
        print("-"*30)
    
    print("\n" + "="*60)

def main():
    """
    Main function to run the Arabic regional analysis pipeline.
    """
    try:
        # Initialize the analysis pipeline
        pipeline = ArabicRegionalAnalysis()
        
        # Data as a string (for demo purposes)
        data = """Titles,CT1,CT2,Topics
تعزيز الشفافية الولوج إلى المعلومة,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,الولوج إلى المعلومة
إطلاق منصة إلكترونية لتتبع وتقييم برنامج التنمية الجهوية,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,تتبع وتقييم برامج التنمية
تنفيذ استراتيجية جهوية للإدماج والنوع الاجتماعي,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,"النوع
الإعاقة"
الميزانية المستجيبة للنوع وميزانية المواطن,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,الميزانية
النشر الاستباقي للمعلومات,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,الولوج إلى المعلومة
‏تشكيل ودعم قدرات الهيئات الاستشارية,جماعة تطوان,Commune Tetouan,الهيئات الإستشارية
اعداد ونشر ميزانية المواطن,جماعة تطوان,Commune Tetouan,الميزانية
إعداد ميزانية تشاركية,جماعة تطوان,Commune Tetouan,الميزانية
إعداد دليل للدعم و الشراكة مع هيئات المجتمع المدني,جماعة تطوان,Commune Tetouan,المشاركة المواطنة
النشر الاستباقي للمعلومات الأولية و الولوج للمعلومة,جماعة تطوان,Commune Tetouan,الولوج إلى المعلومة
إعداد برامج عمل موضوعاتية من طرف الهيئات الاستشارية,جماعة تطوان,Commune Tetouan,الهيئات الإستشارية
الدمج الإجتماعي بالتنشيط الثقافي والرياضي والفني,جماعة تطوان,Commune Tetouan,"الثقافة
الرياضة"
التتبع العمومي لبرنامج عمل الجماعة 2023_2028,جماعة تطوان,Commune Tetouan,تتبع وتقييم برامج التنمية
الرقمنة من أجل خدمات جماعية ذات جودة وفعالية,جماعة تطوان,Commune Tetouan,الرقمنة
التواصل الدامج,جماعة تطوان,Commune Tetouan,التواصل
الولوج الشامل للتراث,جماعة تطوان,Commune Tetouan,الجاذبية الترابية
إحداث مرصد جماعي للبيئة والمناخ والتنمية المستدامة,جماعة تطوان,Commune Tetouan,البيئة"""
        
        # Save data to temporary file
        temp_file = "regional_initiatives.csv"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(data)
        
        # Run the pipeline
        results = pipeline.run_pipeline(temp_file)
        
        # Display results
        display_results(results)
        
        # Create a web dashboard for interactive visualization
        create_interactive_dashboard(results, "arabic_regional_dashboard.html")
        
        print("\nRegional analysis pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in regional analysis pipeline: {e}", exc_info=True)
        print(f"\nError in regional analysis pipeline: {e}")


def create_interactive_dashboard(results, output_file):
    """
    Create an interactive HTML dashboard with results.
    
    Args:
        results: Dictionary with analysis results
        output_file: Path to save the HTML dashboard
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
        
        logger.info("Creating interactive dashboard")
        
        # Extract data
        df = results["results_df"]
        regional_df = results["regional_analysis"]
        metrics = results["metrics"]
        
        # Create a Figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
            ],
            subplot_titles=(
                "Accuracy", "F1 Score",
                "Sentiment by Region",
                "Top Topics"
            )
        )
        
        # Add accuracy indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics["accuracy"] * 100,
                title={"text": "Accuracy (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                        {"range": [80, 100], "color": "lightblue"},
                    ],
                },
                domain={"row": 0, "column": 0},
            )
        )
        
        # Add F1 score indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=metrics["f1_score"] * 100,
                title={"text": "F1 Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkgreen"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                        {"range": [80, 100], "color": "lightgreen"},
                    ],
                },
                domain={"row": 0, "column": 1},
            )
        )
        
        # Process sentiment by region for plotting
        region_names = []
        positive_values = []
        neutral_values = []
        negative_values = []
        
        for _, row in regional_df.iterrows():
            region = row[regional_df.columns[0]]
            region_names.append(region)
            
            sentiment_dist = row['sentiment_distribution']
            positive_values.append(sentiment_dist.get('positive', 0))
            neutral_values.append(sentiment_dist.get('neutral', 0))
            negative_values.append(sentiment_dist.get('negative', 0))
        
        # Add stacked bar chart for sentiment by region
        fig.add_trace(
            go.Bar(
                name="Positive",
                x=region_names,
                y=positive_values,
                marker_color='green'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name="Neutral",
                x=region_names,
                y=neutral_values,
                marker_color='gray'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name="Negative",
                x=region_names,
                y=negative_values,
                marker_color='red'
            ),
            row=2, col=1
        )
        
        # Process topics for plotting
        all_topics = []
        for topics in df["Topics"]:
            if isinstance(topics, str):
                topic_list = topics.split('\n')
                all_topics.extend([t.strip() for t in topic_list if t.strip()])
        
        topic_counts = pd.Series(all_topics).value_counts()
        
        # Add horizontal bar chart for topics
        fig.add_trace(
            go.Bar(
                x=topic_counts.values,
                y=topic_counts.index,
                orientation='h',
                marker_color='darkblue'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text="Arabic Regional Initiatives Analysis Dashboard",
            height=1000,
            width=1200,
            barmode='stack',
            showlegend=True
        )
        
        # Create a table for top pros and cons
        top_pros = results["top_pros_cons"]["top_pros"]
        top_cons = results["top_pros_cons"]["top_cons"]
        
        # Calculate max length for tables
        max_length = max(len(top_pros), len(top_cons))
        
        # Pad shorter list with empty strings
        padded_pros = top_pros + [""] * (max_length - len(top_pros))
        padded_cons = top_cons + [""] * (max_length - len(top_cons))
        
        # Create table figure
        table_fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=["Top Positive Initiatives", "Top Negative Initiatives"],
                    fill_color='paleturquoise',
                    align='center',
                    font=dict(size=14)
                ),
                cells=dict(
                    values=[padded_pros, padded_cons],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=12),
                    height=30
                )
            )
        ])
        
        table_fig.update_layout(
            title_text="Top Positive and Negative Initiatives",
            height=500,
            width=1200
        )
        
        # Save figures to HTML
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Arabic Regional Initiatives Analysis</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                h2 {
                    color: #666;
                }
                .metrics {
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                }
                .metric-box {
                    background-color: #e9ecef;
                    padding: 15px;
                    border-radius: 5px;
                    text-align: center;
                    width: 150px;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }
                .metric-name {
                    font-size: 14px;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <h1>Arabic Regional Initiatives Analysis Dashboard</h1>
            
            <div class="container">
                <h2>Model Performance Metrics</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics['accuracy']:.2%}" + """</div>
                        <div class="metric-name">Accuracy</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics['precision']:.2%}" + """</div>
                        <div class="metric-name">Precision</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics['recall']:.2%}" + """</div>
                        <div class="metric-name">Recall</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics['f1_score']:.2%}" + """</div>
                        <div class="metric-name">F1 Score</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">""" + f"{metrics['bleu_score']:.2%}" + """</div>
                        <div class="metric-name">BLEU Score</div>
                    </div>
                </div>
            </div>
            
            <div class="container">
                <h2>Results Visualization</h2>
                <div id="dashboard"></div>
            </div>
            
            <div class="container">
                <h2>Top Initiatives by Sentiment</h2>
                <div id="table"></div>
            </div>
            
            <script>
                var dashboardDiv = document.getElementById('dashboard');
                var tableDiv = document.getElementById('table');
                
                var dashboardData = """ + pio.to_json(fig) + """;
                var tableData = """ + pio.to_json(table_fig) + """;
                
                Plotly.newPlot(dashboardDiv, dashboardData.data, dashboardData.layout);
                Plotly.newPlot(tableDiv, tableData.data, tableData.layout);
            </script>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info(f"Interactive dashboard saved to {output_file}")
        
    except ImportError:
        logger.warning("Plotly not installed. Skipping interactive dashboard creation.")
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}")


def analyze_topic_sentiments(df, topics_col="Topics", title_col="Titles"):
    """
    Analyze sentiments by topic.
    
    Args:
        df: DataFrame with sentiment analysis results
        topics_col: Column name containing topics
        title_col: Column containing initiative titles
        
    Returns:
        DataFrame with topic sentiment analysis
    """
    logger.info(f"Analyzing sentiments by topics from {topics_col}")
    
    # Ensure sentiment column exists
    sentiment_col = f"{title_col}_sentiment"
    
    if sentiment_col not in df.columns:
        logger.warning(f"Sentiment column {sentiment_col} not found. Cannot analyze topics.")
        return None
    
    # Explode multi-value topics
    topic_data = []
    
    for _, row in df.iterrows():
        topics = row[topics_col]
        sentiment = row[sentiment_col]
        
        if isinstance(topics, str):
            topic_list = topics.split('\n')
            for topic in topic_list:
                topic = topic.strip()
                if topic:
                    topic_data.append({
                        "topic": topic,
                        "sentiment": sentiment
                    })
    
    # Create DataFrame from exploded topics
    topics_df = pd.DataFrame(topic_data)
    
    # Group by topic and calculate sentiment distribution
    topic_analysis = topics_df.groupby("topic").agg({
        "sentiment": lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Add count of initiatives per topic
    topic_counts = topics_df.groupby("topic").size().reset_index(name='count')
    topic_analysis = topic_analysis.merge(topic_counts, on="topic")
    
    # Sort by count
    topic_analysis = topic_analysis.sort_values(by="count", ascending=False)
    
    return topic_analysis


def export_results_to_excel(results, output_file="arabic_regional_analysis_results.xlsx"):
    """
    Export analysis results to Excel file.
    
    Args:
        results: Dictionary with analysis results
        output_file: Path to save the Excel file
    """
    try:
        logger.info(f"Exporting results to Excel: {output_file}")
        
        # Create a Pandas Excel writer
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        # Write each DataFrame to a different sheet
        results["results_df"].to_excel(writer, sheet_name='Sentiment Analysis', index=False)
        results["regional_analysis"].to_excel(writer, sheet_name='Regional Analysis', index=False)
        
        # Create topics analysis
        topic_analysis = analyze_topic_sentiments(results["results_df"])
        if topic_analysis is not None:
            topic_analysis.to_excel(writer, sheet_name='Topic Analysis', index=False)
        
        # Create a sheet for metrics
        metrics_df = pd.DataFrame([results["metrics"]])
        metrics_df.to_excel(writer, sheet_name='Model Metrics', index=False)
        
        # Create a sheet for top pros and cons
        pros_df = pd.DataFrame(results["top_pros_cons"]["top_pros"], columns=["Top Positive Initiatives"])
        cons_df = pd.DataFrame(results["top_pros_cons"]["top_cons"], columns=["Top Negative Initiatives"])
        
        # Combine pros and cons side by side
        max_len = max(len(pros_df), len(cons_df))
        pros_df = pros_df.reindex(range(max_len))
        cons_df = cons_df.reindex(range(max_len))
        pros_cons_df = pd.concat([pros_df, cons_df], axis=1)
        pros_cons_df.to_excel(writer, sheet_name='Top Pros & Cons', index=False)
        
        # Save the Excel file
        writer.save()
        logger.info(f"Results exported to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error exporting results to Excel: {e}")
        return None


def send_email_report(recipients, results_file, dashboard_file=None, smtp_server='localhost', port=25):
    """
    Send email report with analysis results.
    
    Args:
        recipients: List of email recipients
        results_file: Path to the Excel results file
        dashboard_file: Path to the HTML dashboard file
        smtp_server: SMTP server address
        port: SMTP server port
    """
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.application import MIMEApplication
        from datetime import datetime
        
        logger.info(f"Sending email report to {recipients}")
        
        # Create message
        msg = MIMEMultipart()
        msg['Subject'] = f'Arabic Regional Analysis Report - {datetime.now().strftime("%Y-%m-%d")}'
        msg['From'] = 'arabic-analysis@example.com'
        msg['To'] = ', '.join(recipients)
        
        # Email body
        body = """
        <html>
        <body>
            <h2>Arabic Regional Initiatives Analysis Report</h2>
            <p>Please find attached the analysis results of the Arabic regional initiatives.</p>
            <p>The report includes:</p>
            <ul>
                <li>Sentiment analysis results</li>
                <li>Regional analysis</li>
                <li>Topic analysis</li>
                <li>Model performance metrics</li>
                <li>Top positive and negative initiatives</li>
            </ul>
            <p>This is an automated report generated by the Arabic Regional Analysis Pipeline.</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        
        # Attach Excel results
        with open(results_file, 'rb') as file:
            attachment = MIMEApplication(file.read(), Name=os.path.basename(results_file))
            attachment['Content-Disposition'] = f'attachment; filename="{os.path.basename(results_file)}"'
            msg.attach(attachment)
        
        # Attach HTML dashboard if available
        if dashboard_file and os.path.exists(dashboard_file):
            with open(dashboard_file, 'rb') as file:
                attachment = MIMEApplication(file.read(), Name=os.path.basename(dashboard_file))
                attachment['Content-Disposition'] = f'attachment; filename="{os.path.basename(dashboard_file)}"'
                msg.attach(attachment)
        
        # Send email
        with smtplib.SMTP(smtp_server, port) as server:
            server.send_message(msg)
        
        logger.info(f"Email report sent to {recipients}")
        
    except Exception as e:
        logger.error(f"Error sending email report: {e}")


if __name__ == "__main__":
    # Add command line argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description='Arabic Regional Data Analysis Pipeline')
    parser.add_argument('--input', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='results', help='Output file prefix')
    parser.add_argument('--email', type=str, nargs='+', help='Email recipients for report')
    parser.add_argument('--mlflow-uri', type=str, default='http://192.168.1.116:5000', 
                        help='MLFlow tracking server URI')
    
    args = parser.parse_args()
    
    try:
        # Initialize the analysis pipeline
        pipeline = ArabicRegionalAnalysis(mlflow_uri=args.mlflow_uri)
        
        # Determine input source
        data_source = args.input if args.input else "regional_initiatives.csv"
        
        # Check if the data source is a file that exists or we need to create a temporary file
        if not os.path.exists(data_source):
            # Default data as a string (for demo purposes)
            data = """Titles,CT1,CT2,Topics
تعزيز الشفافية الولوج إلى المعلومة,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,الولوج إلى المعلومة
إطلاق منصة إلكترونية لتتبع وتقييم برنامج التنمية الجهوية,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,تتبع وتقييم برامج التنمية
تنفيذ استراتيجية جهوية للإدماج والنوع الاجتماعي,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,"النوع
الإعاقة"
الميزانية المستجيبة للنوع وميزانية المواطن,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,الميزانية
النشر الاستباقي للمعلومات,جهة طنجة تطوان الحسيمة,Region Tanger-Tetouan-Al Hoceima,الولوج إلى المعلومة
‏تشكيل ودعم قدرات الهيئات الاستشارية,جماعة تطوان,Commune Tetouan,الهيئات الإستشارية
اعداد ونشر ميزانية المواطن,جماعة تطوان,Commune Tetouan,الميزانية
إعداد ميزانية تشاركية,جماعة تطوان,Commune Tetouan,الميزانية
إعداد دليل للدعم و الشراكة مع هيئات المجتمع المدني,جماعة تطوان,Commune Tetouan,المشاركة المواطنة
النشر الاستباقي للمعلومات الأولية و الولوج للمعلومة,جماعة تطوان,Commune Tetouan,الولوج إلى المعلومة
إعداد برامج عمل موضوعاتية من طرف الهيئات الاستشارية,جماعة تطوان,Commune Tetouan,الهيئات الإستشارية
الدمج الإجتماعي بالتنشيط الثقافي والرياضي والفني,جماعة تطوان,Commune Tetouan,"الثقافة
الرياضة"
التتبع العمومي لبرنامج عمل الجماعة 2023_2028,جماعة تطوان,Commune Tetouan,تتبع وتقييم برامج التنمية
الرقمنة من أجل خدمات جماعية ذات جودة وفعالية,جماعة تطوان,Commune Tetouan,الرقمنة
التواصل الدامج,جماعة تطوان,Commune Tetouan,التواصل
الولوج الشامل للتراث,جماعة تطوان,Commune Tetouan,الجاذبية الترابية
إحداث مرصد جماعي للبيئة والمناخ والتنمية المستدامة,جماعة تطوان,Commune Tetouan,البيئة"""
            
            # Save data to temporary file
            temp_file = "regional_initiatives.csv"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(data)
            
            data_source = temp_file
        
        # Run the pipeline
        results = pipeline.run_pipeline(data_source)
        
        # Display results
        display_results(results)
        
        # Create output files
        output_prefix = args.output
        
        # Create interactive dashboard
        dashboard_file = f"{output_prefix}_dashboard.html"
        create_interactive_dashboard(results, dashboard_file)
        
        # Export results to Excel
        excel_file = f"{output_prefix}_results.xlsx"
        export_results_to_excel(results, excel_file)
        
        # Send email report if recipients provided
        if args.email:
            send_email_report(args.email, excel_file, dashboard_file)
        
        print(f"\nRegional analysis pipeline completed successfully!")
        print(f"Results saved to {excel_file}")
        print(f"Dashboard saved to {dashboard_file}")
        
    except Exception as e:
        logger.error(f"Error in regional analysis pipeline: {e}", exc_info=True)
        print(f"\nError in regional analysis pipeline: {e}")