import os
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from qdrant_client import QdrantClient
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import TextBlob for simple sentiment analysis
from textblob import TextBlob

# Set to True to enable detailed logging
DEBUG = True

def log(message: str) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")

class SimpleSentimentAnalyzer:
    """Class to analyze sentiment in the dataset using simple TextBlob approach."""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "sentiment_analysis_dataset",
        batch_size: int = 100
    ):
        """Initialize the sentiment analyzer with connection parameters."""
        # Setup Qdrant client
        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key
        )
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # Define sentiment thresholds for TextBlob
        self.sentiment_thresholds = {
            "positive": 0.1,   # polarity > 0.1 is positive
            "negative": -0.1,  # polarity < -0.1 is negative
            # between -0.1 and 0.1 is neutral
        }
        
        # Dataset statistics
        self.stats = {
            "total_records": 0,
            "processed_records": 0,
            "records_with_ground_truth": 0,
            "ground_truth_distribution": {},
            "prediction_distribution": {}
        }
    
    def load_data_from_qdrant(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load sentiment dataset from Qdrant collection."""
        log(f"Loading data from Qdrant collection: {self.collection_name}")
        
        # Get total count
        try:
            total_count = self.client.count(collection_name=self.collection_name).count
            log(f"Total records in collection: {total_count}")
            self.stats["total_records"] = total_count
        except Exception as e:
            log(f"Error getting collection count: {e}")
            total_count = "unknown"
        
        # Retrieve data with pagination
        offset = None
        all_data = []
        record_count = 0
        
        while True:
            try:
                # Get batch of records
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=min(500, limit - record_count if limit else 500),
                    offset=offset
                )
                
                # Handle both tuple return (newer versions) and object return (older versions)
                if isinstance(result, tuple):
                    points, next_page_offset = result
                else:
                    points = result.points
                    next_page_offset = result.next_page_offset
                
                if not points:
                    break
                
                # Process points
                for point in points:
                    record = {
                        "id": point.id,
                        "text": point.payload.get("text", "")
                    }
                    
                    # Extract ground truth sentiment if available
                    # Check different possible locations for sentiment labels
                    if "sentiment_label" in point.payload:
                        record["ground_truth"] = point.payload["sentiment_label"]
                    elif "sentiment" in point.payload:
                        record["ground_truth"] = point.payload["sentiment"]
                    elif "metadata" in point.payload and "sentiment" in point.payload["metadata"]:
                        record["ground_truth"] = point.payload["metadata"]["sentiment"]
                    elif "raw_payload" in point.payload and "sentiment" in point.payload["raw_payload"]:
                        record["ground_truth"] = point.payload["raw_payload"]["sentiment"]
                    else:
                        record["ground_truth"] = None
                    
                    # Extract metadata
                    if "metadata" in point.payload:
                        record["source"] = point.payload["metadata"].get("source_collection", "unknown")
                    else:
                        record["source"] = "unknown"
                    
                    all_data.append(record)
                
                record_count += len(points)
                log(f"Loaded {record_count} records so far")
                
                # Update offset for next batch
                offset = next_page_offset
                if offset is None or (limit and record_count >= limit):
                    break
                    
            except Exception as e:
                log(f"Error loading data from Qdrant: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Count records with ground truth
        self.stats["processed_records"] = len(df)
        self.stats["records_with_ground_truth"] = df["ground_truth"].notna().sum()
        
        # Get ground truth distribution
        if self.stats["records_with_ground_truth"] > 0:
            self.stats["ground_truth_distribution"] = df["ground_truth"].value_counts().to_dict()
        
        log(f"Loaded {len(df)} records, {self.stats['records_with_ground_truth']} with ground truth")
        return df
    
    def load_data_from_mock(self, count: int = 1000) -> pd.DataFrame:
        """Generate mock data for testing when Qdrant is not available."""
        log(f"Generating {count} mock records for testing")
        
        # Sample texts with different sentiments
        positive_texts = [
            "I'm very impressed with the new park development! The city did a great job.",
            "This municipal project has exceeded my expectations. The roads are much better now.",
            "The new community center is fantastic and exactly what we needed in this neighborhood.",
            "I fully support this initiative and believe it will greatly benefit our community.",
            "The efficiency of the municipal services has improved dramatically in recent months."
        ]
        
        negative_texts = [
            "I'm disappointed with the quality of work on the new infrastructure project.",
            "This is a waste of taxpayer money. The project has too many flaws.",
            "The construction is taking far too long and causing significant disruption.",
            "The municipality has ignored community feedback on this important issue.",
            "This project fails to address the real needs of our neighborhood."
        ]
        
        neutral_texts = [
            "The project seems to be progressing according to the timeline provided.",
            "I've noticed the construction work happening on Main Street.",
            "I would like more information about how this project will affect traffic.",
            "The municipality announced that the project will continue until December.",
            "According to the report, this initiative is part of the five-year development plan."
        ]
        
        all_texts = positive_texts + negative_texts + neutral_texts
        all_sentiments = ["positive"] * len(positive_texts) + ["negative"] * len(negative_texts) + ["neutral"] * len(neutral_texts)
        
        data = []
        for i in range(count):
            idx = i % len(all_texts)
            # Add some variation to avoid exact duplicates
            text = f"{all_texts[idx]} [{i}]"
            
            data.append({
                "id": f"mock_{i}",
                "text": text,
                "ground_truth": all_sentiments[idx],
                "source": "mock_data"
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Update stats
        self.stats["processed_records"] = len(df)
        self.stats["records_with_ground_truth"] = len(df)
        self.stats["ground_truth_distribution"] = df["ground_truth"].value_counts().to_dict()
        
        log(f"Generated {len(df)} mock records")
        return df
    
    def analyze_sentiment_with_textblob(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment using TextBlob and add predictions to the DataFrame."""
        log("Analyzing sentiment with TextBlob")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize columns for predictions
        result_df["polarity"] = None
        result_df["subjectivity"] = None
        result_df["predicted_label"] = None
        
        # Process in batches
        texts = result_df["text"].tolist()
        
        for i, text in enumerate(tqdm(texts, desc="Analyzing sentiment")):
            try:
                # Create TextBlob object
                blob = TextBlob(text)
                
                # Get sentiment properties
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Determine sentiment label based on polarity
                if polarity > self.sentiment_thresholds["positive"]:
                    sentiment = "positive"
                elif polarity < self.sentiment_thresholds["negative"]:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                # Store results
                result_df.loc[i, "polarity"] = polarity
                result_df.loc[i, "subjectivity"] = subjectivity
                result_df.loc[i, "predicted_label"] = sentiment
                
            except Exception as e:
                log(f"Error processing text: {e}")
                # Set defaults for error cases
                result_df.loc[i, "polarity"] = 0
                result_df.loc[i, "subjectivity"] = 0
                result_df.loc[i, "predicted_label"] = "neutral"
        
        # Get prediction distribution
        self.stats["prediction_distribution"] = result_df["predicted_label"].value_counts().to_dict()
        
        return result_df
    
    def normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize ground truth and predicted labels for consistent comparison."""
        result_df = df.copy()
        
        # Define mappings for standardization
        sentiment_mapping = {
            # Standardize positive labels
            "positive": "positive",
            "pos": "positive",
            "p": "positive",
            "1": "positive",
            
            # Standardize negative labels
            "negative": "negative",
            "neg": "negative",
            "n": "negative",
            "-1": "negative",
            
            # Standardize neutral labels
            "neutral": "neutral",
            "neu": "neutral",
            "0": "neutral"
        }
        
        # Apply normalization to ground truth if it exists
        if "ground_truth" in result_df.columns:
            result_df["ground_truth_normalized"] = result_df["ground_truth"].astype(str).str.lower()
            result_df["ground_truth_normalized"] = result_df["ground_truth_normalized"].map(
                sentiment_mapping).fillna(result_df["ground_truth"])
        
        # Apply normalization to predicted labels
        if "predicted_label" in result_df.columns:
            result_df["predicted_label_normalized"] = result_df["predicted_label"].astype(str).str.lower()
            result_df["predicted_label_normalized"] = result_df["predicted_label_normalized"].map(
                sentiment_mapping).fillna(result_df["predicted_label"])
        
        return result_df
    
    def evaluate_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance by comparing predictions with ground truth."""
        log("Evaluating model performance")
        
        # Filter records with both prediction and ground truth
        eval_df = df[df["ground_truth_normalized"].notna() & df["predicted_label_normalized"].notna()].copy()
        
        if len(eval_df) == 0:
            log("No records with both prediction and ground truth available for evaluation")
            return {"error": "No records available for evaluation"}
        
        # Compute metrics
        y_true = eval_df["ground_truth_normalized"]
        y_pred = eval_df["predicted_label_normalized"]
        
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        
        # Create results dictionary
        evaluation = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "classes": sorted(set(y_true.unique()) | set(y_pred.unique())),
            "evaluated_records": len(eval_df),
            "total_records": len(df)
        }
        
        log(f"Evaluation complete - Accuracy: {acc:.4f}")
        return evaluation
    
    def save_results(self, df: pd.DataFrame, evaluation: Dict[str, Any], output_dir: str = "./results") -> None:
        """Save analysis results, including predictions and evaluation metrics."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DataFrame with predictions
        csv_path = os.path.join(output_dir, f"sentiment_predictions_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Define a custom JSON encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # Save evaluation metrics
        metrics_path = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "evaluation": evaluation,
                "stats": self.stats,
                "model_info": {
                    "name": "TextBlob",
                    "sentiment_thresholds": self.sentiment_thresholds
                }
            }, f, indent=2, cls=NumpyEncoder)
        
        # Create and save confusion matrix visualization
        if "confusion_matrix" in evaluation:
            self._save_confusion_matrix(evaluation, output_dir, timestamp)
        
        log(f"Results saved to {output_dir}")
        print(f"Predictions saved to: {csv_path}")
        print(f"Evaluation metrics saved to: {metrics_path}")
    
    def _save_confusion_matrix(self, evaluation: Dict[str, Any], output_dir: str, timestamp: str) -> None:
        """Create and save confusion matrix visualization."""
        conf_matrix = np.array(evaluation["confusion_matrix"])
        classes = evaluation["classes"]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Sentiment Analysis Confusion Matrix")
        
        cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
    
    def print_evaluation_summary(self, evaluation: Dict[str, Any]) -> None:
        """Print a summary of the evaluation metrics."""
        if "error" in evaluation:
            print(f"\nEvaluation Error: {evaluation['error']}")
            return
            
        print("\n" + "="*80)
        print("SENTIMENT ANALYSIS EVALUATION SUMMARY")
        print("="*80)
        
        print(f"Model: TextBlob")
        print(f"Total records: {evaluation['total_records']}")
        print(f"Records evaluated: {evaluation['evaluated_records']}")
        print(f"Accuracy: {evaluation['accuracy']:.4f}")
        
        # Print classification report in a readable format
        report = evaluation["classification_report"]
        print("\nClassification Report:")
        print("-" * 60)
        print(f"{'Class':15} {'Precision':10} {'Recall':10} {'F1-score':10} {'Support':10}")
        print("-" * 60)
        
        for cls in sorted(c for c in report.keys() if c not in ["accuracy", "macro avg", "weighted avg"]):
            print(f"{cls:15} {report[cls]['precision']:.4f}{'':6} {report[cls]['recall']:.4f}{'':6} "
                  f"{report[cls]['f1-score']:.4f}{'':6} {report[cls]['support']:10}")
        
        print("-" * 60)
        print(f"{'weighted avg':15} {report['weighted avg']['precision']:.4f}{'':6} "
              f"{report['weighted avg']['recall']:.4f}{'':6} "
              f"{report['weighted avg']['f1-score']:.4f}{'':6} {report['weighted avg']['support']:10}")
        print("="*80)

    def run_sentiment_analysis(
        self, 
        limit: Optional[int] = None,
        output_dir: str = "./results",
        use_mock_data: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the complete sentiment analysis workflow."""
        # Step 1: Load data from Qdrant or generate mock data
        if use_mock_data:
            df = self.load_data_from_mock(count=1000)
        else:
            df = self.load_data_from_qdrant(limit)
        
        if len(df) == 0:
            log("No data found")
            return df, {"error": "No data found"}
        
        # Step 2: Run sentiment analysis
        result_df = self.analyze_sentiment_with_textblob(df)
        
        # Step 3: Normalize labels for comparison
        result_df = self.normalize_labels(result_df)
        
        # Step 4: Evaluate model performance
        evaluation = self.evaluate_model(result_df)
        
        # Step 5: Print evaluation summary
        self.print_evaluation_summary(evaluation)
        
        # Step 6: Save results
        self.save_results(result_df, evaluation, output_dir)
        
        return result_df, evaluation


def add_data_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    count: int = 1000
) -> None:
    """Add mock sentiment data to Qdrant for testing."""
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    
    # Check if collection exists
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    # Create collection if it doesn't exist
    if collection_name not in collection_names:
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "embedding": VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            }
        )
    
    # Sample texts with different sentiments
    positive_texts = [
        "I'm very impressed with the new park development! The city did a great job.",
        "This municipal project has exceeded my expectations. The roads are much better now.",
        "The new community center is fantastic and exactly what we needed in this neighborhood.",
        "I fully support this initiative and believe it will greatly benefit our community.",
        "The efficiency of the municipal services has improved dramatically in recent months."
    ]
    
    negative_texts = [
        "I'm disappointed with the quality of work on the new infrastructure project.",
        "This is a waste of taxpayer money. The project has too many flaws.",
        "The construction is taking far too long and causing significant disruption.",
        "The municipality has ignored community feedback on this important issue.",
        "This project fails to address the real needs of our neighborhood."
    ]
    
    neutral_texts = [
        "The project seems to be progressing according to the timeline provided.",
        "I've noticed the construction work happening on Main Street.",
        "I would like more information about how this project will affect traffic.",
        "The municipality announced that the project will continue until December.",
        "According to the report, this initiative is part of the five-year development plan."
    ]
    
    all_texts = positive_texts + negative_texts + neutral_texts
    all_sentiments = ["positive"] * len(positive_texts) + ["negative"] * len(negative_texts) + ["neutral"] * len(neutral_texts)
    
    # Add points in batches
    batch_size = 100
    for i in range(0, count, batch_size):
        batch_count = min(batch_size, count - i)
        batch_points = []
        
        for j in range(batch_count):
            idx = (i + j) % len(all_texts)
            text = f"{all_texts[idx]} [{i + j}]"
            sentiment = all_sentiments[idx]
            
            # Create point
            batch_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        "text": text,
                        "sentiment_label": sentiment,
                        "metadata": {
                            "source_collection": "mock_data",
                            "created_at": datetime.datetime.now().isoformat()
                        }
                    },
                    vector={
                        "embedding": list(np.random.uniform(-1, 1, 384))
                    }
                )
            )
        
        # Insert batch
        client.upsert(
            collection_name=collection_name,
            points=batch_points
        )
        
        print(f"Added {len(batch_points)} records to Qdrant (total: {i + len(batch_points)})")


def main():
    """Main function to run sentiment analysis on the Qdrant dataset."""
    print("\n" + "="*80)
    print("SIMPLE SENTIMENT ANALYSIS WITH TEXTBLOB")
    print("="*80)
    
    # Define Qdrant connection parameters
    qdrant_host = "localhost"
    qdrant_port = 6333
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    collection_name = "sentiment_analysis_dataset"
    
    # Check if we should populate Qdrant with test data
    populate_qdrant = input("Would you like to populate Qdrant with test data? (y/n): ").lower() == 'y'
    
    if populate_qdrant:
        client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key
        )
        add_data_to_qdrant(client, collection_name, count=1000)
    
    # Create sentiment analyzer
    analyzer = SimpleSentimentAnalyzer(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        batch_size=100
    )
    
    # Check if we should use mock data instead of Qdrant
    use_mock = input("Would you like to use mock data instead of Qdrant? (y/n): ").lower() == 'y'
    
    # Run sentiment analysis
    _, _ = analyzer.run_sentiment_analysis(
        limit=None,  # Set to a number to limit records processed
        output_dir="./sentiment_results",
        use_mock_data=use_mock
    )
    
    print("\nSentiment analysis complete! Results saved to ./sentiment_results")


if __name__ == "__main__":
    main()