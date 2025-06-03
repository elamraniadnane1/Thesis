import os
import json
import argparse
import gc
import time
from typing import Dict, List, Any, Optional, Union
import traceback

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Set to True to enable detailed logging
DEBUG = True

def log(message: str) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")

class MemoryEfficientSentimentAnalyzer:
    """Memory-efficient sentiment analysis for Arabic text."""
    
    def __init__(
        self,
        model_name: str = "Ammar-alhaj-ali/arabic-MARBERT-sentiment",
        device: str = None,
        batch_size: int = 16,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "sentiment_analysis_dataset"
    ):
        """Initialize the analyzer with model and database connection."""
        # Set up device - use GPU if available, otherwise CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Device set to use {device}")
        
        # Set batch size based on device
        self.batch_size = batch_size
        if device == "cuda":
            # Larger batch size for GPU
            self.batch_size = batch_size * 2
        
        # Initialize model and tokenizer
        log(f"Loading sentiment model: {model_name}")
        self.model_name = model_name
        
        # Memory-efficient loading
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optimizations
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Create pipeline with optimized batch size
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model, 
            tokenizer=self.tokenizer,
            device=device,
            batch_size=self.batch_size,
            top_k=None  # Get scores for all labels (replaces return_all_scores=True)
        )
        
        # Map id to label
        self.id2label = model.config.id2label
        log(f"Model labels: {self.id2label}")
        
        # Create a reverse mapping from label name to id
        self.label2id = {v: k for k, v in self.id2label.items()}
        log(f"Label to ID mapping: {self.label2id}")
        
        # Connect to Qdrant
        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key
        )
        self.collection_name = collection_name
    
    def _clear_memory(self):
        """Force garbage collection to free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_data_in_batches(self, batch_size: int = 500, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from Qdrant in batches to conserve memory."""
        log(f"Loading data from Qdrant collection: {self.collection_name}")
        
        # Check if collection exists
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = self.client.count(self.collection_name).count
            log(f"Total records in collection: {total_points}")
        except Exception as e:
            log(f"Error accessing collection: {e}")
            return []
        
        # Initialize data structures
        all_records = []
        offset = None
        total_loaded = 0
        
        # Set limit
        if limit is None:
            limit = total_points
        
        # Load data in batches
        while total_loaded < limit:
            try:
                # Calculate remaining points to fetch
                remaining = min(batch_size, limit - total_loaded)
                if remaining <= 0:
                    break
                
                # Scroll through points
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=remaining,
                    offset=offset
                )
                
                # Handle different return formats
                if isinstance(result, tuple) and len(result) > 0:
                    points, offset = result
                else:
                    points = getattr(result, 'points', [])
                    offset = getattr(result, 'next_page_offset', None)
                
                if not points:
                    break
                
                # Process points
                for point in points:
                    # Extract point data
                    if hasattr(point, 'payload'):
                        payload = point.payload
                        point_id = point.id
                    else:
                        payload = point.get('payload', {})
                        point_id = point.get('id')
                    
                    # Extract text and metadata
                    text = payload.get('text')
                    if not text:
                        continue
                    
                    # Add to records
                    all_records.append({
                        'id': point_id,
                        'text': text,
                        'metadata': payload.get('metadata', {})
                    })
                
                # Update counter
                total_loaded += len(points)
                log(f"Loaded {total_loaded} records so far")
                
                # Clear memory periodically
                if total_loaded % 5000 == 0:
                    self._clear_memory()
                
                # Check if we've reached the end
                if offset is None:
                    break
                
            except Exception as e:
                log(f"Error loading batch: {e}")
                traceback.print_exc()
                time.sleep(1)  # Small delay before retrying
        
        # Count records with ground truth
        has_ground_truth = sum(1 for r in all_records if r.get('metadata', {}).get('sentiment') is not None)
        log(f"Loaded {len(all_records)} records, {has_ground_truth} with ground truth")
        
        return all_records
    
    def analyze_sentiment_in_batches(self, records: List[Dict[str, Any]], processing_batch_size: int = 32) -> List[Dict[str, Any]]:
        """Process sentiment analysis in small batches to avoid memory issues."""
        log(f"Analyzing sentiment with {self.model_name} model")
        
        results = []
        total_batches = (len(records) + processing_batch_size - 1) // processing_batch_size
        
        # Process a small sample first to understand the label format
        sample_batch = records[:min(5, len(records))]
        sample_texts = [r['text'] for r in sample_batch]
        
        try:
            sample_results = self.sentiment_pipeline(sample_texts)
            # Check the format of the first result to understand how to parse labels
            if sample_results and len(sample_results) > 0:
                first_result = sample_results[0]
                log(f"Sample result format: {first_result}")
        except Exception as e:
            log(f"Error processing sample batch: {e}")
            traceback.print_exc()
        
        # Process in batches
        for i in tqdm(range(0, len(records), processing_batch_size), desc="Processing batches", total=total_batches):
            batch = records[i:i+processing_batch_size]
            texts = [r['text'] for r in batch]
            
            try:
                # Process sentiment
                sentiments = self.sentiment_pipeline(texts)
                
                # Format results
                for j, sentiment_result in enumerate(sentiments):
                    record = batch[j].copy()
                    
                    # Extract sentiment scores
                    try:
                        if isinstance(sentiment_result, list):
                            # For direct label outputs (neutral, positive, negative)
                            scores = {}
                            for item in sentiment_result:
                                # Handle both formats: LABEL_X or direct label name
                                if 'LABEL_' in str(item.get('label', '')):
                                    # Format: LABEL_X
                                    try:
                                        label_id = int(item['label'].split('_')[-1])
                                        label = self.id2label.get(label_id, item['label'])
                                    except (ValueError, IndexError):
                                        label = item['label']
                                else:
                                    # Direct label name (neutral, positive, etc)
                                    label = item['label']
                                
                                scores[label] = item['score']
                            
                            # Determine the predicted label
                            predicted_label = max(scores.items(), key=lambda x: x[1])[0]
                            predicted_score = scores[predicted_label]
                        else:
                            # Handle case where only top prediction is returned
                            if 'LABEL_' in str(sentiment_result.get('label', '')):
                                try:
                                    label_id = int(sentiment_result['label'].split('_')[-1])
                                    predicted_label = self.id2label.get(label_id, sentiment_result['label'])
                                except (ValueError, IndexError):
                                    predicted_label = sentiment_result['label']
                            else:
                                predicted_label = sentiment_result['label']
                                
                            predicted_score = sentiment_result['score']
                            scores = {predicted_label: predicted_score}
                        
                        # Add sentiment results to record
                        record['predicted_sentiment'] = predicted_label
                        record['sentiment_score'] = predicted_score
                        record['sentiment_scores'] = scores
                    
                    except Exception as e:
                        log(f"Error processing result {j} in batch {i//processing_batch_size}: {e}")
                        log(f"Result format: {sentiment_result}")
                        record['error'] = f"Error processing result: {str(e)}"
                    
                    results.append(record)
                
                # Clear memory after each batch
                if (i + processing_batch_size) % (processing_batch_size * 10) == 0:
                    self._clear_memory()
                
            except Exception as e:
                log(f"Error processing batch {i//processing_batch_size}: {e}")
                traceback.print_exc()
                
                # Add records with error flag
                for j in range(len(batch)):
                    record = batch[j].copy()
                    record['error'] = str(e)
                    results.append(record)
                
                # Clear memory after error
                self._clear_memory()
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """Save sentiment analysis results to file."""
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(results)
        
        # Save as CSV and JSON
        csv_file = f"{output_file}.csv"
        json_file = f"{output_file}.json"
        
        df.to_csv(csv_file, index=False)
        df.to_json(json_file, orient='records', lines=True)
        
        log(f"Results saved to {csv_file} and {json_file}")
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute sentiment distribution and other metrics."""
        sentiment_counts = {}
        
        # Count predictions
        for result in results:
            sentiment = result.get('predicted_sentiment')
            if sentiment:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Calculate percentages
        total = len(results)
        sentiment_percentages = {k: (v / total) * 100 for k, v in sentiment_counts.items()}
        
        return {
            "total_records": total,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages
        }
    
    def run_analysis(self, limit: Optional[int] = None, output_file: str = "sentiment_results") -> Dict[str, Any]:
        """Run the complete sentiment analysis pipeline."""
        # Load data
        records = self.load_data_in_batches(batch_size=500, limit=limit)
        
        if not records:
            log("No records found for analysis")
            return {"error": "No records found"}
        
        # Analyze sentiment
        results = self.analyze_sentiment_in_batches(records)
        
        # Save results
        self.save_results(results, output_file)
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        return {
            "results_count": len(results),
            "metrics": metrics,
            "output_files": [f"{output_file}.csv", f"{output_file}.json"]
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Memory-efficient sentiment analysis')
    parser.add_argument('--model', default='Ammar-alhaj-ali/arabic-MARBERT-sentiment', help='Sentiment model name')
    parser.add_argument('--device', default=None, help='Device to use (cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=16, help='Processing batch size')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--collection', default='sentiment_analysis_dataset', help='Collection name')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records to process')
    parser.add_argument('--output', default='sentiment_results', help='Output file prefix')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"ARABIC SENTIMENT ANALYSIS WITH {args.model.split('/')[-1].upper()}")
    print("="*80)
    
    # Initialize analyzer
    analyzer = MemoryEfficientSentimentAnalyzer(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection
    )
    
    # Run analysis
    results = analyzer.run_analysis(limit=args.limit, output_file=args.output)
    
    # Print summary
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*80)
    print(f"Total records processed: {results.get('results_count', 0)}")
    
    metrics = results.get('metrics', {})
    if metrics:
        print("\nSentiment distribution:")
        for sentiment, count in metrics.get('sentiment_counts', {}).items():
            percentage = metrics.get('sentiment_percentages', {}).get(sentiment, 0)
            print(f"  - {sentiment}: {count} ({percentage:.2f}%)")
    
    print("\nOutput files:")
    for file in results.get('output_files', []):
        print(f"  - {file}")
    
    print("="*80)

if __name__ == "__main__":
    main()