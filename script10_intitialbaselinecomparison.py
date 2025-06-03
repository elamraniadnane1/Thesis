import os
import json
import argparse
import gc
import time
import psutil
import sys
import traceback
from typing import Dict, List, Any, Optional, Union
from collections import Counter
import re
import math
import random
import pickle
import requests
from pathlib import Path

# Minimize imports of heavy libraries
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient

# Set to True to enable detailed logging
DEBUG = True

def log(message: str) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")

def print_memory_usage():
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    log(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

class LightweightSentimentAnalyzer:
    """
    Ultra-lightweight sentiment analysis without heavy ML libraries.
    Uses a combination of lexicon-based approach and optional API calls.
    """
    
    def __init__(
        self,
        lexicon_path: Optional[str] = None,
        use_api: bool = False,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "sentiment_analysis_dataset",
        cache_dir: str = "./sentiment_cache"
    ):
        """Initialize the analyzer with minimal resource usage."""
        # Memory usage tracking
        self.initial_memory = psutil.Process(os.getpid()).memory_info().rss
        log(f"Initial memory usage: {self.initial_memory / 1024 / 1024:.2f} MB")
        
        # API settings for external sentiment analysis
        self.use_api = use_api
        self.api_url = api_url
        self.api_key = api_key
        
        # Initialize sentiment lexicon
        self.lexicon = self._load_lexicon(lexicon_path)
        
        # Connect to Qdrant with minimal settings
        try:
            self.client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                api_key=qdrant_api_key,
                prefer_grpc=False,  # Use HTTP to reduce dependencies
                timeout=120.0       # Longer timeout for slow systems
            )
            self.collection_name = collection_name
        except Exception as e:
            log(f"Error connecting to Qdrant: {e}")
            self.client = None
        
        # Setup cache directory for incremental processing
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Standard sentiment labels
        self.sentiment_labels = ["negative", "neutral", "positive"]
        
        # Cache for results to avoid reprocessing
        self.results_cache = {}
        self.cache_hits = 0
        
        # Mapping for sentiment normalization
        self.sentiment_mapping = {
            # Positive variants
            "positive": "positive",
            "pos": "positive",
            "p": "positive",
            "1": "positive",
            1: "positive",
            "2": "positive",  # In some systems positive is 2
            2: "positive",
            "4": "positive",  # In 5-point scale, 4-5 are positive
            4: "positive",
            "5": "positive",
            5: "positive",
            
            # Negative variants
            "negative": "negative",
            "neg": "negative",
            "n": "negative",
            "-1": "negative",
            -1: "negative",
            "0": "negative",  # In some systems negative is 0
            0: "negative",
            "1/5": "negative",  # In 5-point scale, 1-2 are negative
            "2/5": "negative",
            
            # Neutral variants
            "neutral": "neutral",
            "neu": "neutral",
            "0": "neutral",  # In some systems neutral is 0
            0: "neutral",
            "3": "neutral",  # In 5-point scale, 3 is neutral
            3: "neutral",
        }
    
    def _load_lexicon(self, lexicon_path: Optional[str]) -> Dict[str, float]:
        """Load sentiment lexicon from file or use a basic built-in one."""
        if lexicon_path and os.path.exists(lexicon_path):
            log(f"Loading lexicon from {lexicon_path}")
            lexicon = {}
            try:
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            word = parts[0]
                            score = float(parts[1])
                            lexicon[word] = score
                log(f"Loaded {len(lexicon)} words from lexicon")
                return lexicon
            except Exception as e:
                log(f"Error loading lexicon: {e}")
        
        # Basic Arabic sentiment words if no lexicon is provided
        # This is a minimal set to reduce memory usage
        log("Using built-in minimal lexicon")
        return {
            # Positive words
            "جيد": 1.0,
            "رائع": 1.0,
            "ممتاز": 1.0,
            "جميل": 1.0,
            "رائعة": 1.0,
            "أحب": 1.0,
            "ممتازة": 1.0,
            "شكرا": 1.0,
            "سعيد": 1.0,
            "حلو": 1.0,
            "فرح": 1.0,
            "نجاح": 1.0,
            "مفيد": 1.0,
            "حب": 1.0,
            
            # Negative words
            "سيء": -1.0,
            "سيئة": -1.0,
            "مشكلة": -1.0,
            "خطأ": -1.0,
            "حزين": -1.0,
            "ضعيف": -1.0,
            "فشل": -1.0,
            "سوء": -1.0,
            "ضعيفة": -1.0,
            "قبيح": -1.0,
            "قبيحة": -1.0,
            "غاضب": -1.0,
            "غضب": -1.0,
            "مشكل": -1.0,
            "صعب": -1.0,
            "مؤسف": -1.0,
            
            # Some English words that might appear in mixed text
            "good": 1.0,
            "great": 1.0,
            "excellent": 1.0,
            "bad": -1.0,
            "poor": -1.0,
            "terrible": -1.0
        }
    
    def _clear_memory(self):
        """Force garbage collection to free memory."""
        gc.collect()
        
        # Report memory usage
        current_memory = psutil.Process(os.getpid()).memory_info().rss
        diff = current_memory - self.initial_memory
        log(f"Current memory: {current_memory / 1024 / 1024:.2f} MB")
        log(f"Memory increase: {diff / 1024 / 1024:.2f} MB")
    
    def _normalize_sentiment_label(self, label):
        """Normalize sentiment labels to a standard format."""
        if label is None:
            return None
        
        # Convert label to string for consistency in lookups
        try:
            str_label = str(label).lower().strip()
        except:
            return None
        
        # Check direct match first
        if str_label in self.sentiment_mapping:
            return self.sentiment_mapping[str_label]
            
        # Check numerical values that might be stored as strings
        try:
            num_label = float(str_label)
            if num_label in self.sentiment_mapping:
                return self.sentiment_mapping[num_label]
        except (ValueError, TypeError):
            pass
            
        # Check for common substrings
        for key, standardized in [
            ("pos", "positive"),
            ("neg", "negative"),
            ("neut", "neutral")
        ]:
            if key in str_label:
                return standardized
                
        # If no match found, return the original label
        return str_label
    
    def _lexicon_sentiment(self, text: str) -> Dict[str, Any]:
        """Calculate sentiment using simple lexicon-based approach."""
        # Check cache first
        text_hash = hash(text)
        if text_hash in self.results_cache:
            self.cache_hits += 1
            return self.results_cache[text_hash]
        
        # Clean text
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned_text.split()
        
        # Count positive and negative words
        pos_score = 0
        neg_score = 0
        
        for word in words:
            if word in self.lexicon:
                score = self.lexicon[word]
                if score > 0:
                    pos_score += score
                elif score < 0:
                    neg_score += abs(score)
        
        # Calculate overall sentiment
        total_score = pos_score - neg_score
        
        # Determine sentiment label
        if total_score > 0.1:  # Small threshold for positive
            sentiment = "positive"
            confidence = min(1.0, pos_score / (pos_score + neg_score + 0.1))
        elif total_score < -0.1:  # Small threshold for negative
            sentiment = "negative"
            confidence = min(1.0, neg_score / (pos_score + neg_score + 0.1))
        else:
            sentiment = "neutral"
            # Calculate confidence based on how close to zero
            confidence = 1.0 - min(1.0, abs(total_score) / 0.1)
        
        # Prepare result
        result = {
            "label": sentiment,
            "score": confidence,
            "pos_score": pos_score,
            "neg_score": neg_score,
            "total_score": total_score
        }
        
        # Cache result
        self.results_cache[text_hash] = result
        
        # Limit cache size to avoid memory issues
        if len(self.results_cache) > 10000:
            # Remove random 1000 items
            keys_to_remove = random.sample(list(self.results_cache.keys()), 1000)
            for key in keys_to_remove:
                del self.results_cache[key]
        
        return result
    
    def _api_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using external API."""
        if not self.use_api or not self.api_url:
            return self._lexicon_sentiment(text)
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.post(
                self.api_url,
                json={"text": text},
                headers=headers,
                timeout=5  # Short timeout to avoid hanging
            )
            
            if response.status_code == 200:
                result = response.json()
                # Normalize API result to our format
                if isinstance(result, dict) and "label" in result:
                    return {
                        "label": result["label"],
                        "score": result.get("score", 0.5),
                        "api_result": result
                    }
                else:
                    return {
                        "label": "neutral",
                        "score": 0.5,
                        "error": "Unexpected API response format",
                        "api_result": result
                    }
            else:
                log(f"API error: {response.status_code} {response.text}")
                # Fall back to lexicon approach
                return self._lexicon_sentiment(text)
        except Exception as e:
            log(f"API request failed: {e}")
            # Fall back to lexicon approach
            return self._lexicon_sentiment(text)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using the appropriate method."""
        if self.use_api:
            return self._api_sentiment(text)
        else:
            return self._lexicon_sentiment(text)
    
    def load_data_in_batches(self, batch_size: int = 50, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Load data from Qdrant in tiny batches to conserve memory."""
        log(f"Loading data from Qdrant collection: {self.collection_name}")
        
        # Check if collection exists
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total_points = self.client.count(self.collection_name).count
            log(f"Total records in collection: {total_points}")
        except Exception as e:
            log(f"Error accessing collection: {e}")
            return []
        
        # Check for existing cache files
        existing_records = []
        cache_file_pattern = self.cache_dir / f"{self.collection_name}_records_*.pkl"
        cache_files = list(self.cache_dir.glob(f"{self.collection_name}_records_*.pkl"))
        
        if cache_files and offset == 0:
            log(f"Found {len(cache_files)} cache files. Loading...")
            for cache_file in tqdm(cache_files, desc="Loading cached records"):
                try:
                    with open(cache_file, 'rb') as f:
                        batch_records = pickle.load(f)
                        existing_records.extend(batch_records)
                except Exception as e:
                    log(f"Error loading cache file {cache_file}: {e}")
            
            log(f"Loaded {len(existing_records)} records from cache")
            
            if limit is not None and len(existing_records) >= limit:
                return existing_records[:limit]
            elif len(existing_records) >= total_points:
                return existing_records
        
        # Initialize data structures
        all_records = existing_records.copy()
        qdrant_offset = None
        total_loaded = len(existing_records) + offset
        ground_truth_count = sum(1 for r in all_records if r.get('ground_truth') is not None)
        
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
                
                # Progress update
                log(f"Loading batch {total_loaded}-{total_loaded + remaining} of {limit}")
                print_memory_usage()
                
                # Scroll through points
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=remaining,
                    offset=qdrant_offset
                )
                
                # Handle different return formats
                if isinstance(result, tuple) and len(result) > 0:
                    points, qdrant_offset = result
                else:
                    points = getattr(result, 'points', [])
                    qdrant_offset = getattr(result, 'next_page_offset', None)
                
                if not points:
                    break
                
                # Process points
                batch_records = []
                for point in points:
                    try:
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
                        
                        # Extract ground truth sentiment if available
                        metadata = payload.get('metadata', {})
                        
                        # Look for sentiment in various possible locations
                        ground_truth = None
                        
                        # Check direct sentiment field in metadata
                        if 'sentiment' in metadata:
                            ground_truth = metadata['sentiment']
                        
                        # Check for sentiment in raw_payload if available
                        raw_payload = payload.get('raw_payload', {})
                        if not ground_truth and 'sentiment' in raw_payload:
                            ground_truth = raw_payload['sentiment']
                        
                        # Check for sentiment score or rating
                        for field in ['sentiment_score', 'rating', 'score', 'label']:
                            if not ground_truth and field in metadata:
                                ground_truth = metadata[field]
                            if not ground_truth and field in raw_payload:
                                ground_truth = raw_payload[field]
                        
                        # Normalize the ground truth label if found
                        if ground_truth is not None:
                            ground_truth = self._normalize_sentiment_label(ground_truth)
                            ground_truth_count += 1
                        
                        # Create minimal record to save memory
                        record = {
                            'id': str(point_id),
                            'text': text,
                            'ground_truth': ground_truth
                        }
                        
                        # Add to batch records
                        batch_records.append(record)
                    except Exception as e:
                        log(f"Error processing point: {e}")
                
                # Save batch to cache
                if batch_records:
                    cache_file = self.cache_dir / f"{self.collection_name}_records_{total_loaded}.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(batch_records, f)
                
                # Add batch to all records
                all_records.extend(batch_records)
                
                # Update counter
                total_loaded += len(batch_records)
                log(f"Loaded {total_loaded} records so far, {ground_truth_count} with ground truth")
                
                # Clear memory periodically
                if total_loaded % (batch_size * 10) == 0:
                    self._clear_memory()
                
                # Check if we've reached the end
                if qdrant_offset is None:
                    break
                
            except Exception as e:
                log(f"Error loading batch: {e}")
                traceback.print_exc()
                time.sleep(1)  # Small delay before retrying
        
        return all_records
    
    def analyze_records_in_batches(self, records: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        """Process sentiment analysis in tiny batches to avoid memory issues."""
        log(f"Analyzing sentiment for {len(records)} records")
        
        results = []
        total_batches = math.ceil(len(records) / batch_size)
        
        # Check for existing results
        results_file = self.cache_dir / f"{self.collection_name}_results.pkl"
        if results_file.exists():
            try:
                with open(results_file, 'rb') as f:
                    existing_results = pickle.load(f)
                    log(f"Loaded {len(existing_results)} existing results")
                    
                    # If we already have all results, return them
                    if len(existing_results) >= len(records):
                        log("All records already analyzed")
                        return existing_results[:len(records)]
                    
                    # Otherwise, use existing results as starting point
                    results = existing_results
            except Exception as e:
                log(f"Error loading existing results: {e}")
        
        # Calculate how many records we still need to process
        records_to_process = records[len(results):]
        log(f"Processing {len(records_to_process)} new records")
        
        # Process in batches
        for i in tqdm(range(0, len(records_to_process), batch_size), desc="Processing batches", total=math.ceil(len(records_to_process)/batch_size)):
            batch = records_to_process[i:i+batch_size]
            
            batch_results = []
            for record in batch:
                try:
                    # Get sentiment
                    sentiment_result = self.analyze_sentiment(record['text'])
                    
                    # Create result record
                    result = record.copy()
                    result['predicted_sentiment'] = sentiment_result['label']
                    result['sentiment_score'] = sentiment_result['score']
                    result['sentiment_details'] = sentiment_result
                    
                    # Add comparison with ground truth
                    if result.get('ground_truth') is not None:
                        result['correct'] = result['ground_truth'] == result['predicted_sentiment']
                    else:
                        result['correct'] = None
                    
                    batch_results.append(result)
                except Exception as e:
                    log(f"Error analyzing record: {e}")
                    # Add record with error
                    result = record.copy()
                    result['error'] = str(e)
                    batch_results.append(result)
            
            # Add batch results
            results.extend(batch_results)
            
            # Save incremental results
            if i % (batch_size * 10) == 0 or i + batch_size >= len(records_to_process):
                with open(results_file, 'wb') as f:
                    pickle.dump(results, f)
                log(f"Saved {len(results)} results")
            
            # Clear memory
            if i % (batch_size * 10) == 0:
                self._clear_memory()
        
        return results
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute sentiment distribution and evaluation metrics."""
        log("Computing metrics")
        print_memory_usage()
        
        # Basic sentiment distribution
        sentiment_counts = Counter()
        for result in results:
            sentiment = result.get('predicted_sentiment')
            if sentiment:
                sentiment_counts[sentiment] += 1
        
        # Calculate percentages
        total = len(results)
        sentiment_percentages = {k: (v / total) * 100 for k, v in sentiment_counts.items()}
        
        # Evaluation metrics if ground truth is available
        eval_metrics = {
            "has_ground_truth": False,
            "accuracy": None,
            "confusion_matrix": None,
            "class_metrics": None
        }
        
        # Extract records with ground truth
        records_with_ground_truth = [r for r in results if r.get('ground_truth') is not None]
        
        if records_with_ground_truth:
            eval_metrics["has_ground_truth"] = True
            eval_metrics["ground_truth_count"] = len(records_with_ground_truth)
            
            # Extract ground truth and predictions
            y_true = [r['ground_truth'] for r in records_with_ground_truth]
            y_pred = [r['predicted_sentiment'] for r in records_with_ground_truth]
            
            # Calculate accuracy
            correct = sum(1 for r in records_with_ground_truth if r.get('correct', False))
            eval_metrics["accuracy"] = correct / len(records_with_ground_truth) if records_with_ground_truth else 0
            
            # Create confusion matrix
            labels = list(set(y_true + y_pred))
            confusion = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
            
            for true, pred in zip(y_true, y_pred):
                confusion[true][pred] += 1
            
            eval_metrics["confusion_matrix"] = confusion
            eval_metrics["labels"] = labels
            
            # Calculate class-wise metrics
            class_metrics = {}
            for label in labels:
                # True positives, false positives, false negatives
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
                
                # Calculate precision, recall, f1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": sum(1 for t in y_true if t == label)
                }
            
            eval_metrics["class_metrics"] = class_metrics
            
            # Calculate ground truth distribution
            ground_truth_counts = Counter(y_true)
            eval_metrics["ground_truth_distribution"] = {
                "counts": dict(ground_truth_counts),
                "percentages": {k: (v / len(y_true)) * 100 for k, v in ground_truth_counts.items()}
            }
        
        return {
            "total_records": total,
            "sentiment_counts": dict(sentiment_counts),
            "sentiment_percentages": sentiment_percentages,
            "evaluation": eval_metrics,
            "cache_hits": self.cache_hits
        }
    
    def save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any], output_file: str) -> None:
        """Save results and metrics in a memory-efficient way."""
        log(f"Saving results to {output_file}")
        
        # Save metrics as JSON
        metrics_file = f"{output_file}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        # Save results in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i+chunk_size]
            chunk_file = f"{output_file}_chunk_{i//chunk_size}.json"
            
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f)
        
        log(f"Results saved in {math.ceil(len(results)/chunk_size)} chunks")
    
    def generate_report(self, metrics: Dict[str, Any], output_file: str) -> None:
        """Generate a simple text report with the metrics."""
        log(f"Generating report to {output_file}")
        
        report = []
        report.append("=" * 80)
        report.append("SENTIMENT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Total records: {metrics['total_records']}")
        report.append("")
        
        report.append("Sentiment distribution:")
        for sentiment, count in metrics['sentiment_counts'].items():
            percentage = metrics['sentiment_percentages'][sentiment]
            report.append(f"  - {sentiment}: {count} ({percentage:.2f}%)")
        report.append("")
        
        if metrics['evaluation']['has_ground_truth']:
            report.append(f"Records with ground truth: {metrics['evaluation']['ground_truth_count']}")
            report.append(f"Overall accuracy: {metrics['evaluation']['accuracy']:.4f} ({metrics['evaluation']['accuracy']*100:.2f}%)")
            report.append("")
            
            report.append("Class metrics:")
            for label, metrics_dict in metrics['evaluation']['class_metrics'].items():
                report.append(f"  {label}:")
                report.append(f"    Precision: {metrics_dict['precision']:.4f}")
                report.append(f"    Recall: {metrics_dict['recall']:.4f}")
                report.append(f"    F1-score: {metrics_dict['f1']:.4f}")
                report.append(f"    Support: {metrics_dict['support']}")
            report.append("")
            
            report.append("Ground truth distribution:")
            for label, count in metrics['evaluation']['ground_truth_distribution']['counts'].items():
                percentage = metrics['evaluation']['ground_truth_distribution']['percentages'][label]
                report.append(f"  - {label}: {count} ({percentage:.2f}%)")
            report.append("")
            
            report.append("Confusion Matrix:")
            labels = metrics['evaluation']['labels']
            matrix = metrics['evaluation']['confusion_matrix']
            
            # Header row
            header = "True \\ Pred |"
            for label in labels:
                header += f" {label} |"
            report.append(header)
            
            # Separator
            report.append("-" * len(header))
            
            # Data rows
            for true_label in labels:
                row = f"{true_label} |"
                for pred_label in labels:
                    row += f" {matrix[true_label][pred_label]} |"
                report.append(row)
        
        report.append("")
        report.append("=" * 80)
        
        # Write report to file
        with open(f"{output_file}_report.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
    
    def run_analysis(self, limit: Optional[int] = None, batch_size: int = 50, output_file: str = "sentiment_results") -> Dict[str, Any]:
        """Run the complete sentiment analysis pipeline with minimal memory usage."""
        start_time = time.time()
        
        # Load data
        records = self.load_data_in_batches(batch_size=batch_size, limit=limit)
        
        if not records:
            log("No records found for analysis")
            return {"error": "No records found"}
        
        # Analyze sentiment
        results = self.analyze_records_in_batches(records, batch_size=batch_size)
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        # Save results
        self.save_results(results, metrics, output_file)
        
        # Generate report
        self.generate_report(metrics, output_file)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        log(f"Analysis complete. Total time: {elapsed:.2f} seconds")
        print_memory_usage()
        
        return {
            "results_count": len(results),
            "metrics": metrics,
            "output_files": [
                f"{output_file}_metrics.json",
                f"{output_file}_report.txt",
                f"{output_file}_chunk_*.json"
            ],
            "elapsed_time": elapsed
        }

def main():
    """Main entry point with extensive error handling and memory monitoring."""
    parser = argparse.ArgumentParser(description='Ultra-lightweight sentiment analysis')
    parser.add_argument('--lexicon', default=None, help='Path to sentiment lexicon file')
    parser.add_argument('--api', action='store_true', help='Use external API for sentiment analysis')
    parser.add_argument('--api_url', default=None, help='URL for sentiment analysis API')
    parser.add_argument('--api_key', default=None, help='API key for sentiment analysis')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--collection', default='sentiment_analysis_dataset', help='Collection name')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records to process')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--output', default='sentiment_results', help='Output file prefix')
    parser.add_argument('--cache_dir', default='./sentiment_cache', help='Directory for caching results')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("ULTRA-LIGHTWEIGHT SENTIMENT ANALYSIS")
    print("=" * 80)
    print(f"System memory: {psutil.virtual_memory().total / (1024**3):.2f} GB total, {psutil.virtual_memory().available / (1024**3):.2f} GB available")
    
    try:
        # Initialize analyzer
        analyzer = LightweightSentimentAnalyzer(
            lexicon_path=args.lexicon,
            use_api=args.api,
            api_url=args.api_url,
            api_key=args.api_key,
            qdrant_host=args.host,
            qdrant_port=args.port,
            collection_name=args.collection,
            cache_dir=args.cache_dir
        )
        
        # Run analysis
        results = analyzer.run_analysis(
            limit=args.limit,
            batch_size=args.batch_size,
            output_file=args.output
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("SENTIMENT ANALYSIS RESULTS")
        print("=" * 80)
        print(f"Total records processed: {results.get('results_count', 0)}")
        
        metrics = results.get('metrics', {})
        if metrics:
            print("\nSentiment distribution:")
            for sentiment, count in metrics.get('sentiment_counts', {}).items():
                percentage = metrics.get('sentiment_percentages', {}).get(sentiment, 0)
                print(f"  - {sentiment}: {count} ({percentage:.2f}%)")
            
            # Print evaluation metrics if available
            evaluation = metrics.get('evaluation', {})
            if evaluation.get('has_ground_truth', False):
                print("\nEvaluation metrics:")
                print(f"  - Records with ground truth: {evaluation.get('ground_truth_count', 0)}")
                print(f"  - Overall accuracy: {evaluation.get('accuracy', 0):.4f} ({evaluation.get('accuracy', 0)*100:.2f}%)")
        
        print("\nOutput files:")
        for file_pattern in results.get('output_files', []):
            print(f"  - {file_pattern}")
        
        print(f"\nTotal time: {results.get('elapsed_time', 0):.2f} seconds")
        print("=" * 80)
        
    except Exception as e:
        print("\nERROR: Analysis failed")
        print(f"Error: {e}")
        traceback.print_exc()
        print("\nMemory information:")
        print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        print(f"Used memory: {psutil.virtual_memory().used / (1024**3):.2f} GB")
        print(f"Process memory: {psutil.Process(os.getpid()).memory_info().rss / (1024**3):.2f} GB")
        sys.exit(1)

if __name__ == "__main__":
    main()