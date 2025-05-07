import pandas as pd
import numpy as np
import os
import json
import mlflow
import mlflow.sklearn
from anthropic import Anthropic
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Any, Optional
import argparse
import logging
from tqdm import tqdm
import csv
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model_quality(df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """
    Evaluate model quality based on confidence scores, sentiment distribution,
    and consistency between issue and solution sentiments.
    
    Args:
        df: DataFrame containing sentiment predictions
        model_name: Name of the model to evaluate
        
    Returns:
        Dictionary with quality metrics
    """
    # Check if model columns exist
    issue_col = f"{model_name}_issue_sentiment"
    issue_conf_col = f"{model_name}_issue_confidence"
    solution_col = f"{model_name}_solution_sentiment"
    solution_conf_col = f"{model_name}_solution_confidence"
    
    if any(col not in df.columns for col in [issue_col, issue_conf_col, solution_col, solution_conf_col]):
        logger.error(f"Model {model_name} columns not found in dataset")
        return {}
    
    # Calculate average confidence
    avg_issue_confidence = df[issue_conf_col].mean()
    avg_solution_confidence = df[solution_conf_col].mean()
    avg_confidence = (avg_issue_confidence + avg_solution_confidence) / 2
    
    # Evaluate against ground truth when available
    accuracy = 0
    f1_score = 0
    
    if "Ground_Truth" in df.columns:
        # Convert model predictions to binary (positive=1, non-positive=0)
        ground_truth = (df["Ground_Truth"] == "positive").astype(int)
        predictions = (df[issue_col] == "positive").astype(int)
        
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='binary', zero_division=0
        )
    
    # Calculate sentiment improvement (solutions should generally be more positive than issues)
    improvement_col = f"{model_name}_sentiment_improvement"
    if improvement_col in df.columns:
        positive_improvements = (df[improvement_col] > 0).mean()
    else:
        # Calculate improvement manually if not provided
        sentiment_values = {"positive": 1, "neutral": 0, "negative": -1}
        issue_values = df[issue_col].map(sentiment_values)
        solution_values = df[solution_col].map(sentiment_values)
        improvements = solution_values - issue_values
        positive_improvements = (improvements > 0).mean()
    
    # Calculate sentiment diversity (a good model should show appropriate distribution)
    issue_distribution = df[issue_col].value_counts(normalize=True)
    solution_distribution = df[solution_col].value_counts(normalize=True)
    
    # Calculate quality score (weighted combination of metrics)
    quality_score = (
        avg_confidence * 0.3 +           # Confidence is important
        accuracy * 0.3 +                 # Accuracy against ground truth
        f1_score * 0.2 +                 # F1 score against ground truth
        positive_improvements * 0.2      # Appropriate improvement direction
    )
    
    return {
        "model": model_name,
        "avg_confidence": avg_confidence,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "positive_improvements": positive_improvements,
        "quality_score": quality_score
    }

def clean_text(text: str) -> str:
    """Clean and prepare text for analysis"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Basic text cleaning
    cleaned = text.strip()
    # Limit very long texts to avoid memory issues
    if len(cleaned) > 500:
        cleaned = cleaned[:500]
    return cleaned

def get_anthropic_sentiment(text: str, client: Anthropic) -> Dict[str, Any]:
    """
    Get sentiment analysis from Anthropic API
    
    Args:
        text: Text to analyze
        client: Anthropic client
        
    Returns:
        Dictionary with sentiment and confidence
    """
    if not isinstance(text, str) or not text.strip():
        return {"sentiment": "neutral", "confidence": 0.5}
    
    # Clean the text first
    text = clean_text(text)
    
    prompt = f"""
    Analyze the sentiment of the following text and classify it as 'positive', 'negative', or 'neutral'.
    Also provide a confidence score between 0 and 1.
    
    Text: "{text}"
    
    Respond in JSON format with 'sentiment' and 'confidence' fields.
    """
    
    try:
        # Updated to use the current model available in 2025
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",  # Updated model name
            max_tokens=150,
            temperature=0,
            system="You are a sentiment analysis expert. You analyze text and return only sentiment with confidence score.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Try to extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            try:
                result = json.loads(json_str)
                # Ensure the required fields are present
                if "sentiment" in result and "confidence" in result:
                    return result
            except json.JSONDecodeError:
                pass
        
        # Fallback: Parse the response manually
        if "positive" in response_text.lower():
            sentiment = "positive"
        elif "negative" in response_text.lower():
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Default confidence
        confidence = 0.7
        
        # Try to extract confidence value
        if "confidence" in response_text.lower():
            confidence_text = response_text.lower().split("confidence")[1]
            confidence_vals = [float(s) for s in confidence_text.split() 
                              if s.replace('.','',1).isdigit()]
            if confidence_vals:
                confidence = confidence_vals[0]
        
        return {"sentiment": sentiment, "confidence": confidence}
            
    except Exception as e:
        logger.error(f"Error calling Anthropic API: {e}")
        return {"sentiment": "neutral", "confidence": 0.5}

def evaluate_with_mlflow(df: pd.DataFrame, best_model: str, client: Anthropic, 
                         max_samples: Optional[int] = None, 
                         batch_size: int = 50,
                         process_limit: Optional[int] = None):
    """
    Evaluate Anthropic API performance against the best model using MLflow
    
    Args:
        df: DataFrame with data
        best_model: Name of the best model to use as ground truth
        client: Anthropic client
        max_samples: Maximum number of samples to evaluate (None for all)
        batch_size: Number of samples to process in each batch
        process_limit: Maximum number of entries to process (None for all)
        
    Returns:
        Dictionary with evaluation metrics
    """
    with mlflow.start_run(run_name=f"anthropic_vs_{best_model}"):
        # Apply process limit first (to take first N entries)
        if process_limit is not None and process_limit < len(df):
            logger.info(f"Limiting processing to first {process_limit} entries")
            df = df.iloc[:process_limit]
            
        # Then apply max_samples if specified (to sample randomly from remaining entries)
        if max_samples is not None and max_samples < len(df):
            logger.info(f"Using {max_samples} samples out of {len(df)} total")
            df_to_evaluate = df.sample(max_samples, random_state=42)
        else:
            logger.info(f"Using all {len(df)} samples for evaluation")
            df_to_evaluate = df
        
        # Log parameters
        mlflow.log_param("ground_truth_model", best_model)
        mlflow.log_param("anthropic_model", "claude-3-7-sonnet-20250219")
        mlflow.log_param("num_samples", len(df_to_evaluate))
        mlflow.log_param("batch_size", batch_size)
        
        # Get ground truth values from the best model
        issue_ground_truth = df_to_evaluate[f"{best_model}_issue_sentiment"].values
        
        # Process in batches to handle large datasets efficiently and save progress
        all_predictions = []
        
        # Calculate number of batches
        num_batches = (len(df_to_evaluate) + batch_size - 1) // batch_size
        
        # Create DataFrame to store progress results instead of direct file writing
        progress_df = pd.DataFrame(columns=["text", "ground_truth", "prediction", "confidence"])
        
        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df_to_evaluate))
            
            batch_df = df_to_evaluate.iloc[start_idx:end_idx]
            batch_texts = batch_df["issue"].values
            batch_ground_truth = batch_df[f"{best_model}_issue_sentiment"].values
            
            logger.info(f"Processing batch {batch_idx+1}/{num_batches} (entries {start_idx+1}-{end_idx})")
            
            # Get predictions from Anthropic for this batch
            batch_predictions = []
            batch_results = []
            
            for i, text in enumerate(tqdm(batch_texts, desc=f"Batch {batch_idx+1}/{num_batches}")):
                result = get_anthropic_sentiment(text, client)
                batch_predictions.append(result)
                
                # Add to batch results
                batch_results.append({
                    "text": text,
                    "ground_truth": batch_ground_truth[i],
                    "prediction": result["sentiment"],
                    "confidence": result["confidence"]
                })
            
            # Add batch results to progress DataFrame
            batch_progress_df = pd.DataFrame(batch_results)
            progress_df = pd.concat([progress_df, batch_progress_df], ignore_index=True)
            
            # Save progress after each batch (this avoids encoding issues)
            progress_df.to_csv("anthropic_evaluation_progress.csv", index=False, encoding='utf-8')
            
            # Accumulate all predictions
            all_predictions.extend(batch_predictions)
            
            logger.info(f"Completed batch {batch_idx+1}/{num_batches}")
            
        # Extract sentiment labels and confidence scores
        anthropic_issue_labels = [pred["sentiment"] for pred in all_predictions]
        anthropic_issue_confidence = [pred["confidence"] for pred in all_predictions]
        
        # Calculate metrics
        issue_accuracy = np.mean(np.array(anthropic_issue_labels) == issue_ground_truth)
        
        # Create a results dataframe
        results_df = pd.DataFrame({
            "text": df_to_evaluate["issue"],
            "ground_truth": issue_ground_truth,
            "anthropic_prediction": anthropic_issue_labels,
            "anthropic_confidence": anthropic_issue_confidence
        })
        
        # Save final results (using UTF-8 encoding)
        final_results_path = "anthropic_evaluation_results.csv"
        results_df.to_csv(final_results_path, index=False, encoding='utf-8')
        mlflow.log_artifact(final_results_path)
        
        # Calculate confusion matrix
        labels = ["positive", "neutral", "negative"]
        cm = confusion_matrix(
            issue_ground_truth, 
            anthropic_issue_labels,
            labels=labels
        )
        
        # Create confusion matrix visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title(f"Anthropic vs {best_model} Sentiment Analysis")
        plt.xlabel("Anthropic Prediction")
        plt.ylabel(f"{best_model} Ground Truth")
        plt.tight_layout()
        
        # Save and log the visualization
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Log metrics
        mlflow.log_metric("accuracy", issue_accuracy)
        mlflow.log_metric("avg_confidence", np.mean(anthropic_issue_confidence))
        
        # Calculate precision, recall, f1 for each sentiment class
        for label in labels:
            true_positives = sum((issue_ground_truth == label) & (np.array(anthropic_issue_labels) == label))
            false_positives = sum((issue_ground_truth != label) & (np.array(anthropic_issue_labels) == label))
            false_negatives = sum((issue_ground_truth == label) & (np.array(anthropic_issue_labels) != label))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            mlflow.log_metric(f"{label}_precision", precision)
            mlflow.log_metric(f"{label}_recall", recall)
            mlflow.log_metric(f"{label}_f1", f1)
        
        return {
            "accuracy": issue_accuracy,
            "avg_confidence": np.mean(anthropic_issue_confidence),
            "confusion_matrix": cm
        }

def save_ground_truth(df: pd.DataFrame, best_model: str, output_file: str = "ground_truth.csv"):
    """
    Save the selected ground truth to a CSV file
    
    Args:
        df: DataFrame with data
        best_model: Name of the best model
        output_file: Output file path
    """
    # Create a copy of the input data
    gt_df = df.copy()
    
    # Add ground truth columns based on the best model
    gt_df["issue_sentiment_ground_truth"] = gt_df[f"{best_model}_issue_sentiment"]
    gt_df["issue_confidence_ground_truth"] = gt_df[f"{best_model}_issue_confidence"]
    gt_df["solution_sentiment_ground_truth"] = gt_df[f"{best_model}_solution_sentiment"]
    gt_df["solution_confidence_ground_truth"] = gt_df[f"{best_model}_solution_confidence"]
    
    # Save to CSV with UTF-8 encoding
    gt_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Ground truth saved to {output_file}")
    
    return gt_df

def main():
    """Main function to run the evaluation pipeline"""
    parser = argparse.ArgumentParser(description="Sentiment Analysis Model Evaluation")
    parser.add_argument("--input", default="sentiment_data.csv", help="Input CSV file")
    parser.add_argument("--output", default="ground_truth.csv", help="Output file for ground truth")
    parser.add_argument("--api-key", help="Anthropic API key (optional)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples for API evaluation (default: all)")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing (default: prompts user)")
    parser.add_argument("--limit", type=int, help="Limit processing to first N entries (default: prompts user)")
    parser.add_argument("--skip-api", action="store_true", help="Skip Anthropic API evaluation")
    args = parser.parse_args()
    
    # Load dataset
    try:
        logger.info(f"Loading dataset from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded dataset with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Extract model names from columns
    model_names = []
    for col in df.columns:
        if col.endswith("_issue_sentiment"):
            model_name = col.replace("_issue_sentiment", "")
            model_names.append(model_name)
    
    logger.info(f"Found models: {', '.join(model_names)}")
    
    # Evaluate each model's quality
    model_results = []
    for model in model_names:
        results = evaluate_model_quality(df, model)
        if results:
            model_results.append(results)
            logger.info(f"Model: {model}")
            logger.info(f"  Avg Confidence: {results['avg_confidence']:.4f}")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  F1 Score: {results['f1_score']:.4f}")
            logger.info(f"  Positive Improvements: {results['positive_improvements']:.4f}")
            logger.info(f"  Quality Score: {results['quality_score']:.4f}")
    
    # Select the best model
    if not model_results:
        logger.error("No valid models found for evaluation")
        return
        
    best_model = max(model_results, key=lambda x: x["quality_score"])
    best_model_name = best_model["model"]
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"  Quality Score: {best_model['quality_score']:.4f}")
    
    # Save ground truth based on the best model
    ground_truth_df = save_ground_truth(df, best_model_name, args.output)
    
    # Skip API evaluation if requested
    if args.skip_api:
        logger.info("Skipping Anthropic API evaluation as requested")
        return
    
    # Set up Anthropic client
    anthropic_api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not provided. Skipping API evaluation.")
        return
    
    client = Anthropic(api_key=anthropic_api_key)
    
    # Configure MLflow
    mlflow_experiment = "sentiment_analysis_evaluation"
    try:
        if mlflow.get_experiment_by_name(mlflow_experiment) is None:
            mlflow.create_experiment(mlflow_experiment)
        mlflow.set_experiment(mlflow_experiment)
    except Exception as e:
        logger.warning(f"MLflow setup error: {e}")
        logger.warning("Continuing without MLflow tracking...")
    
    # Prompt for batch size if not provided in args
    batch_size = args.batch_size
    if batch_size is None:
        try:
            print(f"\nDataset has {len(df)} total entries.")
            batch_size_input = input("Enter batch size for processing (default is 50): ").strip()
            batch_size = int(batch_size_input) if batch_size_input else 50
        except ValueError:
            print("Invalid input. Using default batch size of 50.")
            batch_size = 50
    
    # Prompt for processing limit if not provided in args
    process_limit = args.limit
    if process_limit is None:
        try:
            process_limit_input = input(f"Enter maximum number of entries to process (1-{len(df)}, leave blank to process all): ").strip()
            process_limit = int(process_limit_input) if process_limit_input else None
            if process_limit is not None and (process_limit < 1 or process_limit > len(df)):
                print(f"Invalid limit. Value must be between 1 and {len(df)}.")
                process_limit = None
        except ValueError:
            print("Invalid input. Processing all entries.")
            process_limit = None
    
    # Evaluate Anthropic API
    logger.info(f"Evaluating Anthropic API against {best_model_name}...")
    logger.info(f"Using batch size: {batch_size}")
    if process_limit:
        logger.info(f"Processing limited to first {process_limit} entries")
    
    results = evaluate_with_mlflow(df, best_model_name, client, 
                                  max_samples=args.max_samples,
                                  batch_size=batch_size,
                                  process_limit=process_limit)
    
    logger.info("Evaluation Results:")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Average Confidence: {results['avg_confidence']:.4f}")
    logger.info("MLflow tracking completed")

if __name__ == "__main__":
    main()