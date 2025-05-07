import os
import time
import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st

import openai
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Ensure the basic 'punkt' resource is present (download if missing)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# -------------------------------
# SETTINGS & GLOBAL VARIABLES
# -------------------------------
# Models for evaluation
MODELS = ["gpt-3.5-turbo", "gpt-4", "custom_model_A", "custom_model_B", "baseline_rule_based"]

# Minimum number of experiments per task
MIN_EXPERIMENTS = 10

# Evaluation configuration for tasks
EVAL_CONFIG = {
    "summarization": {"collection": "municipal_projects", "input_key": "description", "reference_key": "title"},
    "sentiment": {"collection": "citizen_comments", "input_key": "comment_text", "reference_key": "sentiment"},
    "offensive": {"collection": "citizen_comments", "input_key": "comment_text", "reference_key": "offensive"},
    "topic": {"collection": "citizen_comments", "input_key": "comment_text", "reference_key": "project_themes"},
    "keywords": {"collection": "citizen_comments", "input_key": "comment_text", "reference_key": "keywords"},
    "engagement": {"collection": "citizen_comments", "input_key": "comment_text", "reference_key": "votes.vote_score"}
}

# Language detection/translation models
LANGUAGE_MODELS = ["gpt-3.5-turbo", "gpt-4", "baseline_rule_based"]

# -------------------------------
# QDRANT HELPER FUNCTIONS
# -------------------------------
def get_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant client."""
    qdrant_url = st.secrets.get("qdrant", {}).get("url", "http://localhost:6333")
    qdrant_api_key = st.secrets.get("qdrant", {}).get("api_key", "")
    
    if qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
        return QdrantClient(url=qdrant_url)

def load_qdrant_documents(collection_name: str, limit: int = 100, filters: Optional[Dict] = None) -> List[Dict]:
    """
    Load documents from the specified Qdrant collection with optional filters.
    """
    try:
        client = get_qdrant_client()
        
        # Check client capabilities and version
        # Different versions of Qdrant client have different parameter names
        
        # Try using 'filter' parameter (newer versions) or 'filters' (older versions)
        try:
            if filters:
                # Convert filters to Qdrant filter format
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key=key,
                            match=qdrant_models.MatchValue(value=value)
                        )
                    )
                filter_query = qdrant_models.Filter(
                    must=filter_conditions
                )
                
                # Try with 'filter' parameter first
                response = client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    filter=filter_query
                )
            else:
                response = client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
        except TypeError:
            # If 'filter' parameter fails, try without filtering
            st.warning("Qdrant client doesn't support the 'filter' parameter. Using alternative approach.")
            
            # Simple approach without filter
            response = client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # If we had filters, do filtering in memory (not efficient but works)
            if filters and response[0]:
                filtered_points = []
                for point in response[0]:
                    if all(point.payload.get(k) == v for k, v in filters.items()):
                        filtered_points.append(point)
                response = (filtered_points, response[1])
        
        # Extract and return documents
        documents = []
        for point in response[0]:
            doc = point.payload
            doc["id"] = point.id
            documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error loading documents from Qdrant: {e}")
        # Fallback to simulated data if there's an error
        st.warning("Using simulated data due to Qdrant connection issue.")
        return generate_simulated_documents(collection_name, limit)

def generate_simulated_documents(collection_name: str, limit: int = 10) -> List[Dict]:
    """Generate simulated documents for testing when Qdrant is unavailable."""
    documents = []
    
    for i in range(limit):
        if collection_name == "citizen_comments":
            # Simulate citizen comments document
            doc = {
                "id": str(uuid.uuid4()),
                "comment_id": str(uuid.uuid4()),
                "project_id": str(uuid.uuid4()),
                "citizen_name": np.random.choice(["Ahmed", "فاطمة", "Mohammed", "Sarah", "John"]),
                "comment_text": f"This is a simulated comment {i+1} about the project.",
                "date_submitted": datetime.now().strftime("%Y-%m-%d"),
                "channel": np.random.choice(["Web", "استمارة", "Mobile", "Email"]),
                "sentiment": np.random.choice(["POS", "NEG", "NEU"], p=[0.4, 0.3, 0.3]),
                "polarity": np.random.uniform(-1, 1),
                "keywords": ["simulated", "test", "demo"],
                "project_title": f"Simulated Project {i+1}",
                "project_themes": np.random.choice([
                    "Infrastructure", 
                    "Education", 
                    "Health", 
                    "Social Services",
                    "تحسين الخدمات الاجتماعية"
                ]),
                "project_status": np.random.choice(["Planned", "In Progress", "Completed"]),
                "votes": {
                    "thumb_up": np.random.randint(0, 50),
                    "thumb_down": np.random.randint(0, 10),
                    "vote_score": np.random.randint(-10, 50)
                },
                "offensive": np.random.choice(["Yes", "No"], p=[0.1, 0.9])
            }
        elif collection_name == "municipal_projects":
            # Simulate municipal projects document
            doc = {
                "id": str(uuid.uuid4()),
                "title": f"Simulated Municipal Project {i+1}",
                "description": f"This is a detailed description of simulated municipal project {i+1}. It includes information about the project goals, timeline, and expected benefits for the community.",
                "status": np.random.choice(["Planned", "In Progress", "Completed"]),
                "location": np.random.choice(["City Center", "North District", "South District", "East District", "West District"]),
                "budget": np.random.randint(10000, 1000000)
            }
        else:
            # Generic document
            doc = {
                "id": str(uuid.uuid4()),
                "title": f"Simulated Document {i+1}",
                "content": f"This is simulated content for document {i+1}."
            }
        
        documents.append(doc)
    
    return documents

def get_collection_stats(collection_name: str) -> Dict[str, Any]:
    """Get statistics about a Qdrant collection."""
    try:
        client = get_qdrant_client()
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            return {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status
            }
        except Exception as e:
            st.warning(f"Couldn't get collection stats: {e}")
            return {"status": "Unavailable or not found", "error": str(e)}
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return {"status": "Connection Error", "error": str(e)}

# -------------------------------
# MONGODB HELPER (for extended functionality)
# -------------------------------
def get_mongo_client():
    connection_string = st.secrets.get("mongodb", {}).get("connection_string", "mongodb://localhost:27017")
    from pymongo import MongoClient
    return MongoClient(connection_string)

# -------------------------------
# MODEL WRAPPER FUNCTIONS (for each task)
# -------------------------------
def get_summary(model: str, input_text: str) -> str:
    """Return a summary using the specified model."""
    if model == "baseline_rule_based":
        sentences = input_text.split('.')
        return sentences[0].strip() + '.' if sentences and sentences[0].strip() else input_text[:100] + '...'
    else:
        prompt = f"Summarize the following text concisely:\n\n{input_text}\n\nSummary:"
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.5,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error using model {model} for summarization: {e}")
            return ""

def get_sentiment(model: str, text: str) -> str:
    """Return predicted sentiment (POS, NEG, or NEU)."""
    if model == "baseline_rule_based":
        t = text.lower()
        pos_words = ["good", "great", "excellent", "ممتاز", "جيد"]
        neg_words = ["bad", "poor", "terrible", "سيء", "رديء"]
        
        for word in pos_words:
            if word in t:
                return "POS"
        for word in neg_words:
            if word in t:
                return "NEG"
        return "NEU"
    else:
        prompt = f"Determine the sentiment of the following text. Answer with one of: POS, NEG, or NEU.\nText: {text}"
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            sentiment = response["choices"][0]["message"]["content"].strip().upper()
            return sentiment if sentiment in ["POS", "NEG", "NEU"] else "NEU"
        except Exception as e:
            print(f"Error in sentiment for model {model}: {e}")
            return "NEU"

def get_offensive(model: str, text: str) -> str:
    """Return 'Yes' if the text contains offensive language; otherwise 'No'."""
    if model == "baseline_rule_based":
        offensive_words = ["badword", "offensive", "inappropriate", "سيء", "مسيء"]
        return "Yes" if any(word in text.lower() for word in offensive_words) else "No"
    else:
        prompt = f"Does the following text contain offensive language? Answer only with 'Yes' or 'No'.\nText: {text}"
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            answer = response["choices"][0]["message"]["content"].strip().capitalize()
            return answer if answer in ["Yes", "No"] else "No"
        except Exception as e:
            print(f"Error in offensive detection for model {model}: {e}")
            return "No"

def get_topic(model: str, text: str) -> str:
    """Return the main topic for the text."""
    if model == "baseline_rule_based":
        topics = {
            "infrastructure": ["infrastructure", "road", "bridge", "بنية تحتية", "طريق", "جسر"],
            "education": ["school", "education", "مدرسة", "تعليم"],
            "health": ["hospital", "health", "مستشفى", "صحة"],
            "social services": ["service", "social", "خدمات", "اجتماعية"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in text.lower() for keyword in keywords):
                return topic
        return "general"
    else:
        prompt = f"Identify the main topic of the following text from these categories: infrastructure, education, health, social services, or general.\nText: {text}\nTopic:"
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
                temperature=0.3,
            )
            return response["choices"][0]["message"]["content"].strip().lower()
        except Exception as e:
            print(f"Error in topic modeling for model {model}: {e}")
            return "general"

def get_keywords(model: str, text: str, num_keywords: int = 3) -> List[str]:
    """Extract keywords from the given text."""
    if model == "baseline_rule_based":
        # Simple frequency-based approach
        words = text.lower().split()
        # Remove common words (stopwords)
        stopwords = ["the", "and", "is", "in", "to", "of", "a", "for", "on", "with", 
                    "من", "في", "على", "إلى", "هو", "هي", "هم"]
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:num_keywords]]
    else:
        prompt = f"Extract {num_keywords} keywords from the following text:\n\n{text}\n\nKeywords (comma-separated):"
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3,
            )
            keywords = response["choices"][0]["message"]["content"].strip().split(',')
            return [k.strip().lower() for k in keywords]
        except Exception as e:
            print(f"Error in keyword extraction for model {model}: {e}")
            return []

def predict_engagement(model: str, text: str) -> int:
    """Predict user engagement score based on comment text."""
    if model == "baseline_rule_based":
        # Simple length and sentiment-based rule
        length_score = min(10, len(text) // 20)  # Up to 10 points for length
        
        # Check for positive/negative words
        sentiment_score = 0
        pos_words = ["good", "great", "excellent", "ممتاز", "جيد"]
        neg_words = ["bad", "poor", "terrible", "سيء", "رديء"]
        
        for word in pos_words:
            if word in text.lower():
                sentiment_score += 3
        for word in neg_words:
            if word in text.lower():
                sentiment_score -= 2
        
        return max(0, min(100, length_score + sentiment_score))
    else:
        prompt = f"""
        Predict the engagement score (0-100) for the following comment. 
        Higher scores indicate comments that are more likely to receive positive reactions:
        
        Comment: {text}
        
        Score (0-100):"""
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3,
            )
            
            # Extract and validate the score
            score_text = response["choices"][0]["message"]["content"].strip()
            try:
                score = int(score_text)
                return max(0, min(100, score))
            except ValueError:
                return 50  # Default if we can't parse the response
        except Exception as e:
            print(f"Error in engagement prediction for model {model}: {e}")
            return 50

def detect_language(model: str, text: str) -> str:
    """Detect the language of the given text."""
    if model == "baseline_rule_based":
        # Simple character-based detection for Arabic vs Latin scripts
        arabic_chars = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
        latin_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        arabic_count = sum(1 for c in text if c in arabic_chars)
        latin_count = sum(1 for c in text if c in latin_chars)
        
        if arabic_count > latin_count:
            return "ar"  # Arabic
        else:
            return "en"  # Default to English
    else:
        prompt = f"Detect the language of the following text. Return only the ISO language code (e.g., 'en' for English, 'ar' for Arabic):\n\n{text}\n\nLanguage code:"
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            return response["choices"][0]["message"]["content"].strip().lower()
        except Exception as e:
            print(f"Error in language detection for model {model}: {e}")
            return "en"  # Default to English on error

def translate_text(model: str, text: str, target_lang: str = "en") -> str:
    """Translate text to the target language."""
    if model == "baseline_rule_based":
        # Very basic translation - just for demonstration purposes
        if target_lang == "en":
            translations = {
                "مرحبا": "hello",
                "شكرا": "thank you",
                "ممتاز": "excellent"
            }
            for arabic, english in translations.items():
                text = text.replace(arabic, english)
            return text
        else:
            return text  # No translation capability in other directions
    else:
        prompt = f"Translate the following text to {target_lang}:\n\n{text}\n\nTranslation:"
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error in translation for model {model}: {e}")
            return text  # Return original text on error

# -------------------------------
# METRIC FUNCTIONS
# -------------------------------
def compute_rouge(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)

def compute_bleu(reference: str, candidate: str) -> float:
    """
    Compute BLEU score using simple whitespace tokenization.
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    smooth_fn = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth_fn)

def compute_keyword_overlap(reference_keywords: List[str], candidate_keywords: List[str]) -> float:
    """
    Calculate the overlap between reference and candidate keywords.
    Returns a value from 0.0 (no overlap) to 1.0 (perfect match).
    """
    if not reference_keywords or not candidate_keywords:
        return 0.0
    
    # Convert to sets and calculate overlap
    ref_set = set(k.lower() for k in reference_keywords)
    cand_set = set(k.lower() for k in candidate_keywords)
    
    # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = len(ref_set.intersection(cand_set))
    union = len(ref_set.union(cand_set))
    
    return intersection / union if union > 0 else 0.0

def compute_engagement_accuracy(reference_score: int, predicted_score: int, tolerance: int = 10) -> float:
    """
    Calculate accuracy of engagement prediction within a tolerance range.
    Returns 1.0 if prediction is within tolerance, otherwise a scaled value.
    """
    error = abs(reference_score - predicted_score)
    if error <= tolerance:
        return 1.0
    else:
        # Linear scaling: decreases as error increases
        max_error = 100  # Maximum possible error
        return max(0.0, 1.0 - (error - tolerance) / (max_error - tolerance))

# -------------------------------
# LOADING EVALUATION DATA FROM QDRANT
# -------------------------------
def load_evaluation_data_for_task(task: str) -> pd.DataFrame:
    """
    Load evaluation data for the given task using the specified Qdrant collection.
    If insufficient samples are found, simulated data will be generated.
    """
    config = EVAL_CONFIG.get(task)
    samples = []
    
    if config:
        collection_name = config["collection"]
        st.write(f"Loading data from collection: {collection_name}")
        
        # Get collection stats
        stats = get_collection_stats(collection_name)
        st.write(f"Collection stats: {stats}")
        
        # Load data from Qdrant
        data = load_qdrant_documents(collection_name, limit=MIN_EXPERIMENTS * 2)
        st.write(f"Found {len(data)} documents in collection")
        
        for doc in data:
            input_key = config["input_key"]
            reference_key = config["reference_key"]
            
            # Handle nested keys with dot notation (e.g., "votes.vote_score")
            input_value = doc.get(input_key, "")
            
            if "." in reference_key:
                # Handle nested reference key
                parts = reference_key.split(".")
                ref_value = doc
                for part in parts:
                    if isinstance(ref_value, dict) and part in ref_value:
                        ref_value = ref_value[part]
                    else:
                        ref_value = None
                        break
            else:
                ref_value = doc.get(reference_key)
            
            if input_value and ref_value is not None:
                sample = {
                    "id": doc.get("id", str(uuid.uuid4())),
                    "input_text": input_value,
                    "reference": ref_value
                }
                
                # Add extra fields for analysis
                if "citizen_name" in doc:
                    sample["citizen_name"] = doc["citizen_name"]
                if "date_submitted" in doc:
                    sample["date_submitted"] = doc["date_submitted"]
                if "channel" in doc:
                    sample["channel"] = doc["channel"]
                if "project_title" in doc:
                    sample["project_title"] = doc["project_title"]
                
                samples.append(sample)
    
    # If insufficient samples, generate simulated data
    if len(samples) < MIN_EXPERIMENTS:
        st.warning(f"Only found {len(samples)} samples for task '{task}'. Using simulated data.")
        for i in range(MIN_EXPERIMENTS - len(samples)):
            if task == "sentiment":
                reference = np.random.choice(["POS", "NEG", "NEU"])
            elif task == "offensive":
                reference = np.random.choice(["Yes", "No"], p=[0.2, 0.8])
            elif task == "topic":
                reference = np.random.choice(["infrastructure", "education", "health", "social services", "general"])
            elif task == "keywords":
                reference = ["keyword1", "keyword2", "keyword3"]
            elif task == "engagement":
                reference = int(np.random.normal(50, 20))  # Mean 50, std 20
            else:
                reference = f"Sample reference for {task} task."
                
            samples.append({
                "id": str(uuid.uuid4()),
                "input_text": f"Sample input text for {task} task number {i+1}.",
                "reference": reference,
                "simulated": True
            })
    
    return pd.DataFrame(samples)

# -------------------------------
# EXPERIMENT FUNCTIONS FOR EACH TASK
# -------------------------------
def run_summarization_experiments(evaluation_data: pd.DataFrame, models: list) -> list:
    results = []
    
    # Create experiment log for detailed analysis
    experiment_log = []
    
    for model in models:
        rouge1_list, rougeL_list, bleu_list, latency_list = [], [], [], []
        
        for idx, row in evaluation_data.iterrows():
            input_text = row["input_text"]
            reference = row["reference"]
            
            start = time.time()
            candidate = get_summary(model, input_text)
            latency = time.time() - start
            latency_list.append(latency)
            
            # Compute metrics
            rouge_scores = compute_rouge(reference, candidate)
            rouge1 = rouge_scores["rouge1"].fmeasure
            rougeL = rouge_scores["rougeL"].fmeasure
            bleu = compute_bleu(reference, candidate)
            
            rouge1_list.append(rouge1)
            rougeL_list.append(rougeL)
            bleu_list.append(bleu)
            
            # Log experiment details
            experiment_log.append({
                "task": "Summarization",
                "model": model,
                "input_id": row.get("id", idx),
                "input_text": input_text,
                "reference": reference,
                "prediction": candidate,
                "rouge1": round(rouge1, 3),
                "rougeL": round(rougeL, 3),
                "bleu": round(bleu, 3),
                "latency": round(latency, 3),
                "simulated": row.get("simulated", False)
            })
        
        # Compile results
        results.append({
            "Model": model,
            "Task": "Summarization",
            "Experiments": len(evaluation_data),
            "Avg ROUGE-1 F1": round(np.mean(rouge1_list), 3),
            "Avg ROUGE-L F1": round(np.mean(rougeL_list), 3),
            "Avg BLEU": round(np.mean(bleu_list), 3),
            "Avg Latency (s)": round(np.mean(latency_list), 3)
        })
    
    # Save experiment log to session state for detailed analysis
    st.session_state["summarization_experiments"] = pd.DataFrame(experiment_log)
    return results

def run_sentiment_experiments(evaluation_data: pd.DataFrame, models: list) -> list:
    results = []
    experiment_log = []
    
    for model in models:
        correct, total, latency_list = 0, 0, []
        
        for idx, row in evaluation_data.iterrows():
            input_text = row["input_text"]
            reference = str(row["reference"]).strip().upper()
            
            start = time.time()
            prediction = get_sentiment(model, input_text)
            latency = time.time() - start
            latency_list.append(latency)
            
            total += 1
            is_correct = prediction == reference
            if is_correct:
                correct += 1
                
            # Log experiment details
            experiment_log.append({
                "task": "Sentiment Analysis",
                "model": model,
                "input_id": row.get("id", idx),
                "input_text": input_text,
                "reference": reference,
                "prediction": prediction,
                "correct": is_correct,
                "latency": round(latency, 3),
                "project_title": row.get("project_title", ""),
                "citizen_name": row.get("citizen_name", ""),
                "simulated": row.get("simulated", False)
            })
        
        accuracy = correct / total if total > 0 else 0
        results.append({
            "Model": model,
            "Task": "Sentiment Analysis",
            "Experiments": total,
            "Accuracy": round(accuracy, 3),
            "Avg Latency (s)": round(np.mean(latency_list), 3)
        })
    
    # Save experiment log
    st.session_state["sentiment_experiments"] = pd.DataFrame(experiment_log)
    return results

def run_offensive_experiments(evaluation_data: pd.DataFrame, models: list) -> list:
    results = []
    experiment_log = []
    
    for model in models:
        correct, total, latency_list = 0, 0, []
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        
        for idx, row in evaluation_data.iterrows():
            input_text = row["input_text"]
            reference = str(row["reference"]).strip().capitalize()  # Expect "Yes" or "No"
            
            start = time.time()
            prediction = get_offensive(model, input_text)
            latency = time.time() - start
            latency_list.append(latency)
            
            total += 1
            is_correct = prediction == reference
            if is_correct:
                correct += 1
                if reference == "Yes":
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if reference == "Yes":
                    false_neg += 1
                else:
                    false_pos += 1
                    
            # Log experiment details
            experiment_log.append({
                "task": "Offensive Language Detection",
                "model": model,
                "input_id": row.get("id", idx),
                "input_text": input_text,
                "reference": reference,
                "prediction": prediction,
                "correct": is_correct,
                "latency": round(latency, 3),
                "project_title": row.get("project_title", ""),
                "simulated": row.get("simulated", False)
            })
        
        accuracy = correct / total if total > 0 else 0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "Model": model,
            "Task": "Offensive Language Detection",
            "Experiments": total,
            "Accuracy": round(accuracy, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1 Score": round(f1, 3),
            "Avg Latency (s)": round(np.mean(latency_list), 3)
        })
    
    # Save experiment log
    st.session_state["offensive_experiments"] = pd.DataFrame(experiment_log)
    return results

def run_topic_experiments(evaluation_data: pd.DataFrame, models: list) -> list:
    results = []
    experiment_log = []
    
    for model in models:
        correct, total, latency_list = 0, 0, []
        
        for idx, row in evaluation_data.iterrows():
            input_text = row["input_text"]
            reference = str(row["reference"]).strip().lower()
            
            start = time.time()
            prediction = get_topic(model, input_text).strip().lower()
            latency = time.time() - start
            latency_list.append(latency)
            
            total += 1
            # Check for exact match or partial match (topic may be part of a longer reference string)
            exact_match = prediction == reference
            partial_match = prediction in reference or reference in prediction
            
            is_correct = exact_match or partial_match
            if is_correct:
                correct += 1
                
            # Log experiment details
            experiment_log.append({
                "task": "Topic Modeling",
                "model": model,
                "input_id": row.get("id", idx),
                "input_text": input_text,
                "reference": reference,
                "prediction": prediction,
                "exact_match": exact_match,
                "partial_match": partial_match,
                "correct": is_correct,
                "latency": round(latency, 3),
                "project_title": row.get("project_title", ""),
                "simulated": row.get("simulated", False)
            })
        
        accuracy = correct / total if total > 0 else 0
        results.append({
            "Model": model,
            "Task": "Topic Modeling",
            "Experiments": total,
            "Accuracy": round(accuracy, 3),
            "Avg Latency (s)": round(np.mean(latency_list), 3)
        })
    
    # Save experiment log
    st.session_state["topic_experiments"] = pd.DataFrame(experiment_log)
    return results

def run_keyword_experiments(evaluation_data: pd.DataFrame, models: list) -> list:
    results = []
    experiment_log = []
    
    for model in models:
        overlap_scores, latency_list = [], []
        
        for idx, row in evaluation_data.iterrows():
            input_text = row["input_text"]
            reference_keywords = row["reference"]
            
            # Ensure reference is a list
            if isinstance(reference_keywords, str):
                try:
                    reference_keywords = json.loads(reference_keywords)
                except:
                    reference_keywords = [reference_keywords]
            elif not isinstance(reference_keywords, list):
                reference_keywords = [str(reference_keywords)]
            
            start = time.time()
            predicted_keywords = get_keywords(model, input_text)
            latency = time.time() - start
            latency_list.append(latency)
            
            # Compute overlap
            overlap = compute_keyword_overlap(reference_keywords, predicted_keywords)
            overlap_scores.append(overlap)
            
            # Log experiment details
            experiment_log.append({
                "task": "Keyword Extraction",
                "model": model,
                "input_id": row.get("id", idx),
                "input_text": input_text,
                "reference": str(reference_keywords),
                "prediction": str(predicted_keywords),
                "overlap_score": round(overlap, 3),
                "latency": round(latency, 3),
                "project_title": row.get("project_title", ""),
                "simulated": row.get("simulated", False)
            })
        
        # Calculate average scores
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
        avg_latency = np.mean(latency_list) if latency_list else 0
        
        results.append({
            "Model": model,
            "Task": "Keyword Extraction",
            "Experiments": len(overlap_scores),
            "Avg Overlap Score": round(avg_overlap, 3),
            "Avg Latency (s)": round(avg_latency, 3)
        })
    
    # Save experiment log
    st.session_state["keyword_experiments"] = pd.DataFrame(experiment_log)
    return results

def run_engagement_experiments(evaluation_data: pd.DataFrame, models: list) -> list:
    results = []
    experiment_log = []
    
    for model in models:
        accuracy_scores, mae_scores, latency_list = [], [], []
        
        for idx, row in evaluation_data.iterrows():
            input_text = row["input_text"]
            reference_score = row["reference"]
            
            # Ensure reference is an integer
            try:
                if isinstance(reference_score, str):
                    if reference_score.isdigit():
                        reference_score = int(reference_score)
                    else:
                        reference_score = 50  # Default value
                elif isinstance(reference_score, (dict, list)):
                    reference_score = 50  # Default if reference is a complex structure
                else:
                    reference_score = int(reference_score)
            except:
                reference_score = 50
            
            start = time.time()
            predicted_score = predict_engagement(model, input_text)
            latency = time.time() - start
            latency_list.append(latency)
            
            # Compute metrics
            accuracy = compute_engagement_accuracy(reference_score, predicted_score)
            mae = abs(reference_score - predicted_score)
            
            accuracy_scores.append(accuracy)
            mae_scores.append(mae)
            
            # Log experiment details
            experiment_log.append({
                "task": "Engagement Prediction",
                "model": model,
                "input_id": row.get("id", idx),
                "input_text": input_text,
                "reference_score": reference_score,
                "predicted_score": predicted_score,
                "accuracy": round(accuracy, 3),
                "mae": mae,
                "latency": round(latency, 3),
                "project_title": row.get("project_title", ""),
                "simulated": row.get("simulated", False)
            })
        
        # Calculate average scores
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
        avg_mae = np.mean(mae_scores) if mae_scores else 0
        avg_latency = np.mean(latency_list) if latency_list else 0
        
        results.append({
            "Model": model,
            "Task": "Engagement Prediction",
            "Experiments": len(accuracy_scores),
            "Avg Accuracy": round(avg_accuracy, 3),
            "Avg MAE": round(avg_mae, 1),
            "Avg Latency (s)": round(avg_latency, 3)
        })
    
    # Save experiment log
    st.session_state["engagement_experiments"] = pd.DataFrame(experiment_log)
    return results

def run_language_detection_experiments(evaluation_data: pd.DataFrame, models: list) -> list:
    """Run experiments for language detection task"""
    results = []
    experiment_log = []
    
    for model in models:
        correct, total, latency_list = 0, 0, []
        
        for idx, row in evaluation_data.iterrows():
            input_text = row["input_text"]
            
            # For language detection, we'll manually check a few languages based on known patterns
            # In a real implementation, this reference would come from labeled data
            if any(c in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي" for c in input_text):
                reference_lang = "ar"  # Arabic
            elif any(c in "éèêëàâäôöùûüÿç" for c in input_text):
                reference_lang = "fr"  # French
            else:
                reference_lang = "en"  # Default to English
                
            start = time.time()
            predicted_lang = detect_language(model, input_text)
            latency = time.time() - start
            latency_list.append(latency)
            
            total += 1
            is_correct = predicted_lang == reference_lang
            if is_correct:
                correct += 1
                
            # Log experiment details
            experiment_log.append({
                "task": "Language Detection",
                "model": model,
                "input_id": row.get("id", idx),
                "input_text": input_text,
                "reference_lang": reference_lang,
                "predicted_lang": predicted_lang,
                "correct": is_correct,
                "latency": round(latency, 3),
                "citizen_name": row.get("citizen_name", ""),
                "simulated": row.get("simulated", False)
            })
        
        accuracy = correct / total if total > 0 else 0
        results.append({
            "Model": model,
            "Task": "Language Detection",
            "Experiments": total,
            "Accuracy": round(accuracy, 3),
            "Avg Latency (s)": round(np.mean(latency_list), 3)
        })
    
    # Save experiment log
    st.session_state["language_experiments"] = pd.DataFrame(experiment_log)
    return results

def run_translation_experiments(evaluation_data: pd.DataFrame, models: list) -> list:
    """Run experiments for translation quality"""
    results = []
    experiment_log = []
    
    # For translation experiments, we need a reference translation
    # In a real setup, this would come from parallel corpora
    
    for model in models:
        bleu_scores, latency_list = [], []
        
        for idx, row in evaluation_data.iterrows():
            input_text = row["input_text"]
            
            # Detect source language
            source_lang = detect_language(model, input_text)
            
            # Skip if already in English
            if source_lang == "en":
                continue
                
            # Translate to English
            start = time.time()
            translation = translate_text(model, input_text, "en")
            latency = time.time() - start
            latency_list.append(latency)
            
            # For evaluation purposes, we'd need a reference translation
            # Since we don't have one, we'll use a second model as reference (not ideal)
            if model != "gpt-4" and "gpt-4" in models:
                reference_translation = translate_text("gpt-4", input_text, "en")
            else:
                # Fallback reference - not ideal but for demonstration
                reference_translation = "This is a placeholder reference translation."
                
            # Compute BLEU score 
            bleu = compute_bleu(reference_translation, translation)
            bleu_scores.append(bleu)
                
            # Log experiment details
            experiment_log.append({
                "task": "Translation",
                "model": model,
                "input_id": row.get("id", idx),
                "input_text": input_text,
                "source_lang": source_lang,
                "translation": translation,
                "reference_translation": reference_translation,
                "bleu": round(bleu, 3),
                "latency": round(latency, 3),
                "citizen_name": row.get("citizen_name", ""),
                "simulated": row.get("simulated", False)
            })
        
        if bleu_scores:
            results.append({
                "Model": model,
                "Task": "Translation (to English)",
                "Experiments": len(bleu_scores),
                "Avg BLEU": round(np.mean(bleu_scores), 3),
                "Avg Latency (s)": round(np.mean(latency_list), 3)
            })
    
    # Save experiment log
    st.session_state["translation_experiments"] = pd.DataFrame(experiment_log)
    return results

# -------------------------------
# ANALYSIS & VISUALIZATION FUNCTIONS
# -------------------------------
def analyze_citizen_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze citizen comments collection to extract insights."""
    # Basic statistics
    stats = {
        "Total Comments": len(df),
        "Unique Citizens": len(df["citizen_name"].unique()) if "citizen_name" in df.columns else "N/A",
        "Positive Comments": sum(df["sentiment"] == "POS") if "sentiment" in df.columns else "N/A",
        "Negative Comments": sum(df["sentiment"] == "NEG") if "sentiment" in df.columns else "N/A",
        "Neutral Comments": sum(df["sentiment"] == "NEU") if "sentiment" in df.columns else "N/A"
    }
    
    # Analyze by channel if available
    if "channel" in df.columns:
        channel_counts = df["channel"].value_counts().to_dict()
        stats.update({f"Channel: {k}": v for k, v in channel_counts.items()})
    
    # Analyze by project if available
    if "project_title" in df.columns:
        project_counts = df["project_title"].value_counts().head(5).to_dict()
        stats.update({f"Top Project: {k[:20]}...": v for k, v in project_counts.items()})
    
    # Convert to DataFrame for display
    return pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])

def create_visualizations(results_df: pd.DataFrame):
    """Create advanced visualizations based on experiment results."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Group results by task
        tasks = results_df["Task"].unique()
        
        # Create tabs for different visualization groups
        tabs = st.tabs(["Performance by Task", "Model Comparison", "Latency Analysis", "Error Analysis"])
        
        with tabs[0]:  # Performance by Task
            st.subheader("Performance Metrics by Task")
            
            for task in tasks:
                task_df = results_df[results_df["Task"] == task]
                
                # Choose appropriate metrics based on task
                if task == "Summarization":
                    metrics = ["Avg ROUGE-1 F1", "Avg ROUGE-L F1", "Avg BLEU"]
                    title = "Summarization Metrics"
                elif task in ["Sentiment Analysis", "Offensive Language Detection", "Topic Modeling"]:
                    metrics = ["Accuracy"]
                    if "Precision" in task_df.columns and "Recall" in task_df.columns:
                        metrics.extend(["Precision", "Recall", "F1 Score"])
                    title = f"{task} Accuracy"
                elif task == "Keyword Extraction":
                    metrics = ["Avg Overlap Score"]
                    title = "Keyword Extraction Performance"
                elif task == "Engagement Prediction":
                    metrics = ["Avg Accuracy", "Avg MAE"]
                    title = "Engagement Prediction Performance"
                elif task == "Translation (to English)":
                    metrics = ["Avg BLEU"]
                    title = "Translation Quality (BLEU)"
                else:
                    metrics = [col for col in task_df.columns if "Avg" in col and col != "Avg Latency (s)"]
                    title = f"{task} Performance"
                
                if metrics:
                    melted_df = task_df.melt(id_vars=["Model"], value_vars=metrics, 
                                        var_name="Metric", value_name="Score")
                    
                    fig = px.bar(melted_df, x="Model", y="Score", color="Metric", 
                                barmode="group", title=title)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:  # Model Comparison
            st.subheader("Model Comparison Across Tasks")
            
            # Create a radar chart for each model
            for model in results_df["Model"].unique():
                model_df = results_df[results_df["Model"] == model]
                
                # Extract key performance metric for each task
                performance_data = []
                for _, row in model_df.iterrows():
                    task = row["Task"]
                    if "Accuracy" in row:
                        perf = row["Accuracy"]
                    elif "Avg ROUGE-1 F1" in row:
                        perf = row["Avg ROUGE-1 F1"]
                    elif "Avg Overlap Score" in row:
                        perf = row["Avg Overlap Score"]
                    elif "Avg BLEU" in row:
                        perf = row["Avg BLEU"]
                    else:
                        perf = 0.5  # Default
                    
                    performance_data.append((task, perf))
                
                if performance_data:
                    # Sort by task name
                    performance_data.sort(key=lambda x: x[0])
                    tasks, scores = zip(*performance_data)
                    
                    # Create radar chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=tasks,
                        fill='toself',
                        name=model
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        title=f"Performance Profile: {model}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:  # Latency Analysis
            st.subheader("Latency Analysis")
            
            # Bar chart of latency by model and task
            latency_fig = px.bar(
                results_df, 
                x="Model", 
                y="Avg Latency (s)", 
                color="Task",
                barmode="group", 
                title="Average Latency by Model and Task"
            )
            st.plotly_chart(latency_fig, use_container_width=True)
            
            # Latency vs Performance scatter plot
            performance_cols = [col for col in results_df.columns 
                               if any(metric in col for metric in ["Accuracy", "ROUGE", "BLEU", "F1", "Overlap"])]
            
            if performance_cols:
                # Choose first performance metric as example
                perf_col = performance_cols[0]
                
                scatter_fig = px.scatter(
                    results_df, 
                    x="Avg Latency (s)", 
                    y=perf_col,
                    color="Task", 
                    symbol="Model",
                    title=f"Latency vs {perf_col}",
                    labels={"Avg Latency (s)": "Average Latency (seconds)"}
                )
                scatter_fig.update_traces(marker=dict(size=12))
                st.plotly_chart(scatter_fig, use_container_width=True)
        
        with tabs[3]:  # Error Analysis
            st.subheader("Error Analysis")
            
            # If we have experiment logs in session state
            for task_key in ["sentiment_experiments", "offensive_experiments", "topic_experiments"]:
                if task_key in st.session_state:
                    task_df = st.session_state[task_key]
                    
                    if "correct" in task_df.columns:
                        # Show error rate by model
                        error_rates = task_df.groupby("model")["correct"].mean().reset_index()
                        error_rates["error_rate"] = 1 - error_rates["correct"]
                        error_rates = error_rates.sort_values("error_rate")
                        
                        # Bar chart of error rates
                        error_fig = px.bar(
                            error_rates,
                            x="model",
                            y="error_rate",
                            title=f"Error Rate by Model for {task_key.split('_')[0].capitalize()}",
                            labels={"model": "Model", "error_rate": "Error Rate"}
                        )
                        st.plotly_chart(error_fig, use_container_width=True)
                        
                        # Show examples of errors
                        errors_df = task_df[task_df["correct"] == False].head(5)
                        if not errors_df.empty:
                            st.write(f"Examples of errors for {task_key.split('_')[0]}:")
                            st.dataframe(errors_df[["model", "input_text", "reference", "prediction"]])
                    
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")

# -------------------------------
# MAIN EVALUATION PIPELINE
# -------------------------------
def main():

    
    st.title("📊 Comprehensive LLM Evaluation Dashboard")
    st.write("""
    This dashboard evaluates multiple language models across various NLP tasks using data from Qdrant collections.
    Tasks include summarization, sentiment analysis, offensive language detection, topic modeling, and more.
    """)
    
    # Check Qdrant connection
    try:
        client = get_qdrant_client()
        st.success("✓ Connected to Qdrant server")
    except Exception as e:
        st.error(f"⚠️ Failed to connect to Qdrant: {e}")
        st.warning("The evaluation will use simulated data instead.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    selected_tasks = st.sidebar.multiselect(
        "Select Tasks to Evaluate",
        ["summarization", "sentiment", "offensive", "topic", "keywords", "engagement", "language", "translation"],
        default=["sentiment", "offensive", "topic"]
    )
    
    selected_models = st.sidebar.multiselect(
        "Select Models to Evaluate",
        MODELS,
        default=["gpt-3.5-turbo", "gpt-4", "baseline_rule_based"]
    )
    
    run_button = st.sidebar.button("Run Evaluation", type="primary")
    
    # Show collection stats
    st.sidebar.header("Qdrant Collection Stats")
    for task in selected_tasks:
        if task in EVAL_CONFIG:
            collection = EVAL_CONFIG[task]["collection"]
            stats = get_collection_stats(collection)
            st.sidebar.write(f"{collection}: {stats}")
    
    # Collection analysis
    st.header("📋 Collection Analysis")
    
    # Analyze citizen_comments collection
    if "citizen_comments" in [EVAL_CONFIG.get(task, {}).get("collection") for task in selected_tasks]:
        st.subheader("Citizen Comments Analysis")
        
        # Load sample data for analysis
        sample_data = load_qdrant_documents("citizen_comments", limit=100)
        if sample_data:
            comments_df = pd.DataFrame(sample_data)
            
            # Display analysis
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_df = analyze_citizen_comments(comments_df)
                st.table(analysis_df)
            
            with col2:
                try:
                    import plotly.express as px
                    
                    # Sentiment distribution
                    if "sentiment" in comments_df.columns:
                        sentiment_counts = comments_df["sentiment"].value_counts().reset_index()
                        sentiment_counts.columns = ["Sentiment", "Count"]
                        
                        fig = px.pie(sentiment_counts, values="Count", names="Sentiment", 
                                    title="Comment Sentiment Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error in visualization: {e}")
            
            # Show sample data
            with st.expander("Sample comments"):
                st.dataframe(comments_df.head(5))
    
    # Run evaluations when button is clicked
    if run_button:
        st.header("🧪 Evaluation Results")
        
        all_results = []
        
        with st.spinner("Running evaluations..."):
            # Initialize progress bar
            total_steps = len(selected_tasks) * len(selected_models)
            progress_bar = st.progress(0)
            progress_count = 0
            
            for task in selected_tasks:
                st.subheader(f"Task: {task.capitalize()}")
                
                # Load data
                eval_data = load_evaluation_data_for_task(task)
                st.write(f"Loaded {len(eval_data)} evaluation samples")
                
                # Show sample
                with st.expander("Data sample"):
                    st.dataframe(eval_data.head(3))
                
                # Run appropriate experiments based on task
                if task == "summarization":
                    results = run_summarization_experiments(eval_data, selected_models)
                elif task == "sentiment":
                    results = run_sentiment_experiments(eval_data, selected_models)
                elif task == "offensive":
                    results = run_offensive_experiments(eval_data, selected_models)
                elif task == "topic":
                    results = run_topic_experiments(eval_data, selected_models)
                elif task == "keywords":
                    results = run_keyword_experiments(eval_data, selected_models)
                elif task == "engagement":
                    results = run_engagement_experiments(eval_data, selected_models)
                elif task == "language":
                    results = run_language_detection_experiments(eval_data, selected_models)
                elif task == "translation":
                    results = run_translation_experiments(eval_data, selected_models)
                else:
                    results = []
                
                # Update progress
                progress_count += len(selected_models)
                progress_bar.progress(min(1.0, progress_count / total_steps))
                
                # Display task results
                if results:
                    all_results.extend(results)
                    st.dataframe(pd.DataFrame(results))
                else:
                    st.warning(f"No results for task: {task}")
                
                st.markdown("---")
        
        # Show combined results
        st.header("📈 Combined Evaluation Results")
        combined_df = pd.DataFrame(all_results)
        st.dataframe(combined_df)
        
        # Generate visualizations
        st.header("📊 Visualizations")
        create_visualizations(combined_df)
        
        # Export options
        st.header("💾 Export Results")
        csv = combined_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="llm_evaluation_results.csv",
            mime="text/csv",
            key="download-csv"
        )
        
        # Save to session state
        st.session_state["evaluation_results"] = combined_df

# Run the main function
if __name__ == "__main__":
    main()