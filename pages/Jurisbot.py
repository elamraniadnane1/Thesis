import streamlit as st
import os
import json
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import arabic_reshaper
from bidi.algorithm import get_display
import logging
import qdrant_client
from qdrant_client.http import models
import openai
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
# Add this at the beginning of your main.py file to handle OpenAI version compatibility

import importlib.metadata
import re

# Check OpenAI version and warn if incompatible
def check_openai_version():
    try:
        openai_version = importlib.metadata.version("openai")
        version_number = float(re.findall(r'^\d+\.\d+', openai_version)[0])
        
        if version_number >= 1.0:
            logger.info(f"Using OpenAI API v{openai_version} (new client)")
            return "new"
        else:
            logger.warning(f"Using OpenAI API v{openai_version} (legacy client)")
            return "legacy"
    except Exception as e:
        logger.warning(f"Could not determine OpenAI version: {e}")
        return "unknown"

# Class factory to get the right GPTInterface for the installed OpenAI version
def get_gpt_interface_class():
    openai_version_type = check_openai_version()
    
    if openai_version_type == "legacy":
        # Legacy OpenAI API (pre-1.0)
        class LegacyGPTInterface:
            def __init__(self, api_key, model=Config.GPT_MODEL):
                openai.api_key = api_key
                self.model = model
                
            def generate_completion(self, prompt, max_tokens=500, temperature=0.7):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Error in GPT API call: {e}")
                    time.sleep(5)  # Rate limit handling
                    return None
            
            def generate_embedding(self, text):
                try:
                    response = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=text
                    )
                    return response['data'][0]['embedding']
                except Exception as e:
                    logger.error(f"Error in embedding generation: {e}")
                    time.sleep(5)  # Rate limit handling
                    return None
        
        return LegacyGPTInterface
    else:
        # Modern OpenAI API (v1.0+)
        class ModernGPTInterface:
            def __init__(self, api_key, model=Config.GPT_MODEL):
                self.client = openai.OpenAI(api_key=api_key)
                self.model = model
                
            def generate_completion(self, prompt, max_tokens=500, temperature=0.7):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Error in GPT API call: {e}")
                    time.sleep(5)  # Rate limit handling
                    return None
            
            def generate_embedding(self, text):
                try:
                    response = self.client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=text
                    )
                    return response.data[0].embedding
                except Exception as e:
                    logger.error(f"Error in embedding generation: {e}")
                    time.sleep(5)  # Rate limit handling
                    return None
        
        return ModernGPTInterface

# Then replace the GPTInterface class initialization in your AugmentationPipeline __init__ method:
# self.gpt = get_gpt_interface_class()(api_key=config.OPENAI_API_KEY, model=config.GPT_MODEL)
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_augmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    # OpenAI API configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
    GPT_MODEL = "gpt-4"  # Can be changed to gpt-3.5-turbo
    
    # Qdrant configuration
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    
    # Collections to augment
    COLLECTIONS = [
        "citizen_comments",
        "citizen_ideas",
        "hespress_politics_comments",
        "hespress_politics_details",
        "municipal_projects",
        "remacto_comments",
        "remacto_projects"
    ]
    
    # New collections naming pattern
    NEW_COLLECTION_PREFIX = "augmented_"
    
    # Augmentation parameters
    AUGMENTATION_FACTOR = 3  # How many times to augment each document
    BATCH_SIZE = 50  # Batch size for processing
    MAX_WORKERS = 5  # Max parallel workers
    
    # Prompts for different augmentation strategies
    PROMPTS = {
        "paraphrase": """
        أعد صياغة النص التالي بأسلوب مختلف مع الحفاظ على المعنى الأصلي. اجعل الصياغة الجديدة طبيعية ومتنوعة:
        
        النص: {text}
        
        الصياغة الجديدة:
        """,
        
        "add_details": """
        أضف تفاصيل وتوضيحات إلى النص التالي مع الحفاظ على المعنى الأصلي والمحتوى الرئيسي. قم بتوسيع النص بشكل معقول:
        
        النص الأصلي: {text}
        
        النص الموسع:
        """,
        
        "change_perspective": """
        أعد كتابة النص التالي من وجهة نظر مختلفة (مثلاً: من وجهة نظر مواطن عادي، أو مسؤول حكومي، أو خبير) مع الحفاظ على المعنى الأصلي:
        
        النص الأصلي: {text}
        
        النص من وجهة نظر جديدة:
        """,
        
        "sentiment_analysis": """
        قم بتحليل المشاعر في النص التالي وتصنيفه إلى إيجابي (POS)، سلبي (NEG)، أو محايد (NEU).
        قم أيضًا بتحديد درجة قوة المشاعر على مقياس من -1 (سلبي للغاية) إلى 1 (إيجابي للغاية).
        
        النص: {text}
        
        التحليل:
        التصنيف: 
        الدرجة: 
        """,
        
        "topic_modeling": """
        قم بتحديد الموضوع الرئيسي للنص التالي واختر تصنيفًا من القائمة:
        [البيئة، الأمن، البنية التحتية، الخدمات الاجتماعية، الشفافية، النقل، الرقمنة، التنمية الاقتصادية، التنمية الاجتماعية، النظافة]
        
        النص: {text}
        
        الموضوع الرئيسي:
        """,
        
        "extract_keywords": """
        استخرج 3-5 كلمات مفتاحية من النص التالي:
        
        النص: {text}
        
        الكلمات المفتاحية: 
        """,
        
        "translate_back": """
        First, translate this Arabic text to English:
        
        {text}
        
        Now, translate the English version back to Arabic with your own words:
        """
    }
    
    # Custom user prompts (can be modified at runtime)
    USER_PROMPTS = {}

# Class for GPT API interactions
# Updated GPTInterface class compatible with OpenAI's newer API (v1.0.0+)
logger = logging.getLogger(__name__)

import openai
import time
import logging

logger = logging.getLogger(__name__)

class GPTInterface:
    def __init__(self, api_key, model=Config.GPT_MODEL):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def generate_completion(self, prompt, max_tokens=500, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in GPT API call: {e}")
            time.sleep(5)  # Rate limit handling
            return None
    
    def generate_embedding(self, text):
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            time.sleep(5)  # Rate limit handling
            return None

# Class for data augmentation strategies
class DataAugmenter:
    def __init__(self, gpt_interface):
        self.gpt = gpt_interface
        self.strategies = {
            "paraphrase": self.paraphrase,
            "add_details": self.add_details,
            "change_perspective": self.change_perspective,
            "translate_back": self.translate_back,
            "rule_based": self.rule_based_augmentation
        }
        self.analysis_strategies = {
            "sentiment_analysis": self.analyze_sentiment,
            "topic_modeling": self.analyze_topic,
            "extract_keywords": self.extract_keywords
        }
    
    def paraphrase(self, text, prompt_template=None):
        if not prompt_template:
            prompt_template = Config.PROMPTS["paraphrase"]
        prompt = prompt_template.format(text=text)
        return self.gpt.generate_completion(prompt)
    
    def add_details(self, text, prompt_template=None):
        if not prompt_template:
            prompt_template = Config.PROMPTS["add_details"]
        prompt = prompt_template.format(text=text)
        return self.gpt.generate_completion(prompt)
    
    def change_perspective(self, text, prompt_template=None):
        if not prompt_template:
            prompt_template = Config.PROMPTS["change_perspective"]
        prompt = prompt_template.format(text=text)
        return self.gpt.generate_completion(prompt)
    
    def translate_back(self, text, prompt_template=None):
        if not prompt_template:
            prompt_template = Config.PROMPTS["translate_back"]
        prompt = prompt_template.format(text=text)
        return self.gpt.generate_completion(prompt)
    
    def rule_based_augmentation(self, text):
        """Simple rule-based augmentation without using API"""
        words = text.split()
        if len(words) < 3:
            return text
            
        # Random operations
        op = random.choice(["swap", "remove", "add"])
        
        if op == "swap" and len(words) > 3:
            idx1, idx2 = sorted(random.sample(range(len(words)), 2))
            words[idx1], words[idx2] = words[idx2], words[idx1]
        elif op == "remove" and len(words) > 3:
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
        elif op == "add":
            connectors = ["و", "ثم", "لكن", "أيضا", "بالإضافة"]
            words.insert(random.randint(0, len(words)), random.choice(connectors))
            
        return " ".join(words)
    
    def analyze_sentiment(self, text, prompt_template=None):
        if not prompt_template:
            prompt_template = Config.PROMPTS["sentiment_analysis"]
        prompt = prompt_template.format(text=text)
        response = self.gpt.generate_completion(prompt)
        
        # Default values
        sentiment = "NEU"
        polarity = 0.0
        
        # Only proceed if we got a valid response
        if response:
            try:
                if "التصنيف:" in response:
                    sentiment_line = [line for line in response.split('\n') if "التصنيف:" in line][0]
                    if "POS" in sentiment_line or "إيجابي" in sentiment_line:
                        sentiment = "POS"
                    elif "NEG" in sentiment_line or "سلبي" in sentiment_line:
                        sentiment = "NEG"
                
                if "الدرجة:" in response:
                    polarity_line = [line for line in response.split('\n') if "الدرجة:" in line][0]
                    polarity_match = re.search(r'-?\d+\.?\d*', polarity_line)
                    if polarity_match:
                        polarity = float(polarity_match.group())
            except Exception as e:
                logger.error(f"Error parsing sentiment analysis: {e}")
        else:
            logger.warning("No response received from GPT for sentiment analysis")
        
        return {"sentiment": sentiment, "polarity": polarity}
    
    def analyze_topic(self, text, prompt_template=None):
        if not prompt_template:
            prompt_template = Config.PROMPTS["topic_modeling"]
        prompt = prompt_template.format(text=text)
        response = self.gpt.generate_completion(prompt)
        
        # List of predefined topics
        topics = ["البيئة", "الأمن", "البنية التحتية", "الخدمات الاجتماعية", 
                 "الشفافية", "النقل", "الرقمنة", "التنمية الاقتصادية", 
                 "التنمية الاجتماعية", "النظافة"]
        
        # Find the topic in the response
        topic = "البيئة"  # Default
        for t in topics:
            if t in response:
                topic = t
                break
        
        return {"topic": topic}
    
    def extract_keywords(self, text, prompt_template=None):
        if not prompt_template:
            prompt_template = Config.PROMPTS["extract_keywords"]
        prompt = prompt_template.format(text=text)
        response = self.gpt.generate_completion(prompt)
        
        keywords = []
        if "الكلمات المفتاحية:" in response:
            keyword_text = response.split("الكلمات المفتاحية:")[-1].strip()
            keywords = [kw.strip() for kw in re.split(r'[،,\n]', keyword_text) if kw.strip()]
        else:
            # Fallback:
            # If no keywords format is found, try to extract Arabic words
            arabic_words = re.findall(r'[\u0600-\u06FF]+', response)
            keywords = arabic_words[:5]  # Limit to max 5 keywords
        
        return {"keywords": keywords}
    
    def augment_document(self, document, text_fields, strategy_name="paraphrase", prompt_template=None):
        """Augment a document using the specified strategy"""
        augmented_doc = document.copy()
        
        for field in text_fields:
            if field in document and document[field]:
                if strategy_name in self.strategies:
                    strategy_func = self.strategies[strategy_name]
                    augmented_text = strategy_func(document[field], prompt_template)
                    if augmented_text:
                        augmented_doc[field] = augmented_text
        
        return augmented_doc
    
    def analyze_document(self, document, text_fields):
        """Analyze document for sentiment, topics, and keywords"""
        analysis_results = {}
        
        # Combine all text fields for analysis
        combined_text = " ".join([document.get(field, "") for field in text_fields])
        
        # Perform analysis
        for strategy_name, strategy_func in self.analysis_strategies.items():
            analysis_results.update(strategy_func(combined_text))
        
        return analysis_results

# Class for QDrant database operations
# Fix for the QdrantHandler class - the get_points method needs updating

class QdrantHandler:
    def __init__(self, host, port):
        self.client = qdrant_client.QdrantClient(host=host, port=port)
        
    def get_collection_info(self, collection_name):
        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
            return collection_info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def list_collections(self):
        try:
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def create_collection(self, collection_name, vector_size, distance="Cosine"):
        try:
            # Check if collection exists and delete it if it does
            if self.client.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
            
            # Create new collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def get_points(self, collection_name, limit=100, offset=0):
        try:
            response = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset
            )
            # The response is a tuple with (points, next_offset)
            if isinstance(response, tuple) and len(response) > 0:
                return response[0]  # Return just the points
            return []
        except Exception as e:
            logger.error(f"Error getting points: {e}")
            return []
    
    def count_points(self, collection_name):
        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error counting points: {e}")
            return 0
    
    def insert_points(self, collection_name, points):
        try:
            if not points:  # Check if points list is empty
                return False
                
            # Try inserting in smaller batches if the list is large
            if len(points) > 20:
                # Insert in batches of 20
                success = True
                for i in range(0, len(points), 20):
                    batch = points[i:i+20]
                    batch_success = self.insert_points(collection_name, batch)
                    success = success and batch_success
                return success
                
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"Error inserting points: {e}")
            # Try individual point insert if batch fails
            if len(points) > 1:
                logger.info("Trying individual point insertion")
                success = True
                for point in points:
                    point_success = self.insert_points(collection_name, [point])
                    success = success and point_success
                return success
            return False

# Class for data augmentation pipeline
class AugmentationPipeline:
    def __init__(self, config):
        self.config = config
        self.gpt = GPTInterface(api_key=config.OPENAI_API_KEY, model=config.GPT_MODEL)
        self.augmenter = DataAugmenter(self.gpt)
        self.qdrant = QdrantHandler(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.stats = {}
        
    def get_text_fields(self, collection_name):
        """Get text fields for each collection type"""
        if "citizen_comments" in collection_name:
            return ["comment_text"]
        elif "citizen_ideas" in collection_name:
            return ["challenge", "solution"]
        elif "hespress_politics_comments" in collection_name:
            return ["comment_text"]
        elif "hespress_politics_details" in collection_name:
            return ["title", "content"]
        elif "municipal_projects" in collection_name:
            return ["title", "themes", "impact"]
        elif "remacto_comments" in collection_name:
            return ["comment_text"]
        elif "remacto_projects" in collection_name:
            return ["title", "description"]
        else:
            # Default - try to guess from the first point
            try:
                points = self.qdrant.get_points(collection_name, limit=1)
                if points:
                    # Look for common text field names
                    payload = points[0].payload
                    potential_fields = [
                        k for k, v in payload.items() 
                        if isinstance(v, str) and len(v) > 10
                    ]
                    return potential_fields
            except:
                pass
            return []
    
    def process_collection(self, collection_name, augmentation_strategies=None, progress_bar=None):
        """Process a single collection"""
        if augmentation_strategies is None:
            augmentation_strategies = ["paraphrase", "add_details", "change_perspective"]
        
        logger.info(f"Processing collection: {collection_name}")
        
        # Get collection info
        collection_info = self.qdrant.get_collection_info(collection_name)
        if not collection_info:
            logger.error(f"Collection {collection_name} not found")
            return False
        
        # Get vector size
        vector_size = collection_info.config.params.vectors.size
        
        # Create new collection
        new_collection_name = f"{self.config.NEW_COLLECTION_PREFIX}{collection_name}"
        if not self.qdrant.create_collection(new_collection_name, vector_size):
            logger.error(f"Failed to create new collection: {new_collection_name}")
            return False
        
        # Get text fields for this collection
        text_fields = self.get_text_fields(collection_name)
        if not text_fields:
            logger.error(f"No text fields found for collection: {collection_name}")
            return False
        
        # Initialize stats for this collection
        self.stats[collection_name] = {
            "original_count": 0,
            "augmented_count": 0,
            "strategies": {strategy: 0 for strategy in augmentation_strategies},
            "sentiment_distribution": {"POS": 0, "NEG": 0, "NEU": 0},
            "topic_distribution": {},
            "keyword_frequency": {}
        }
        
        # Process in batches
        total_points = self.qdrant.count_points(collection_name)
        self.stats[collection_name]["original_count"] = total_points
        
        # In the process_collection method, there's an indentation error
        # Fix the indentation for the ThreadPoolExecutor section:

        for offset in range(0, total_points, self.config.BATCH_SIZE):
            points = self.qdrant.get_points(
                collection_name, 
                limit=self.config.BATCH_SIZE, 
                offset=offset
            )
            
            if not points:
                continue
            
            # Process points in parallel - fix the indentation here
            with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                futures = []
                for point in points:
                    for i in range(self.config.AUGMENTATION_FACTOR):
                        strategy = random.choice(augmentation_strategies)
                        futures.append(
                            executor.submit(
                                self.augment_point, 
                                point, 
                                text_fields, 
                                strategy,
                                collection_name
                            )
                        )
                    
                # Collect results
                augmented_points = []
                for future in futures:
                    result = future.result()
                    if result:
                        augmented_points.append(result)
                        # Save periodically to avoid data loss
                        if len(augmented_points) >= 10:
                            success = self.qdrant.insert_points(new_collection_name, augmented_points)
                            if success:
                                self.stats[collection_name]["augmented_count"] += len(augmented_points)
                                augmented_points = []
            # Insert augmented points to new collection
            if augmented_points:
                self.qdrant.insert_points(new_collection_name, augmented_points)
                self.stats[collection_name]["augmented_count"] += len(augmented_points)
            
            # Update progress bar if provided
            if progress_bar:
                progress_percentage = min(offset + self.config.BATCH_SIZE, total_points) / total_points
                progress_bar.progress(progress_percentage)
            
            # Log progress
            logger.info(f"Processed {min(offset + self.config.BATCH_SIZE, total_points)}/{total_points} points")
        
        return True
    
    def augment_point(self, point, text_fields, strategy, collection_name):
        """Augment a single point"""
        try:
            # Extract payload and ID
            payload = point.payload
            point_id = point.id
            
            # Try the selected strategy first
            augmented_payload = self.augmenter.augment_document(
                payload, 
                text_fields, 
                strategy,
                Config.USER_PROMPTS.get(strategy, None)
            )
            
            # If that fails, use rule-based as fallback
            if not augmented_payload:
                logger.warning(f"Primary augmentation failed - using rule-based fallback for point {point_id}")
                augmented_payload = payload.copy()
                for field in text_fields:
                    if field in payload and payload[field]:
                        augmented_payload[field] = self.augmenter.rule_based_augmentation(payload[field])
                
                # Add metadata about augmentation
                augmented_payload["_augmentation_info"] = {
                    "original_id": str(point_id),
                    "strategy": strategy,
                    "timestamp": time.time()
                }
                
            # Update statistics
            self.stats[collection_name]["strategies"][strategy] += 1
            
            # Analyze the augmented document
            analysis_results = self.augmenter.analyze_document(augmented_payload, text_fields)
            
            # Update payload with analysis results
            for key, value in analysis_results.items():
                if key == "sentiment":
                    augmented_payload["sentiment"] = value
                    self.stats[collection_name]["sentiment_distribution"][value] += 1
                elif key == "polarity":
                    augmented_payload["polarity"] = value
                elif key == "topic":
                    augmented_payload["topic"] = value
                    self.stats[collection_name]["topic_distribution"][value] = \
                        self.stats[collection_name]["topic_distribution"].get(value, 0) + 1
                elif key == "keywords":
                    augmented_payload["keywords"] = value
                    for kw in value:
                        self.stats[collection_name]["keyword_frequency"][kw] = \
                            self.stats[collection_name]["keyword_frequency"].get(kw, 0) + 1
            
            # Generate new vector embedding
            combined_text = " ".join([augmented_payload.get(field, "") for field in text_fields])
            vector = self.gpt.generate_embedding(combined_text)
            
            if not vector:
                return None
            
            # Create new point
            # Create new point
            new_point = models.PointStruct(
                # Use a numeric ID instead of a string ID
                id=random.randint(100000000, 999999999),  # Simple numeric ID
                vector=vector,
                payload=augmented_payload
            )
                        
            return new_point
            
        except Exception as e:
            logger.error(f"Error augmenting point: {e}")
            return None
    
    def run_pipeline(self, collections=None, augmentation_strategies=None, progress_bar=None):
        """Run the entire pipeline"""
        if collections is None:
            collections = self.config.COLLECTIONS
        
        logger.info(f"Starting augmentation for collections: {collections}")
        
        for collection in collections:
            self.process_collection(collection, augmentation_strategies, progress_bar)
        
        # Generate summary report
        logger.info("Augmentation completed. Generating report...")
        return self.generate_report()
    
    def generate_report(self):
        """Generate a detailed report of the augmentation process"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "collections": {}
        }
        
        for collection_name, stats in self.stats.items():
            report["collections"][collection_name] = {
                "original_count": stats["original_count"],
                "augmented_count": stats["augmented_count"],
                "augmentation_factor": stats["augmented_count"] / max(stats["original_count"], 1),
                "strategies": stats["strategies"],
                "sentiment_distribution": stats["sentiment_distribution"],
                "topic_distribution": stats["topic_distribution"],
                "top_keywords": dict(sorted(
                    stats["keyword_frequency"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20])  # Top 20 keywords
            }
        
        # Save report to file
        with open(f"augmentation_report_{int(time.time())}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def update_prompt_strategy(self, strategy_name, new_prompt_template):
        """Update a prompt strategy at runtime"""
        Config.USER_PROMPTS[strategy_name] = new_prompt_template
        logger.info(f"Updated prompt strategy: {strategy_name}")
        return True

# Evaluation utilities
class AugmentationEvaluator:
    def __init__(self, qdrant_handler):
        self.qdrant = qdrant_handler
        
    def calculate_diversity(self, collection_name, text_fields, sample_size=100):
        """Calculate diversity metrics for augmented data"""
        points = self.qdrant.get_points(collection_name, limit=sample_size)
        
        if not points:
            return {"error": "No points found"}
        
        # Extract text from points
        texts = []
        for point in points:
            combined_text = " ".join([point.payload.get(field, "") for field in text_fields if field in point.payload])
            texts.append(combined_text)
        
        # Calculate pairwise similarity
        vectors = []
        gpt = GPTInterface(api_key=Config.OPENAI_API_KEY)
        
        for text in texts:
            vector = gpt.generate_embedding(text)
            if vector:
                vectors.append(vector)
        
        if len(vectors) < 2:
            return {"error": "Not enough vectors for comparison"}
        
        similarities = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                similarity = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                similarities.append(similarity)
        
        # Calculate statistics
        avg_similarity = sum(similarities) / len(similarities)
        diversity_score = 1.0 - avg_similarity
        
        return {
            "average_similarity": avg_similarity,
            "diversity_score": diversity_score,
            "min_similarity": min(similarities),
            "max_similarity": max(similarities)
        }
    
    def compare_original_vs_augmented(self, original_collection, augmented_collection, text_fields):
        """Compare statistics between original and augmented collections"""
        # Get sample from both collections
        original_points = self.qdrant.get_points(original_collection, limit=100)
        augmented_points = self.qdrant.get_points(augmented_collection, limit=100)
        
        if not original_points or not augmented_points:
            return {"error": "Couldn't get points from collections"}
        
        # Text length comparison
        original_lengths = []
        for point in original_points:
            combined_text = " ".join([point.payload.get(field, "") for field in text_fields if field in point.payload])
            original_lengths.append(len(combined_text))
        
        augmented_lengths = []
        for point in augmented_points:
            combined_text = " ".join([point.payload.get(field, "") for field in text_fields if field in point.payload])
            augmented_lengths.append(len(combined_text))
        
        # Calculate statistics
        avg_original_length = sum(original_lengths) / len(original_lengths) if original_lengths else 0
        avg_augmented_length = sum(augmented_lengths) / len(augmented_lengths) if augmented_lengths else 0
        
        # Topic distribution comparison
        original_topics = [point.payload.get("topic", "unknown") for point in original_points]
        augmented_topics = [point.payload.get("topic", "unknown") for point in augmented_points]
        
        original_topic_dist = Counter(original_topics)
        augmented_topic_dist = Counter(augmented_topics)
        
        return {
            "text_length": {
                "original_avg": avg_original_length,
                "augmented_avg": avg_augmented_length,
                "percent_change": ((avg_augmented_length - avg_original_length) / avg_original_length * 100) 
                                  if avg_original_length else 0
            },
            "topic_distribution": {
                "original": dict(original_topic_dist),
                "augmented": dict(augmented_topic_dist)
            }
        }
    
    def test_augmentation(self, text, strategy, prompt=None):
        """Test a single augmentation on a text sample"""
        gpt = GPTInterface(api_key=Config.OPENAI_API_KEY)
        augmenter = DataAugmenter(gpt)
        
        if strategy in augmenter.strategies:
            strategy_func = augmenter.strategies[strategy]
            if prompt:
                augmented_text = strategy_func(text, prompt)
            else:
                augmented_text = strategy_func(text)
            return augmented_text
        return None

# Streamlit UI
def main():

    st.sidebar.title("🌙 مزيد")
    st.sidebar.subheader("Arabic Data Augmentation")
    
    # Initialize connection to Qdrant
    if 'qdrant' not in st.session_state:
        st.session_state.qdrant = QdrantHandler(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "✨ Augmentation", "💬 Prompt Engineering", "🔍 Testing & Evaluation", "⚙️ Settings"]
    )
    
    # Initialize OpenAI API key from session state
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    # Settings page
    if page == "⚙️ Settings":
        st.title("⚙️ Settings")
        
        st.header("API Configuration")
        api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
        if api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key
            Config.OPENAI_API_KEY = api_key
            st.success("API key updated successfully!")
        
        model = st.selectbox(
            "GPT Model",
            ["gpt-4", "gpt-3.5-turbo"],
            index=0 if Config.GPT_MODEL == "gpt-4" else 1
        )
        if model != Config.GPT_MODEL:
            Config.GPT_MODEL = model
            st.success(f"Model changed to {model}")
        
        st.header("Qdrant Configuration")
        qdrant_host = st.text_input("Qdrant Host", value=Config.QDRANT_HOST)
        qdrant_port = st.number_input("Qdrant Port", value=Config.QDRANT_PORT)
        
        if qdrant_host != Config.QDRANT_HOST or qdrant_port != Config.QDRANT_PORT:
            if st.button("Update Qdrant Connection"):
                Config.QDRANT_HOST = qdrant_host
                Config.QDRANT_PORT = qdrant_port
                st.session_state.qdrant = QdrantHandler(host=qdrant_host, port=qdrant_port)
                st.success("Qdrant connection updated!")
        
        st.header("Augmentation Parameters")
        augmentation_factor = st.slider("Augmentation Factor", 1, 10, Config.AUGMENTATION_FACTOR)
        batch_size = st.slider("Batch Size", 10, 200, Config.BATCH_SIZE)
        max_workers = st.slider("Max Workers", 1, 10, Config.MAX_WORKERS)
        
        if augmentation_factor != Config.AUGMENTATION_FACTOR or batch_size != Config.BATCH_SIZE or max_workers != Config.MAX_WORKERS:
            if st.button("Update Augmentation Parameters"):
                Config.AUGMENTATION_FACTOR = augmentation_factor
                Config.BATCH_SIZE = batch_size
                Config.MAX_WORKERS = max_workers
                st.success("Augmentation parameters updated!")
        
        st.header("Collection Configuration")
        collection_prefix = st.text_input("New Collection Prefix", value=Config.NEW_COLLECTION_PREFIX)
        if collection_prefix != Config.NEW_COLLECTION_PREFIX:
            Config.NEW_COLLECTION_PREFIX = collection_prefix
            st.success(f"New collection prefix set to: {collection_prefix}")
    
    # Dashboard page
    elif page == "📊 Dashboard":
        st.title("📊 Dashboard")
        
        collections = st.session_state.qdrant.list_collections()
        if not collections:
            st.warning("No collections found in Qdrant database.")
            st.info("Please make sure Qdrant is running and configured correctly in the Settings page.")
        else:
            st.success(f"Found {len(collections)} collections in the Qdrant database.")
            
            # Display collections
            st.header("Collections")
            col1, col2 = st.columns(2)
            
            original_collections = [c for c in collections if not c.startswith(Config.NEW_COLLECTION_PREFIX)]
            augmented_collections = [c for c in collections if c.startswith(Config.NEW_COLLECTION_PREFIX)]
            
            with col1:
                st.subheader("Original Collections")
                for coll in original_collections:
                    count = st.session_state.qdrant.count_points(coll)
                    st.write(f"**{coll}**: {count} points")
            
            with col2:
                st.subheader("Augmented Collections")
                for coll in augmented_collections:
                    count = st.session_state.qdrant.count_points(coll)
                    st.write(f"**{coll}**: {count} points")
            
            # Visualization if there are augmented collections
            if augmented_collections:
                st.header("Augmented Data Analysis")
                selected_collection = st.selectbox("Select an augmented collection", augmented_collections)
                
                if selected_collection:
                    try:
                        # Get sample data from collection
                        points = st.session_state.qdrant.get_points(selected_collection, limit=100)
                        
                        if points:
                            # Analyze sentiment distribution
                            sentiments = [point.payload.get('sentiment', 'unknown') for point in points]
                            sentiment_count = Counter(sentiments)
                            
                            # Analyze topics
                            topics = [point.payload.get('topic', 'unknown') for point in points]
                            topic_count = Counter(topics)
                            
                            # Get augmentation strategies
                            strategies = [point.payload.get('_augmentation_info', {}).get('strategy', 'unknown') for point in points]
                            strategy_count = Counter(strategies)
                            
                            # Visualize
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Sentiment Distribution")
                                fig, ax = plt.subplots()
                                if sum(sentiment_count.values()) > 0:
                                    ax.pie(sentiment_count.values(), labels=sentiment_count.keys(), autopct='%1.1f%%')
                                else:
                                    ax.text(0.5, 0.5, "No sentiment data", ha='center', va='center')
                                    ax.axis('off')
                                st.pyplot(fig)
                                st.subheader("Augmentation Strategies")
                                fig, ax = plt.subplots()
                                ax.bar(strategy_count.keys(), strategy_count.values())
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                            
                            with col2:
                                st.subheader("Top Topics")
                                fig, ax = plt.subplots()
                                # Get top 5 topics
                                top_topics = dict(topic_count.most_common(5))
                                ax.bar(top_topics.keys(), top_topics.values())
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                                
                                # Show sample data
                                st.subheader("Sample Augmented Data")
                                sample_point = random.choice(points)
                                st.json(sample_point.payload)
                    except Exception as e:
                        st.error(f"Error analyzing collection: {e}")
    
    # Augmentation page
    elif page == "✨ Augmentation":
        st.title("✨ Augmentation")
        
        if not st.session_state.openai_api_key:
            st.warning("OpenAI API key not set. Please configure it in the Settings page.")
            return
        
        collections = st.session_state.qdrant.list_collections()
        original_collections = [c for c in collections if not c.startswith(Config.NEW_COLLECTION_PREFIX)]
        
        if not original_collections:
            st.warning("No original collections found to augment.")
            return
        
        st.header("Run Augmentation")
        
        # Collection selection
        selected_collections = st.multiselect(
            "Select collections to augment",
            original_collections,
            default=original_collections[0] if original_collections else None
        )
        
        # Strategy selection
        strategies = ["paraphrase", "add_details", "change_perspective", "translate_back", "rule_based"]
        selected_strategies = st.multiselect(
            "Select augmentation strategies",
            strategies,
            default=["paraphrase", "add_details", "change_perspective"]
        )
        
        # Augmentation button
        if st.button("Start Augmentation", disabled=not selected_collections or not selected_strategies):
            # Initialize pipeline
            pipeline = AugmentationPipeline(Config)
            
            # Create progress bar
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            # Run augmentation for each collection
            for i, collection in enumerate(selected_collections):
                status_text.text(f"Augmenting collection {i+1}/{len(selected_collections)}: {collection}")
                
                try:
                    pipeline.process_collection(
                        collection,
                        augmentation_strategies=selected_strategies,
                        progress_bar=progress_bar
                    )
                except Exception as e:
                    st.error(f"Error processing collection {collection}: {e}")
            
            # Generate report
            report = pipeline.generate_report()
            
            # Display summary
            status_text.text("Augmentation completed!")
            progress_bar.progress(1.0)
            
            st.header("Augmentation Summary")
            for collection_name, stats in report["collections"].items():
                st.subheader(f"Collection: {collection_name}")
                st.write(f"Original documents: {stats['original_count']}")
                st.write(f"Augmented documents: {stats['augmented_count']}")
                st.write(f"Augmentation factor: {stats['augmented_count'] / max(stats['original_count'], 1):.2f}x")
                
                # Display strategy distribution
                st.write("Strategy distribution:")
                fig, ax = plt.subplots()
                strategy_names = list(stats["strategies"].keys())
                strategy_counts = list(stats["strategies"].values())
                ax.bar(strategy_names, strategy_counts)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Display sentiment distribution
                # Display sentiment distribution
                st.write("Sentiment distribution:")
                fig, ax = plt.subplots()
                sentiment_labels = list(stats["sentiment_distribution"].keys())
                sentiment_counts = list(stats["sentiment_distribution"].values())

                # Add this check to prevent errors with empty data
                if sum(sentiment_counts) > 0:
                    ax.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%')
                else:
                    ax.text(0.5, 0.5, "No sentiment data", ha='center', va='center')
                    ax.axis('off')  # Hide the axes

                st.pyplot(fig)
    
    # Prompt Engineering page
    elif page == "💬 Prompt Engineering":
        st.title("💬 Prompt Engineering")
        
        if not st.session_state.openai_api_key:
            st.warning("OpenAI API key not set. Please configure it in the Settings page.")
            return
        
        st.header("Prompt Strategies")
        
        selected_strategy = st.selectbox(
            "Select a strategy to customize",
            list(Config.PROMPTS.keys())
        )
        
        if selected_strategy:
            current_prompt = Config.USER_PROMPTS.get(selected_strategy, Config.PROMPTS[selected_strategy])
            
            st.subheader(f"Customize Prompt for: {selected_strategy}")
            new_prompt = st.text_area("Prompt Template (use {text} as placeholder)", current_prompt, height=200)
            
            if st.button("Update Prompt"):
                pipeline = AugmentationPipeline(Config)
                pipeline.update_prompt_strategy(selected_strategy, new_prompt)
                st.success(f"Updated prompt for {selected_strategy}")
            
            # Test prompt
            st.subheader("Test Prompt")
            sample_text = st.text_area("Enter sample text to test", "نقص في نظافة يؤدي إلى مشاكل أمنية.")
            
            if st.button("Test"):
                evaluator = AugmentationEvaluator(st.session_state.qdrant)
                result = evaluator.test_augmentation(sample_text, selected_strategy, new_prompt)
                
                if result:
                    st.success("Augmentation successful!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Text")
                        st.write(sample_text)
                    
                    with col2:
                        st.subheader("Augmented Text")
                        st.write(result)
                else:
                    st.error("Augmentation failed. Check the logs for details.")
    
    # Testing & Evaluation page
    elif page == "🔍 Testing & Evaluation":
        st.title("🔍 Testing & Evaluation")
        
        if not st.session_state.openai_api_key:
            st.warning("OpenAI API key not set. Please configure it in the Settings page.")
            return
        
        collections = st.session_state.qdrant.list_collections()
        
        if not collections:
            st.warning("No collections found to evaluate.")
            return
        
        st.header("Collection Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            original_collection = st.selectbox(
                "Select original collection",
                [c for c in collections if not c.startswith(Config.NEW_COLLECTION_PREFIX)]
            )
        
        with col2:
            augmented_collections = [c for c in collections if c.startswith(Config.NEW_COLLECTION_PREFIX)]
            augmented_collection = st.selectbox(
                "Select augmented collection",
                augmented_collections,
                index=0 if augmented_collections else None
            )
        
        if original_collection and augmented_collection:
            # Get fields
            pipeline = AugmentationPipeline(Config)
            text_fields = pipeline.get_text_fields(original_collection)
            
            selected_fields = st.multiselect(
                "Select text fields to analyze",
                text_fields,
                default=text_fields
            )
            
            if selected_fields and st.button("Compare Collections"):
                with st.spinner("Comparing collections..."):
                    evaluator = AugmentationEvaluator(st.session_state.qdrant)
                    comparison = evaluator.compare_original_vs_augmented(
                        original_collection, 
                        augmented_collection,
                        selected_fields
                    )
                    
                    if "error" in comparison:
                        st.error(comparison["error"])
                    else:
                        st.subheader("Text Length Comparison")
                        st.write(f"Original average length: {comparison['text_length']['original_avg']:.2f} characters")
                        st.write(f"Augmented average length: {comparison['text_length']['augmented_avg']:.2f} characters")
                        st.write(f"Percent change: {comparison['text_length']['percent_change']:.2f}%")
                        
                        st.subheader("Topic Distribution Comparison")
                        
                        # Create dataframe for visualization
                        topics = set(list(comparison['topic_distribution']['original'].keys()) + 
                                   list(comparison['topic_distribution']['augmented'].keys()))
                        
                        data = []
                        for topic in topics:
                            data.append({
                                'Topic': topic,
                                'Original': comparison['topic_distribution']['original'].get(topic, 0),
                                'Augmented': comparison['topic_distribution']['augmented'].get(topic, 0)
                            })
                        
                        df = pd.DataFrame(data)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot grouped bar chart
                        x = np.arange(len(df))
                        width = 0.35
                        
                        ax.bar(x - width/2, df['Original'], width, label='Original')
                        ax.bar(x + width/2, df['Augmented'], width, label='Augmented')
                        
                        ax.set_xticks(x)
                        ax.set_xticklabels(df['Topic'], rotation=45)
                        ax.legend()
                        
                        st.pyplot(fig)
                
                # Calculate diversity
                with st.spinner("Calculating diversity metrics..."):
                    diversity = evaluator.calculate_diversity(augmented_collection, selected_fields)
                    
                    if "error" in diversity:
                        st.error(diversity["error"])
                    else:
                        st.subheader("Diversity Metrics")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Diversity Score (0-1)", f"{diversity['diversity_score']:.4f}")
                            st.write("Higher value indicates more diverse augmentation")
                        
                        with col2:
                            st.metric("Average Similarity", f"{diversity['average_similarity']:.4f}")
                            st.write("Lower value indicates more diverse augmentation")
                        
                        st.write(f"Min Similarity: {diversity['min_similarity']:.4f}")
                        st.write(f"Max Similarity: {diversity['max_similarity']:.4f}")
        
        # Single test area
        st.header("Test Single Augmentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_strategy = st.selectbox(
                "Select strategy",
                list(Config.PROMPTS.keys())
            )
        
        with col2:
            use_custom_prompt = st.checkbox("Use custom prompt")
        
        test_text = st.text_area("Enter text to augment", "هناك مشكلة في إنارة تؤثر على السكان.")
        
        if use_custom_prompt:
            if test_strategy in Config.PROMPTS:
                current_prompt = Config.USER_PROMPTS.get(test_strategy, Config.PROMPTS[test_strategy])
                test_prompt = st.text_area("Custom prompt (use {text} as placeholder)", current_prompt, height=150)
            else:
                test_prompt = st.text_area("Custom prompt (use {text} as placeholder)", "", height=150)
        else:
            test_prompt = None
        
        if st.button("Test Augmentation"):
            with st.spinner("Augmenting..."):
                evaluator = AugmentationEvaluator(st.session_state.qdrant)
                result = evaluator.test_augmentation(test_text, test_strategy, test_prompt if use_custom_prompt else None)
                
                if result:
                    st.subheader("Augmentation Result")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Text**")
                        st.write(test_text)
                    
                    with col2:
                        st.markdown("**Augmented Text**")
                        st.write(result)
                    
                    # Quick analysis
                    gpt = GPTInterface(api_key=Config.OPENAI_API_KEY)
                    augmenter = DataAugmenter(gpt)
                    
                    with st.spinner("Analyzing result..."):
                        sentiment = augmenter.analyze_sentiment(result)
                        topic = augmenter.analyze_topic(result)
                        keywords = augmenter.extract_keywords(result)
                        
                        st.subheader("Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("**Sentiment**")
                            st.write(f"Classification: {sentiment['sentiment']}")
                            st.write(f"Polarity: {sentiment['polarity']:.2f}")
                        
                        with col2:
                            st.markdown("**Topic**")
                            st.write(topic['topic'])
                        
                        with col3:
                            st.markdown("**Keywords**")
                            st.write(", ".join(keywords['keywords']))
                else:
                    st.error("Augmentation failed. Check the logs for details.")

if __name__ == "__main__":
    main()