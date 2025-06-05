#!/usr/bin/env python3
"""
Script to create a pros & cons detection dataset from existing Qdrant collections.
Designed for Arabic and Darija (Moroccan dialect) text analysis.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import random
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue
)
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qdrant connection details
QDRANT_URL = "http://154.44.186.241:6333"
NEW_COLLECTION_NAME = "pros_cons_detection_dataset"

# Arabic/Darija pros indicators (positive aspects)
PROS_PATTERNS_AR = [
    # Standard Arabic
    r'\b(ممتاز|جيد|رائع|مفيد|إيجابي|نافع|مهم|ضروري|فعال|ناجح)\b',
    r'\b(أحسن|أفضل|أجود|أنسب|أكثر فائدة)\b',
    r'\b(يساعد|يدعم|يحسن|يطور|يسهل|ينمي)\b',
    r'\b(ميزة|فائدة|إيجابية|قوة|نجاح)\b',
    r'\b(الحمد لله|ما شاء الله|بارك الله)\b',
    
    # Darija
    r'\b(زوين|مزيان|هايل|عاجبني|مبروك|تبارك الله)\b',
    r'\b(خدام|ناجح|مفيد|باين|واضح)\b',
    r'\b(كيساعد|كيخدم|كيفيد|كينفع)\b',
    r'\b(نقطة إيجابية|حاجة مزيانة|شي حاجة زوينة)\b',
    
    # Expressions of satisfaction
    r'\b(راضي|مرتاح|فرحان|مسرور|مبسوط)\b',
    r'\b(شكرا|بارك الله فيك|الله يحفظك|الله يعطيك الصحة)\b'
]

# Arabic/Darija cons indicators (negative aspects)
CONS_PATTERNS_AR = [
    # Standard Arabic
    r'\b(سيء|سلبي|ضعيف|مشكل|عيب|نقص|خلل|فشل)\b',
    r'\b(أسوأ|أضعف|أقل|غير مناسب|غير فعال)\b',
    r'\b(يعيق|يمنع|يضر|يؤخر|يعطل)\b',
    r'\b(مشكلة|عقبة|صعوبة|تحدي|معضلة)\b',
    r'\b(للأسف|مع الأسف|لا يمكن|غير ممكن)\b',
    
    # Darija
    r'\b(خايب|ماشي مزيان|عيان|مقلق|مشكيل)\b',
    r'\b(ما كيخدمش|ما نافعش|ما مزيانش)\b',
    r'\b(كيعطل|كيخرب|كيضر|ما كينفعش)\b',
    r'\b(نقطة سلبية|حاجة خايبة|مشكل كبير)\b',
    
    # Expressions of dissatisfaction
    r'\b(مقلق|زعفان|قلقان|مستاء|غاضب)\b',
    r'\b(واش هاد الشي|علاش|كيفاش|فين)\b'
]

# Neutral/balanced expressions
NEUTRAL_PATTERNS_AR = [
    r'\b(من جهة|من ناحية|لكن|ولكن|غير أن|إلا أن)\b',
    r'\b(بالرغم من|مع ذلك|في المقابل|من جانب آخر)\b',
    r'\b(إيجابيات وسلبيات|له وعليه|حسنات وسيئات)\b',
    r'\b(واخا|ولو|بصح|معا ذلك)\b'  # Darija
]

# Collections to process with their text field mappings
COLLECTIONS_CONFIG = {
    "citizen_comments": {
        "text_fields": ["comment", "text", "content", "feedback", "message"],
        "metadata_fields": ["citizen_id", "date", "topic", "sentiment", "location", "category"],
        "source_type": "citizen_feedback",
        "priority": "high",
        "min_text_length": 30
    },
    "remacto_comments": {
        "text_fields": ["comment", "text", "content", "message"],
        "metadata_fields": ["author", "date", "project_id", "sentiment", "rating"],
        "source_type": "platform_comments",
        "priority": "high",
        "min_text_length": 30
    },
    "citizen_ideas": {
        "text_fields": ["idea", "description", "proposal", "content", "details"],
        "metadata_fields": ["citizen_id", "date", "category", "votes", "status"],
        "source_type": "citizen_proposals",
        "priority": "high",
        "min_text_length": 50
    },
    "hespress_politics_comments": {
        "text_fields": ["comment", "text", "content"],
        "metadata_fields": ["article_id", "author", "date", "likes"],
        "source_type": "news_comments",
        "priority": "medium",
        "min_text_length": 30
    },
    "project_updates": {
        "text_fields": ["update", "description", "content", "message"],
        "metadata_fields": ["project_id", "date", "status", "author"],
        "source_type": "project_feedback",
        "priority": "medium",
        "min_text_length": 50
    },
    "municipal_projects": {
        "text_fields": ["description", "title", "objectives", "benefits", "challenges"],
        "metadata_fields": ["project_id", "status", "budget", "location"],
        "source_type": "project_descriptions",
        "priority": "medium",
        "min_text_length": 100
    },
    "budget_allocations": {
        "text_fields": ["description", "justification", "benefits", "risks"],
        "metadata_fields": ["department", "amount", "year", "category"],
        "source_type": "budget_analysis",
        "priority": "low",
        "min_text_length": 50
    }
}

class ProsConsDetectionDatasetCreator:
    def __init__(self, qdrant_url: str):
        """Initialize the dataset creator with Qdrant client."""
        self.client = QdrantClient(url=qdrant_url)
        self.processed_texts = set()  # To avoid duplicates
        self.vector_dim = None  # Will be determined from data
        self.statistics = {
            "total_texts": 0,
            "pros_detected": 0,
            "cons_detected": 0,
            "mixed_detected": 0,
            "neutral_detected": 0,
            "by_source": {},
            "by_classification": {"pros": 0, "cons": 0, "mixed": 0, "neutral": 0}
        }
    
    def determine_vector_dimension(self) -> int:
        """Determine the most common vector dimension from existing collections."""
        dim_counts = {}
        
        for collection_name, config in COLLECTIONS_CONFIG.items():
            try:
                collection_info = self.client.get_collection(collection_name)
                vectors_config = collection_info.config.params.vectors
                
                if hasattr(vectors_config, 'size'):
                    dim = vectors_config.size
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1
                elif isinstance(vectors_config, dict) and 'size' in vectors_config:
                    dim = vectors_config['size'].size
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1
                    
                logger.info(f"Collection {collection_name} has vector dimension: {dim if 'dim' in locals() else 'unknown'}")
                
            except Exception as e:
                logger.warning(f"Could not get vector info for {collection_name}: {e}")
        
        if dim_counts:
            most_common_dim = max(dim_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Most common vector dimension: {most_common_dim}")
            return most_common_dim
        else:
            logger.warning("Could not determine vector dimension, using default 384")
            return 384
    
    def create_collection(self, collection_name: str, vector_size: int = None):
        """Create the new pros & cons detection collection."""
        if vector_size is None:
            vector_size = self.vector_dim or 384
            
        try:
            # Delete collection if it exists
            collections = self.client.get_collections().collections
            if any(col.name == collection_name for col in collections):
                logger.info(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
            
            # Create new collection
            logger.info(f"Creating new collection: {collection_name} with vector size: {vector_size}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                on_disk_payload=True
            )
            logger.info(f"Collection {collection_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def extract_text_from_payload(self, payload: Dict[str, Any], text_fields: List[str]) -> Optional[str]:
        """Extract text content from payload based on configured fields."""
        texts = []
        
        for field in text_fields:
            if field in payload and payload[field]:
                text = str(payload[field]).strip()
                if text:
                    texts.append(text)
        
        if texts:
            return " ".join(texts)
        return None
    
    def detect_pros_cons(self, text: str) -> Tuple[int, int, bool]:
        """
        Detect pros and cons in Arabic/Darija text.
        Returns: (pros_score, cons_score, has_neutral_markers)
        """
        pros_score = 0
        cons_score = 0
        has_neutral = False
        
        # Check for pros patterns
        for pattern in PROS_PATTERNS_AR:
            matches = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
            pros_score += len(matches)
        
        # Check for cons patterns
        for pattern in CONS_PATTERNS_AR:
            matches = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
            cons_score += len(matches)
        
        # Check for neutral/balanced expressions
        for pattern in NEUTRAL_PATTERNS_AR:
            if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
                has_neutral = True
                break
        
        return pros_score, cons_score, has_neutral
    
    def classify_text(self, text: str) -> str:
        """
        Classify text based on pros/cons detection.
        Returns: 'pros', 'cons', 'mixed', or 'neutral'
        """
        pros_score, cons_score, has_neutral = self.detect_pros_cons(text)
        
        # If text explicitly mentions both pros and cons or uses balanced language
        if has_neutral and (pros_score > 0 or cons_score > 0):
            return 'mixed'
        
        # If both pros and cons are detected
        if pros_score > 0 and cons_score > 0:
            # If one significantly outweighs the other
            if pros_score > cons_score * 2:
                return 'pros'
            elif cons_score > pros_score * 2:
                return 'cons'
            else:
                return 'mixed'
        
        # Clear pros or cons
        if pros_score > 0:
            return 'pros'
        elif cons_score > 0:
            return 'cons'
        else:
            return 'neutral'
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features relevant for pros/cons detection."""
        pros_score, cons_score, has_neutral = self.detect_pros_cons(text)
        
        # Count sentence endings (Arabic uses different punctuation)
        sentence_count = len(re.findall(r'[.!?؟।।]', text))
        
        # Count questions (Arabic question mark)
        question_count = text.count('؟') + text.count('?')
        
        # Calculate sentiment indicators
        exclamation_count = text.count('!') + text.count('।')
        
        features = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "sentence_count": max(sentence_count, 1),
            "pros_score": pros_score,
            "cons_score": cons_score,
            "has_neutral_markers": has_neutral,
            "pros_cons_ratio": pros_score / max(cons_score, 1),
            "total_indicators": pros_score + cons_score,
            "question_count": question_count,
            "exclamation_count": exclamation_count,
            "avg_sentence_length": len(text.split()) / max(sentence_count, 1),
            "classification": self.classify_text(text)
        }
        
        return features
    
    def process_collection(self, collection_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single collection and extract pros/cons data."""
        logger.info(f"Processing collection: {collection_name} (Priority: {config['priority']})")
        processed_data = []
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if not any(col.name == collection_name for col in collections):
                logger.warning(f"Collection {collection_name} not found, skipping...")
                return processed_data
            
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            points_count = collection_info.points_count
            
            if points_count == 0:
                logger.info(f"Collection {collection_name} is empty, skipping...")
                return processed_data
            
            logger.info(f"Found {points_count} points in {collection_name}")
            
            # Scroll through all points
            offset = None
            batch_size = 100
            total_processed = 0
            
            while True:
                result = self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = result
                
                if not points:
                    break
                
                for point in points:
                    if not point.payload:
                        continue
                    
                    # Extract text
                    text = self.extract_text_from_payload(point.payload, config["text_fields"])
                    
                    if text and len(text) >= config["min_text_length"]:
                        # Check for duplicates
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        if text_hash in self.processed_texts:
                            continue
                        
                        self.processed_texts.add(text_hash)
                        
                        # Extract features
                        features = self.extract_features(text)
                        
                        # Extract metadata
                        metadata = {}
                        for field in config["metadata_fields"]:
                            if field in point.payload and point.payload[field] is not None:
                                metadata[field] = point.payload[field]
                        
                        # Prepare data point
                        data_point = {
                            "id": str(uuid.uuid4()),
                            "text": text,
                            "source_collection": collection_name,
                            "source_type": config["source_type"],
                            "priority": config["priority"],
                            "metadata": metadata,
                            "features": features,
                            "classification": features["classification"],
                            "pros_score": features["pros_score"],
                            "cons_score": features["cons_score"],
                            "has_neutral_markers": features["has_neutral_markers"],
                            "original_id": str(point.id),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Handle vectors with dimension check
                        if point.vector:
                            vector_extracted = False
                            if isinstance(point.vector, dict):
                                if "size" in point.vector and isinstance(point.vector["size"], list):
                                    if len(point.vector["size"]) == self.vector_dim:
                                        data_point["vector"] = point.vector["size"]
                                        vector_extracted = True
                            elif isinstance(point.vector, list):
                                if len(point.vector) == self.vector_dim:
                                    data_point["vector"] = point.vector
                                    vector_extracted = True
                            
                            if not vector_extracted:
                                data_point["vector"] = [0.0] * self.vector_dim
                        else:
                            data_point["vector"] = [0.0] * self.vector_dim
                        
                        processed_data.append(data_point)
                        total_processed += 1
                        
                        # Update statistics
                        self.statistics["total_texts"] += 1
                        self.statistics["by_classification"][features["classification"]] += 1
                        self.statistics["by_source"][config["source_type"]] = \
                            self.statistics["by_source"].get(config["source_type"], 0) + 1
                
                offset = next_offset
                
                if offset is None:
                    break
            
            logger.info(f"Processed {total_processed} valid texts from {collection_name}")
            
        except Exception as e:
            logger.error(f"Error processing collection {collection_name}: {e}")
        
        return processed_data
    
    def balance_dataset(self, all_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance the dataset to ensure good representation of all classifications."""
        # Group by classification
        by_classification = {'pros': [], 'cons': [], 'mixed': [], 'neutral': []}
        
        for item in all_data:
            classification = item["classification"]
            by_classification[classification].append(item)
        
        # Log distribution
        logger.info("Classification distribution before balancing:")
        for cls, items in by_classification.items():
            logger.info(f"  {cls}: {len(items)}")
        
        # Balance the dataset
        balanced_data = []
        
        # Find the minimum class size (excluding neutral if it's too small)
        non_empty_classes = {k: v for k, v in by_classification.items() if len(v) > 100}
        if non_empty_classes:
            target_size = min(len(items) for items in non_empty_classes.values())
            target_size = min(target_size * 2, 10000)  # Cap at 10k per class
            
            for classification, items in by_classification.items():
                if items:
                    # Sample up to target size
                    sample_size = min(len(items), target_size)
                    sampled = random.sample(items, sample_size)
                    balanced_data.extend(sampled)
        else:
            # If no good distribution, just use all data
            balanced_data = all_data
        
        random.shuffle(balanced_data)
        
        logger.info(f"Balanced dataset size: {len(balanced_data)}")
        return balanced_data
    
    def insert_data_batch(self, collection_name: str, data_batch: List[Dict[str, Any]]):
        """Insert a batch of data into the collection."""
        points = []
        
        for item in data_batch:
            # Prepare payload
            payload = {
                "text": item["text"],
                "source_collection": item["source_collection"],
                "source_type": item["source_type"],
                "priority": item["priority"],
                "classification": item["classification"],
                "pros_score": item["pros_score"],
                "cons_score": item["cons_score"],
                "has_neutral_markers": item["has_neutral_markers"],
                "original_id": item["original_id"],
                "timestamp": item["timestamp"],
                "features": json.dumps(item["features"]),
                "text_length": item["features"]["text_length"],
                "word_count": item["features"]["word_count"]
            }
            
            # Add metadata fields
            if item["metadata"]:
                payload["metadata"] = json.dumps(item["metadata"])
            
            # Create point with proper vector
            vector = item.get("vector", [0.0] * self.vector_dim)
            
            point = PointStruct(
                id=item["id"],
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # Insert points
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Inserted {len(points)} points")
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
    
    def run(self):
        """Main execution method."""
        logger.info("Starting pros & cons detection dataset creation for Arabic/Darija...")
        
        # Determine vector dimension from existing collections
        self.vector_dim = self.determine_vector_dimension()
        
        # Create new collection
        if not self.create_collection(NEW_COLLECTION_NAME, self.vector_dim):
            logger.error("Failed to create collection, exiting...")
            return
        
        # Process collections by priority
        all_data = []
        
        for collection_name, config in sorted(COLLECTIONS_CONFIG.items(), 
                                            key=lambda x: (x[1]["priority"] != "high", x[0])):
            data = self.process_collection(collection_name, config)
            all_data.extend(data)
            logger.info(f"Total data points so far: {len(all_data)}")
        
        # Balance dataset
        logger.info("Balancing dataset...")
        balanced_data = self.balance_dataset(all_data)
        
        # Insert data in batches
        logger.info(f"Inserting {len(balanced_data)} data points into {NEW_COLLECTION_NAME}")
        
        batch_size = 100
        for i in range(0, len(balanced_data), batch_size):
            batch = balanced_data[i:i + batch_size]
            self.insert_data_batch(NEW_COLLECTION_NAME, batch)
            logger.info(f"Progress: {min(i + batch_size, len(balanced_data))}/{len(balanced_data)}")
        
        # Verify collection
        collection_info = self.client.get_collection(NEW_COLLECTION_NAME)
        logger.info(f"Collection created successfully with {collection_info.points_count} points")
        
        # Print summary statistics
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"Total texts processed: {self.statistics['total_texts']}")
        
        logger.info("\nClassification Distribution:")
        for cls, count in sorted(self.statistics['by_classification'].items()):
            percentage = (count / max(self.statistics['total_texts'], 1)) * 100
            logger.info(f"  {cls}: {count} ({percentage:.1f}%)")
        
        logger.info("\nSource Type Distribution:")
        for source, count in sorted(self.statistics['by_source'].items(), 
                                   key=lambda x: x[1], reverse=True):
            logger.info(f"  {source}: {count}")
        
        logger.info("\nDataset Characteristics:")
        logger.info(f"  - Vector dimension: {self.vector_dim}")
        logger.info(f"  - Optimized for Arabic/Darija text")
        logger.info(f"  - Includes pros/cons scoring and classification")
        logger.info(f"  - Balanced representation across classifications")
        logger.info(f"  - Ready for pros & cons detection model training")

def main():
    """Main function."""
    creator = ProsConsDetectionDatasetCreator(QDRANT_URL)
    creator.run()

if __name__ == "__main__":
    main()