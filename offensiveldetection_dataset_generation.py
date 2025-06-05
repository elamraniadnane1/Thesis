#!/usr/bin/env python3
"""
Script to create an offensive language detection dataset from existing Qdrant collections.
This will aggregate text data suitable for training models to detect inappropriate content.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
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
NEW_COLLECTION_NAME = "offensive_language_detection_dataset"

# Collections to process with their text field mappings and priorities
COLLECTIONS_CONFIG = {
    "remacto_comments": {
        "text_fields": ["comment", "text", "content", "message"],
        "metadata_fields": ["author", "date", "project_id", "sentiment", "rating", "status"],
        "source_type": "platform_comments",
        "priority": "high",  # User comments are high priority for offensive content
        "min_text_length": 20
    },
    "citizen_comments": {
        "text_fields": ["comment", "text", "content", "feedback", "message"],
        "metadata_fields": ["citizen_id", "date", "topic", "sentiment", "location", "category", "status"],
        "source_type": "citizen_feedback",
        "priority": "high",
        "min_text_length": 20
    },
    "hespress_politics_comments": {
        "text_fields": ["comment", "text", "content"],
        "metadata_fields": ["article_id", "author", "date", "likes", "dislikes", "replies"],
        "source_type": "news_comments",
        "priority": "high",
        "min_text_length": 20
    },
    "citizen_ideas": {
        "text_fields": ["idea", "description", "proposal", "content", "details"],
        "metadata_fields": ["citizen_id", "date", "category", "votes", "status"],
        "source_type": "citizen_proposals",
        "priority": "medium",
        "min_text_length": 30
    },
    "project_updates": {
        "text_fields": ["update", "description", "content", "message"],
        "metadata_fields": ["project_id", "date", "status", "author"],
        "source_type": "official_updates",
        "priority": "low",
        "min_text_length": 50
    },
    "municipal_projects": {
        "text_fields": ["description", "title", "objectives"],
        "metadata_fields": ["project_id", "status", "budget", "location"],
        "source_type": "project_descriptions",
        "priority": "low",
        "min_text_length": 50
    },
    "hespress_politics_details": {
        "text_fields": ["content", "title", "summary", "article"],
        "metadata_fields": ["url", "date", "author", "tags"],
        "source_type": "news_articles",
        "priority": "medium",
        "min_text_length": 100
    },
    "citizens": {
        "text_fields": ["bio", "interests", "concerns"],
        "metadata_fields": ["citizen_id", "location", "age_group"],
        "source_type": "user_profiles",
        "priority": "low",
        "min_text_length": 30
    }
}

# Common patterns that might indicate content needing review (not necessarily offensive)
REVIEW_PATTERNS = [
    r'\b(complaint|complain|problem|issue|angry|frustrated|disappointed)\b',
    r'\b(unfair|unjust|corrupt|corruption|scandal)\b',
    r'\b(protest|demand|refuse|reject)\b',
    r'[!]{3,}',  # Multiple exclamation marks
    r'[A-Z]{5,}',  # All caps words (shouting)
    r'\b(hate|dislike|terrible|awful|worst)\b'
]

class OffensiveLanguageDatasetCreator:
    def __init__(self, qdrant_url: str):
        """Initialize the dataset creator with Qdrant client."""
        self.client = QdrantClient(url=qdrant_url)
        self.processed_texts = set()  # To avoid duplicates
        self.text_statistics = {
            "total_texts": 0,
            "flagged_for_review": 0,
            "by_source": {},
            "by_length": {"short": 0, "medium": 0, "long": 0}
        }
        self.vector_dim = None  # Will be determined from data
        
    def determine_vector_dimension(self) -> int:
        """Determine the most common vector dimension from existing collections."""
        dim_counts = {}
        
        for collection_name, config in COLLECTIONS_CONFIG.items():
            try:
                collection_info = self.client.get_collection(collection_name)
                
                # Get vector configuration
                vectors_config = collection_info.config.params.vectors
                
                if hasattr(vectors_config, 'size'):
                    # Single vector config
                    dim = vectors_config.size
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1
                elif isinstance(vectors_config, dict):
                    # Named vectors - get the 'size' vector dimension
                    if 'size' in vectors_config:
                        dim = vectors_config['size'].size
                        dim_counts[dim] = dim_counts.get(dim, 0) + 1
                        
                logger.info(f"Collection {collection_name} has vector dimension: {dim if 'dim' in locals() else 'unknown'}")
                
            except Exception as e:
                logger.warning(f"Could not get vector info for {collection_name}: {e}")
        
        # Use the most common dimension, defaulting to 384
        if dim_counts:
            most_common_dim = max(dim_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Most common vector dimension: {most_common_dim}")
            return most_common_dim
        else:
            logger.warning("Could not determine vector dimension, using default 384")
            return 384
    
    def create_collection(self, collection_name: str, vector_size: int = None):
        """Create the new offensive language detection collection."""
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
        
        # Combine all found texts
        if texts:
            return " ".join(texts)
        return None
    
    def clean_text(self, text: str) -> str:
        """Clean text while preserving content that might be relevant for detection."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        # Keep punctuation and special characters as they might be relevant
        return text.strip()
    
    def calculate_text_features(self, text: str) -> Dict[str, Any]:
        """Calculate features that might be relevant for offensive language detection."""
        features = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
            "caps_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "special_char_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
            "avg_word_length": sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            "max_word_length": max(len(word) for word in text.split()) if text.split() else 0,
            "unique_words": len(set(text.lower().split())),
            "has_urls": bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            "has_email": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
            "has_phone": bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        }
        
        # Check for review patterns
        for pattern in REVIEW_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                features["needs_review"] = True
                break
        else:
            features["needs_review"] = False
        
        return features
    
    def categorize_text_length(self, text_length: int) -> str:
        """Categorize text by length."""
        if text_length < 100:
            return "short"
        elif text_length < 500:
            return "medium"
        else:
            return "long"
    
    def process_collection(self, collection_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single collection and extract data for offensive language detection."""
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
                        
                        # Clean text
                        cleaned_text = self.clean_text(text)
                        
                        # Calculate features
                        features = self.calculate_text_features(cleaned_text)
                        
                        # Extract metadata
                        metadata = {}
                        for field in config["metadata_fields"]:
                            if field in point.payload and point.payload[field] is not None:
                                metadata[field] = point.payload[field]
                        
                        # Prepare data point
                        data_point = {
                            "id": str(uuid.uuid4()),
                            "text": cleaned_text,
                            "original_text": text,  # Keep original for reference
                            "source_collection": collection_name,
                            "source_type": config["source_type"],
                            "priority": config["priority"],
                            "metadata": metadata,
                            "features": features,
                            "text_length": len(cleaned_text),
                            "length_category": self.categorize_text_length(len(cleaned_text)),
                            "needs_review": features["needs_review"],
                            "original_id": str(point.id),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Use existing vector if available
                        if point.vector:
                            vector_extracted = False
                            if isinstance(point.vector, dict):
                                if "size" in point.vector and isinstance(point.vector["size"], list):
                                    vector = point.vector["size"]
                                    # Check vector dimension
                                    if len(vector) == self.vector_dim:
                                        data_point["vector"] = vector
                                        vector_extracted = True
                                    else:
                                        logger.debug(f"Skipping vector with dimension {len(vector)}, expected {self.vector_dim}")
                                else:
                                    # Try to get the first available vector
                                    for key, vec in point.vector.items():
                                        if isinstance(vec, list) and len(vec) == self.vector_dim:
                                            data_point["vector"] = vec
                                            vector_extracted = True
                                            break
                            elif isinstance(point.vector, list):
                                # Single vector
                                if len(point.vector) == self.vector_dim:
                                    data_point["vector"] = point.vector
                                    vector_extracted = True
                                else:
                                    logger.debug(f"Skipping vector with dimension {len(point.vector)}, expected {self.vector_dim}")
                            
                            # If no compatible vector found, create zero vector
                            if not vector_extracted:
                                data_point["vector"] = [0.0] * self.vector_dim
                        
                        processed_data.append(data_point)
                        total_processed += 1
                        
                        # Update statistics
                        self.text_statistics["total_texts"] += 1
                        if features["needs_review"]:
                            self.text_statistics["flagged_for_review"] += 1
                        self.text_statistics["by_source"][config["source_type"]] = \
                            self.text_statistics["by_source"].get(config["source_type"], 0) + 1
                        self.text_statistics["by_length"][data_point["length_category"]] += 1
                
                offset = next_offset
                
                if offset is None:
                    break
            
            logger.info(f"Processed {total_processed} valid texts from {collection_name}")
            
        except Exception as e:
            logger.error(f"Error processing collection {collection_name}: {e}")
        
        return processed_data
    
    def balance_dataset(self, all_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance the dataset to ensure good representation from different sources."""
        # Group by source type
        by_source = {}
        for item in all_data:
            source = item["source_type"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item)
        
        # Prioritize high-priority sources but ensure diversity
        balanced_data = []
        
        # First, add all high-priority flagged content
        for item in all_data:
            if item["priority"] == "high" and item["needs_review"]:
                balanced_data.append(item)
        
        # Then add samples from each source
        remaining_quota = max(50000 - len(balanced_data), 0)  # Target size
        
        for source_type, items in by_source.items():
            # Calculate quota for this source
            if source_type in ["platform_comments", "citizen_feedback", "news_comments"]:
                source_quota = int(remaining_quota * 0.25)  # 25% each for high-priority sources
            else:
                source_quota = int(remaining_quota * 0.05)  # 5% for other sources
            
            # Add items up to quota
            items_to_add = [item for item in items if item not in balanced_data]
            random.shuffle(items_to_add)
            balanced_data.extend(items_to_add[:source_quota])
        
        random.shuffle(balanced_data)
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
                "text_length": item["text_length"],
                "length_category": item["length_category"],
                "needs_review": item["needs_review"],
                "original_id": item["original_id"],
                "timestamp": item["timestamp"],
                "features": json.dumps(item["features"]),
                "char_count": item["features"]["char_count"],
                "word_count": item["features"]["word_count"],
                "caps_ratio": item["features"]["caps_ratio"],
                "exclamation_count": item["features"]["exclamation_count"]
            }
            
            # Add metadata fields
            if item["metadata"]:
                payload["metadata"] = json.dumps(item["metadata"])
            
            # Create point with proper vector handling
            vector = item.get("vector", [0.0] * self.vector_dim)
            
            # Ensure vector has correct dimension
            if len(vector) != self.vector_dim:
                logger.warning(f"Vector dimension mismatch: got {len(vector)}, expected {self.vector_dim}")
                vector = [0.0] * self.vector_dim
            
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
        logger.info("Starting offensive language detection dataset creation...")
        
        # Determine vector dimension from existing collections
        self.vector_dim = self.determine_vector_dimension()
        
        # Create new collection with determined dimension
        if not self.create_collection(NEW_COLLECTION_NAME, self.vector_dim):
            logger.error("Failed to create collection, exiting...")
            return
        
        # Process collections by priority
        all_data = []
        
        # Process high priority collections first
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
        logger.info(f"Total texts processed: {self.text_statistics['total_texts']}")
        logger.info(f"Texts flagged for review: {self.text_statistics['flagged_for_review']} "
                   f"({self.text_statistics['flagged_for_review'] / max(self.text_statistics['total_texts'], 1) * 100:.1f}%)")
        
        logger.info("\nSource Type Distribution:")
        for source, count in sorted(self.text_statistics['by_source'].items(), 
                                   key=lambda x: x[1], reverse=True):
            logger.info(f"  {source}: {count}")
        
        logger.info("\nText Length Distribution:")
        for length_cat, count in self.text_statistics['by_length'].items():
            logger.info(f"  {length_cat}: {count}")
        
        logger.info("\nDataset Characteristics:")
        logger.info(f"  - Vector dimension: {self.vector_dim}")
        logger.info(f"  - High priority sources emphasized")
        logger.info(f"  - Includes text features for analysis")
        logger.info(f"  - Balanced representation across sources")
        logger.info(f"  - Ready for offensive language detection training")

def main():
    """Main function."""
    creator = OffensiveLanguageDatasetCreator(QDRANT_URL)
    creator.run()

if __name__ == "__main__":
    main()