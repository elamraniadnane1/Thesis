#!/usr/bin/env python3
"""
Script to create a comprehensive topic modeling collection from existing Qdrant collections.
This will aggregate text data from all collections for training models and LLMs.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    ScrollRequest, UpdateStatus
)
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qdrant connection details
QDRANT_URL = "http://154.44.186.241:6333"
NEW_COLLECTION_NAME = "topic_modeling_dataset"

# Collections to process with their text field mappings
COLLECTIONS_CONFIG = {
    "remacto_comments": {
        "text_fields": ["comment", "text", "content"],
        "metadata_fields": ["author", "date", "project_id", "sentiment"],
        "category": "citizen_feedback"
    },
    "municipal_projects": {
        "text_fields": ["description", "title", "objectives", "summary"],
        "metadata_fields": ["project_id", "status", "budget", "location", "department"],
        "category": "municipal_projects"
    },
    "project_updates": {
        "text_fields": ["update", "description", "content", "summary"],
        "metadata_fields": ["project_id", "date", "status", "author"],
        "category": "project_updates"
    },
    "hespress_politics_details": {
        "text_fields": ["content", "title", "summary", "article"],
        "metadata_fields": ["url", "date", "author", "tags"],
        "category": "news_politics"
    },
    "citizen_comments": {
        "text_fields": ["comment", "text", "content", "feedback"],
        "metadata_fields": ["citizen_id", "date", "topic", "sentiment", "location"],
        "category": "citizen_feedback"
    },
    "remacto_projects": {
        "text_fields": ["description", "title", "goals", "content"],
        "metadata_fields": ["project_id", "status", "budget", "timeline"],
        "category": "remacto_initiatives"
    },
    "citizen_ideas": {
        "text_fields": ["idea", "description", "proposal", "content"],
        "metadata_fields": ["citizen_id", "date", "category", "votes", "status"],
        "category": "citizen_proposals"
    },
    "citizens": {
        "text_fields": ["bio", "interests", "concerns", "profile"],
        "metadata_fields": ["citizen_id", "location", "age_group", "occupation"],
        "category": "citizen_profiles"
    },
    "morocco_centres": {
        "text_fields": ["description", "services", "information"],
        "metadata_fields": ["name", "location", "type", "region"],
        "category": "administrative_info"
    },
    "morocco_cercles": {
        "text_fields": ["description", "information", "services"],
        "metadata_fields": ["name", "province", "region"],
        "category": "administrative_info"
    },
    "municipal_officials": {
        "text_fields": ["bio", "responsibilities", "platform", "background"],
        "metadata_fields": ["name", "position", "department", "term"],
        "category": "municipal_governance"
    },
    "budget_allocations": {
        "text_fields": ["description", "justification", "details"],
        "metadata_fields": ["department", "amount", "year", "category"],
        "category": "budget_finance"
    },
    "morocco_arrondissements": {
        "text_fields": ["description", "characteristics", "information"],
        "metadata_fields": ["name", "city", "population"],
        "category": "administrative_info"
    },
    "engagement_metrics": {
        "text_fields": ["description", "analysis", "insights"],
        "metadata_fields": ["metric_type", "value", "date", "source"],
        "category": "analytics"
    },
    "hespress_politics_comments": {
        "text_fields": ["comment", "text", "content"],
        "metadata_fields": ["article_id", "author", "date", "likes"],
        "category": "news_comments"
    }
}

class TopicModelingCollectionCreator:
    def __init__(self, qdrant_url: str):
        """Initialize the collection creator with Qdrant client."""
        self.client = QdrantClient(url=qdrant_url)
        self.processed_texts = set()  # To avoid duplicates
        
    def create_collection(self, collection_name: str, vector_size: int = 384):
        """Create the new topic modeling collection."""
        try:
            # Delete collection if it exists
            collections = self.client.get_collections().collections
            if any(col.name == collection_name for col in collections):
                logger.info(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
            
            # Create new collection
            logger.info(f"Creating new collection: {collection_name}")
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
                if text and len(text) > 10:  # Minimum text length
                    texts.append(text)
        
        # Combine all found texts
        if texts:
            return " ".join(texts)
        return None
    
    def extract_metadata(self, payload: Dict[str, Any], metadata_fields: List[str]) -> Dict[str, Any]:
        """Extract metadata from payload."""
        metadata = {}
        
        for field in metadata_fields:
            if field in payload and payload[field] is not None:
                metadata[field] = payload[field]
        
        return metadata
    
    def generate_text_hash(self, text: str) -> str:
        """Generate hash for text to detect duplicates."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def process_collection(self, collection_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single collection and extract topic modeling data."""
        logger.info(f"Processing collection: {collection_name}")
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
                    
                    if text:
                        # Check for duplicates
                        text_hash = self.generate_text_hash(text)
                        if text_hash in self.processed_texts:
                            continue
                        
                        self.processed_texts.add(text_hash)
                        
                        # Extract metadata
                        metadata = self.extract_metadata(point.payload, config["metadata_fields"])
                        
                        # Prepare data point
                        data_point = {
                            "id": str(uuid.uuid4()),
                            "text": text,
                            "source_collection": collection_name,
                            "category": config["category"],
                            "metadata": metadata,
                            "text_length": len(text),
                            "word_count": len(text.split()),
                            "original_id": str(point.id),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Use existing vector if available
                        if point.vector:
                            if isinstance(point.vector, dict):
                                # Handle named vectors
                                if "size" in point.vector:
                                    data_point["vector"] = point.vector["size"]
                                else:
                                    # Use the first available vector
                                    first_key = next(iter(point.vector))
                                    data_point["vector"] = point.vector[first_key]
                            else:
                                data_point["vector"] = point.vector
                        
                        processed_data.append(data_point)
                        total_processed += 1
                
                offset = next_offset
                
                if offset is None:
                    break
            
            logger.info(f"Processed {total_processed} valid texts from {collection_name}")
            
        except Exception as e:
            logger.error(f"Error processing collection {collection_name}: {e}")
        
        return processed_data
    
    def insert_data_batch(self, collection_name: str, data_batch: List[Dict[str, Any]]):
        """Insert a batch of data into the collection."""
        points = []
        
        for item in data_batch:
            # Prepare payload
            payload = {
                "text": item["text"],
                "source_collection": item["source_collection"],
                "category": item["category"],
                "text_length": item["text_length"],
                "word_count": item["word_count"],
                "original_id": item["original_id"],
                "timestamp": item["timestamp"]
            }
            
            # Add metadata fields
            if item["metadata"]:
                payload["metadata"] = json.dumps(item["metadata"])
            
            # Create point
            point = PointStruct(
                id=item["id"],
                vector=item.get("vector", [0.0] * 384),  # Use zero vector if not available
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
        logger.info("Starting topic modeling collection creation process...")
        
        # Create new collection
        if not self.create_collection(NEW_COLLECTION_NAME):
            logger.error("Failed to create collection, exiting...")
            return
        
        # Process all collections
        all_data = []
        
        for collection_name, config in COLLECTIONS_CONFIG.items():
            data = self.process_collection(collection_name, config)
            all_data.extend(data)
            logger.info(f"Total data points so far: {len(all_data)}")
        
        # Insert data in batches
        logger.info(f"Inserting {len(all_data)} total data points into {NEW_COLLECTION_NAME}")
        
        batch_size = 100
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            self.insert_data_batch(NEW_COLLECTION_NAME, batch)
            logger.info(f"Progress: {min(i + batch_size, len(all_data))}/{len(all_data)}")
        
        # Verify collection
        collection_info = self.client.get_collection(NEW_COLLECTION_NAME)
        logger.info(f"Collection created successfully with {collection_info.points_count} points")
        
        # Print summary statistics
        logger.info("\n=== Summary Statistics ===")
        logger.info(f"Total unique texts processed: {len(self.processed_texts)}")
        
        # Category distribution
        category_counts = {}
        for item in all_data:
            category = item["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.info("\nCategory Distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {count}")
        
        # Source collection distribution
        source_counts = {}
        for item in all_data:
            source = item["source_collection"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("\nSource Collection Distribution:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {source}: {count}")
        
        # Text length statistics
        text_lengths = [item["text_length"] for item in all_data]
        if text_lengths:
            logger.info(f"\nText Length Statistics:")
            logger.info(f"  Average: {sum(text_lengths) / len(text_lengths):.2f} characters")
            logger.info(f"  Min: {min(text_lengths)} characters")
            logger.info(f"  Max: {max(text_lengths)} characters")

def main():
    """Main function."""
    creator = TopicModelingCollectionCreator(QDRANT_URL)
    creator.run()

if __name__ == "__main__":
    main()