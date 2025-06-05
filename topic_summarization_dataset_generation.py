#!/usr/bin/env python3
"""
Script to create a comprehensive topic summarization collection from existing Qdrant collections.
This will aggregate text data suitable for training summarization models and LLMs.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    ScrollRequest, UpdateStatus
)
import hashlib
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qdrant connection details
QDRANT_URL = "http://154.44.186.241:6333"
NEW_COLLECTION_NAME = "topic_summarization_dataset"

# Collections to process with their text field mappings for summarization
COLLECTIONS_CONFIG = {
    "remacto_comments": {
        "full_text_fields": ["comment", "text", "content"],
        "summary_fields": ["summary", "brief"],
        "title_fields": ["title", "subject"],
        "metadata_fields": ["author", "date", "project_id", "sentiment", "rating"],
        "content_type": "user_feedback",
        "min_text_length": 100  # Minimum chars for summarization
    },
    "municipal_projects": {
        "full_text_fields": ["description", "objectives", "full_description", "details"],
        "summary_fields": ["summary", "brief", "abstract"],
        "title_fields": ["title", "project_name"],
        "metadata_fields": ["project_id", "status", "budget", "location", "department", "timeline"],
        "content_type": "project_documentation",
        "min_text_length": 200
    },
    "project_updates": {
        "full_text_fields": ["update", "description", "content", "details", "report"],
        "summary_fields": ["summary", "brief", "highlights"],
        "title_fields": ["title", "update_title"],
        "metadata_fields": ["project_id", "date", "status", "author", "update_type"],
        "content_type": "progress_reports",
        "min_text_length": 150
    },
    "hespress_politics_details": {
        "full_text_fields": ["content", "article", "body", "full_text"],
        "summary_fields": ["summary", "abstract", "lead"],
        "title_fields": ["title", "headline"],
        "metadata_fields": ["url", "date", "author", "tags", "category"],
        "content_type": "news_articles",
        "min_text_length": 300
    },
    "citizen_comments": {
        "full_text_fields": ["comment", "text", "content", "feedback", "message"],
        "summary_fields": ["summary", "brief"],
        "title_fields": ["subject", "topic"],
        "metadata_fields": ["citizen_id", "date", "topic", "sentiment", "location", "category"],
        "content_type": "citizen_feedback",
        "min_text_length": 100
    },
    "remacto_projects": {
        "full_text_fields": ["description", "goals", "content", "objectives", "details"],
        "summary_fields": ["summary", "overview", "abstract"],
        "title_fields": ["title", "project_name"],
        "metadata_fields": ["project_id", "status", "budget", "timeline", "priority"],
        "content_type": "project_proposals",
        "min_text_length": 200
    },
    "citizen_ideas": {
        "full_text_fields": ["idea", "description", "proposal", "content", "details"],
        "summary_fields": ["summary", "brief", "abstract"],
        "title_fields": ["title", "idea_title"],
        "metadata_fields": ["citizen_id", "date", "category", "votes", "status", "impact"],
        "content_type": "citizen_proposals",
        "min_text_length": 150
    },
    "citizens": {
        "full_text_fields": ["bio", "concerns", "background", "interests_detailed"],
        "summary_fields": ["summary", "profile_summary"],
        "title_fields": ["name", "profile_title"],
        "metadata_fields": ["citizen_id", "location", "age_group", "occupation"],
        "content_type": "citizen_profiles",
        "min_text_length": 100
    },
    "municipal_officials": {
        "full_text_fields": ["bio", "responsibilities", "platform", "background", "experience"],
        "summary_fields": ["summary", "brief_bio"],
        "title_fields": ["name", "position"],
        "metadata_fields": ["official_id", "department", "term", "party"],
        "content_type": "official_profiles",
        "min_text_length": 150
    },
    "budget_allocations": {
        "full_text_fields": ["description", "justification", "details", "rationale"],
        "summary_fields": ["summary", "brief"],
        "title_fields": ["title", "allocation_name"],
        "metadata_fields": ["department", "amount", "year", "category", "priority"],
        "content_type": "budget_documents",
        "min_text_length": 100
    },
    "hespress_politics_comments": {
        "full_text_fields": ["comment", "text", "content"],
        "summary_fields": ["summary"],
        "title_fields": ["subject"],
        "metadata_fields": ["article_id", "author", "date", "likes", "replies"],
        "content_type": "news_comments",
        "min_text_length": 80
    }
}

class TopicSummarizationCollectionCreator:
    def __init__(self, qdrant_url: str):
        """Initialize the collection creator with Qdrant client."""
        self.client = QdrantClient(url=qdrant_url)
        self.processed_texts = set()  # To avoid duplicates
        self.summarization_pairs = []  # Store source-summary pairs
        
    def create_collection(self, collection_name: str, vector_size: int = 384):
        """Create the new topic summarization collection."""
        try:
            # Delete collection if it exists
            collections = self.client.get_collections().collections
            if any(col.name == collection_name for col in collections):
                logger.info(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
            
            # Create new collection with named vectors for both full text and summary
            logger.info(f"Creating new collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "full_text": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    ),
                    "summary": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                },
                on_disk_payload=True
            )
            logger.info(f"Collection {collection_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def extract_text_content(self, payload: Dict[str, Any], field_list: List[str]) -> Optional[str]:
        """Extract text content from payload based on field list."""
        texts = []
        
        for field in field_list:
            if field in payload and payload[field]:
                text = str(payload[field]).strip()
                if text and len(text) > 10:
                    texts.append(text)
        
        # Combine all found texts
        if texts:
            return " ".join(texts)
        return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for summarization."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\"\']+', '', text)
        return text.strip()
    
    def estimate_summary_length(self, full_text_length: int) -> int:
        """Estimate appropriate summary length based on full text length."""
        if full_text_length < 500:
            return min(100, full_text_length // 2)
        elif full_text_length < 1000:
            return 150
        elif full_text_length < 2000:
            return 200
        else:
            return min(300, full_text_length // 5)
    
    def create_extractive_summary(self, text: str, target_length: int) -> str:
        """Create a simple extractive summary by taking first sentences."""
        sentences = re.split(r'[.!?]+', text)
        summary = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                summary.append(sentence)
                current_length += len(sentence)
                if current_length >= target_length:
                    break
        
        return '. '.join(summary) + '.' if summary else text[:target_length]
    
    def process_collection(self, collection_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single collection and extract summarization data."""
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
                    
                    # Extract full text
                    full_text = self.extract_text_content(point.payload, config["full_text_fields"])
                    
                    if full_text and len(full_text) >= config["min_text_length"]:
                        # Check for duplicates
                        text_hash = hashlib.md5(full_text.encode()).hexdigest()
                        if text_hash in self.processed_texts:
                            continue
                        
                        self.processed_texts.add(text_hash)
                        
                        # Clean text
                        full_text = self.clean_text(full_text)
                        
                        # Extract or generate summary
                        existing_summary = self.extract_text_content(point.payload, config["summary_fields"])
                        title = self.extract_text_content(point.payload, config["title_fields"])
                        
                        # Create summary if not exists
                        if existing_summary:
                            summary = self.clean_text(existing_summary)
                        elif title and len(title) > 20:
                            # Use title as a very short summary
                            summary = self.clean_text(title)
                        else:
                            # Generate extractive summary
                            target_length = self.estimate_summary_length(len(full_text))
                            summary = self.create_extractive_summary(full_text, target_length)
                        
                        # Extract metadata
                        metadata = {}
                        for field in config["metadata_fields"]:
                            if field in point.payload and point.payload[field] is not None:
                                metadata[field] = point.payload[field]
                        
                        # Prepare data point
                        data_point = {
                            "id": str(uuid.uuid4()),
                            "full_text": full_text,
                            "summary": summary,
                            "title": title if title else "",
                            "source_collection": collection_name,
                            "content_type": config["content_type"],
                            "metadata": metadata,
                            "full_text_length": len(full_text),
                            "summary_length": len(summary),
                            "compression_ratio": len(summary) / len(full_text),
                            "word_count_full": len(full_text.split()),
                            "word_count_summary": len(summary.split()),
                            "has_existing_summary": bool(existing_summary),
                            "original_id": str(point.id),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Handle vectors
                        if point.vector:
                            if isinstance(point.vector, dict):
                                # Use existing vectors if available
                                if "size" in point.vector:
                                    data_point["full_text_vector"] = point.vector["size"]
                                    data_point["summary_vector"] = point.vector["size"]  # Use same for now
                                else:
                                    first_key = next(iter(point.vector))
                                    data_point["full_text_vector"] = point.vector[first_key]
                                    data_point["summary_vector"] = point.vector[first_key]
                            else:
                                data_point["full_text_vector"] = point.vector
                                data_point["summary_vector"] = point.vector
                        
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
                "full_text": item["full_text"],
                "summary": item["summary"],
                "title": item["title"],
                "source_collection": item["source_collection"],
                "content_type": item["content_type"],
                "full_text_length": item["full_text_length"],
                "summary_length": item["summary_length"],
                "compression_ratio": item["compression_ratio"],
                "word_count_full": item["word_count_full"],
                "word_count_summary": item["word_count_summary"],
                "has_existing_summary": item["has_existing_summary"],
                "original_id": item["original_id"],
                "timestamp": item["timestamp"]
            }
            
            # Add metadata fields
            if item["metadata"]:
                payload["metadata"] = json.dumps(item["metadata"])
            
            # Prepare vectors
            vectors = {}
            if "full_text_vector" in item:
                vectors["full_text"] = item["full_text_vector"]
            else:
                vectors["full_text"] = [0.0] * 384  # Zero vector if not available
            
            if "summary_vector" in item:
                vectors["summary"] = item["summary_vector"]
            else:
                vectors["summary"] = [0.0] * 384
            
            # Create point
            point = PointStruct(
                id=item["id"],
                vector=vectors,
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
        logger.info("Starting topic summarization collection creation process...")
        
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
        
        # Content type distribution
        content_type_counts = {}
        for item in all_data:
            content_type = item["content_type"]
            content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        logger.info("\nContent Type Distribution:")
        for content_type, count in sorted(content_type_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {content_type}: {count}")
        
        # Summary statistics
        with_existing_summary = sum(1 for item in all_data if item["has_existing_summary"])
        logger.info(f"\nSummary Sources:")
        logger.info(f"  With existing summaries: {with_existing_summary}")
        logger.info(f"  Generated summaries: {len(all_data) - with_existing_summary}")
        
        # Text length statistics
        full_lengths = [item["full_text_length"] for item in all_data]
        summary_lengths = [item["summary_length"] for item in all_data]
        compression_ratios = [item["compression_ratio"] for item in all_data]
        
        if full_lengths:
            logger.info(f"\nFull Text Length Statistics:")
            logger.info(f"  Average: {sum(full_lengths) / len(full_lengths):.2f} characters")
            logger.info(f"  Min: {min(full_lengths)} characters")
            logger.info(f"  Max: {max(full_lengths)} characters")
            
            logger.info(f"\nSummary Length Statistics:")
            logger.info(f"  Average: {sum(summary_lengths) / len(summary_lengths):.2f} characters")
            logger.info(f"  Min: {min(summary_lengths)} characters")
            logger.info(f"  Max: {max(summary_lengths)} characters")
            
            logger.info(f"\nCompression Ratio Statistics:")
            logger.info(f"  Average: {sum(compression_ratios) / len(compression_ratios):.2%}")
            logger.info(f"  Min: {min(compression_ratios):.2%}")
            logger.info(f"  Max: {max(compression_ratios):.2%}")
        
        # Source collection distribution
        source_counts = {}
        for item in all_data:
            source = item["source_collection"]
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("\nSource Collection Distribution:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {source}: {count}")

def main():
    """Main function."""
    creator = TopicSummarizationCollectionCreator(QDRANT_URL)
    creator.run()

if __name__ == "__main__":
    main()