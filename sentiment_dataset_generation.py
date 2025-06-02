import os
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import pandas as pd
import numpy as np

# Set to True to enable detailed logging
DEBUG = True

def log(message: str) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")

class SentimentCollectionBuilder:
    """Class to build a sentiment analysis collection from existing Qdrant collections."""
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6333,
        api_key: Optional[str] = None,
        vector_size: int = 384,
        new_collection_name: str = "sentiment_analysis_dataset",
        grpc_port: Optional[int] = None
    ):
        """Initialize the builder with connection parameters."""
        self.client = QdrantClient(
            host=host, 
            port=port,
            api_key=api_key,
            grpc_port=grpc_port
        )
        self.vector_size = vector_size
        self.new_collection_name = new_collection_name
        
        # Store statistics for reporting
        self.stats = {
            "processed_collections": 0,
            "total_records": 0,
            "records_per_collection": {},
            "skipped_records": 0,
        }
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        collections = self.client.get_collections().collections
        return [collection.name for collection in collections]
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a collection."""
        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
            points_count = self.client.count(collection_name=collection_name).count
            return {
                "name": collection_name,
                "vectors_count": points_count,
                "config": collection_info.config,
                "schema": self._get_collection_schema(collection_name)
            }
        except Exception as e:
            log(f"Error getting info for collection {collection_name}: {e}")
            return {"name": collection_name, "error": str(e)}
    
    def _get_collection_schema(self, collection_name: str) -> Dict[str, Any]:
        """Try to infer collection schema by sampling points."""
        try:
            # Get a sample point to analyze schema
            result = self.client.scroll(
                collection_name=collection_name,
                limit=1
            )
            
            # Handle both tuple return (newer versions) and object return (older versions)
            if isinstance(result, tuple):
                points, _ = result
            else:
                points = result.points
                
            if points and len(points) > 0:
                # Extract payload structure
                payload_schema = {}
                for key, value in points[0].payload.items():
                    payload_schema[key] = type(value).__name__
                return payload_schema
            return {}
        except Exception as e:
            log(f"Error inferring schema for {collection_name}: {e}")
            return {}
    
    def create_sentiment_collection(self) -> None:
        """Create a new collection for sentiment analysis."""
        # Check if collection already exists
        collections = self.list_collections()
        if self.new_collection_name in collections:
            log(f"Collection {self.new_collection_name} already exists. Recreating...")
            self.client.delete_collection(collection_name=self.new_collection_name)
        
        # Create the new collection with appropriate schema for sentiment analysis
        self.client.create_collection(
            collection_name=self.new_collection_name,
            vectors_config={
                "embedding": VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            }
        )
        
        log(f"Created new collection: {self.new_collection_name}")
    
    def _extract_comment_text(self, payload: Dict[str, Any]) -> Optional[str]:
        """Extract comment text from a payload with various possible structures."""
        # Try common field names for comment text
        for field in ["text", "comment", "content", "body", "message", "description", "title", "idea", "update"]:
            if field in payload:
                return payload[field]
        
        # Look for fields containing these terms
        for field in payload:
            if any(term in field.lower() for term in ["text", "comment", "content", "body", "message", "description"]):
                return payload[field]
        
        # Return the longest string value as a fallback
        text_fields = {k: v for k, v in payload.items() if isinstance(v, str) and len(v) > 10}
        if text_fields:
            return max(text_fields.items(), key=lambda x: len(x[1]))[1]
        
        return None
    
    def _extract_metadata(self, payload: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """Extract useful metadata from a payload."""
        metadata = {
            "source_collection": collection_name,
            "extracted_at": datetime.datetime.now().isoformat()
        }
        
        # Try to extract useful metadata fields
        for field in [
            "id", "user_id", "author", "date", "created_at", "timestamp", 
            "project_id", "category", "topic", "tags", "location", "language",
            "sentiment", "rating", "score", "votes", "likes", "dislikes"
        ]:
            if field in payload:
                metadata[field] = payload[field]
        
        # Extract specific fields based on collection type
        if "comments" in collection_name.lower():
            for field in ["article_id", "post_id", "parent_id", "reply_to"]:
                if field in payload:
                    metadata[field] = payload[field]
        
        if "project" in collection_name.lower():
            for field in ["status", "budget", "timeline", "priority", "municipality"]:
                if field in payload:
                    metadata[field] = payload[field]
        
        return metadata
    
    def _generate_embedding_placeholder(self) -> List[float]:
        """
        Generate a placeholder embedding vector.
        In a real implementation, this would use a text embedding model.
        """
        # In a real implementation, you would use a text embedding model here
        # For now, we'll just generate a random vector for demonstration
        return list(np.random.uniform(-1, 1, self.vector_size))
    
    def process_collection(self, collection_name: str, limit: int = None) -> int:
        """Process a collection and extract data for sentiment analysis."""
        log(f"Processing collection: {collection_name}")
        
        # Get points from the collection with pagination
        offset = None
        processed_count = 0
        batch_size = 1000
        
        while True:
            try:
                result = self.client.scroll(
                    collection_name=collection_name,
                    limit=min(batch_size, limit - processed_count if limit else batch_size),
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
                
                batch_data = []
                
                for point in points:
                    # Extract text content for sentiment analysis
                    text = self._extract_comment_text(point.payload)
                    if not text:
                        self.stats["skipped_records"] += 1
                        continue
                    
                    # Extract metadata
                    metadata = self._extract_metadata(point.payload, collection_name)
                    
                    # Generate point for the new collection
                    batch_data.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            payload={
                                "text": text,
                                "metadata": metadata,
                                "raw_payload": point.payload  # Store original payload for reference
                            },
                            vector={
                                "embedding": self._generate_embedding_placeholder()
                            }
                        )
                    )
                
                # Insert the batch into the new collection
                if batch_data:
                    self.client.upsert(
                        collection_name=self.new_collection_name,
                        points=batch_data
                    )
                
                processed_count += len(batch_data)
                log(f"Processed {processed_count} records from {collection_name}")
                
                # Update offset for next batch
                offset = next_page_offset
                if offset is None or (limit and processed_count >= limit):
                    break
                
            except Exception as e:
                log(f"Error processing {collection_name}: {e}")
                break
        
        # Update statistics
        self.stats["processed_collections"] += 1
        self.stats["total_records"] += processed_count
        self.stats["records_per_collection"][collection_name] = processed_count
        
        return processed_count
    
    def build_sentiment_collection(self, comment_collections: List[str], context_collections: List[str], limit_per_collection: Optional[int] = None) -> Dict[str, Any]:
        """
        Build the sentiment analysis collection from specified collections.
        
        Args:
            comment_collections: Primary collections containing comments/opinions
            context_collections: Secondary collections providing context
            limit_per_collection: Maximum records to process per collection
        """
        # Create the new collection
        self.create_sentiment_collection()
        
        # Process comment collections (prioritize these)
        for collection in tqdm(comment_collections, desc="Processing comment collections"):
            self.process_collection(collection, limit=limit_per_collection)
        
        # Process context collections (with lower priority/limit if needed)
        context_limit = limit_per_collection // 2 if limit_per_collection else None
        for collection in tqdm(context_collections, desc="Processing context collections"):
            self.process_collection(collection, limit=context_limit)
        
        # Print summary statistics
        summary = {
            "new_collection": self.new_collection_name,
            "total_records": self.stats["total_records"],
            "processed_collections": self.stats["processed_collections"],
            "records_per_collection": self.stats["records_per_collection"],
            "skipped_records": self.stats["skipped_records"]
        }
        
        log(f"Sentiment collection built successfully: {json.dumps(summary, indent=2)}")
        return summary

def create_sentiment_analysis_dataset(
    host: str = "localhost", 
    port: int = 6333,
    api_key: Optional[str] = None,
    limit_per_collection: Optional[int] = 5000  # Limit records per collection to keep dataset balanced
) -> None:
    """
    Create a comprehensive sentiment analysis dataset from existing collections.
    
    This function identifies relevant collections, processes them, and builds
    a new collection specifically for sentiment analysis training.
    """
    # Initialize the builder
    builder = SentimentCollectionBuilder(
        host=host,
        port=port,
        api_key=api_key,
        new_collection_name="sentiment_analysis_dataset"
    )
    
    # Define collection categories for processing
    # Primary collections contain direct sentiment information
    primary_collections = [
        "remacto_comments",         # Comments on Remacto projects
        "citizen_comments",         # General citizen comments
        "hespress_politics_comments" # Political comments
    ]
    
    # Secondary collections provide context and additional sentiment data
    secondary_collections = [
        "citizen_ideas",           # May contain opinions/sentiment
        "project_updates",         # May contain feedback
        "municipal_projects",      # For context about projects being commented on
        "remacto_projects"         # For context about Remacto projects
    ]
    
    # Build the sentiment analysis collection
    summary = builder.build_sentiment_collection(
        comment_collections=primary_collections,
        context_collections=secondary_collections,
        limit_per_collection=limit_per_collection
    )
    
    # Get info about the new collection
    collection_info = builder.get_collection_info("sentiment_analysis_dataset")
    
    print("\n" + "="*80)
    print(f"SENTIMENT ANALYSIS DATASET CREATION COMPLETE")
    print("="*80)
    print(f"Collection name: {collection_info['name']}")
    print(f"Total records: {collection_info['vectors_count']}")
    print(f"Records per source collection:")
    for collection, count in summary['records_per_collection'].items():
        print(f"  - {collection}: {count}")
    print("="*80)
    print("\nNext steps:")
    print("1. Use a text embedding model to replace placeholder embeddings with real embeddings")
    print("2. Add sentiment labels (positive, negative, neutral) for supervised learning")
    print("3. Split the dataset into training, validation, and test sets")
    print("="*80)

if __name__ == "__main__":
    # You can customize these parameters
    create_sentiment_analysis_dataset(
        host="localhost",  # Replace with your Qdrant host
        port=6333,         # Replace with your Qdrant port
        api_key=os.environ.get("QDRANT_API_KEY"),  # Set API key in environment variable
        limit_per_collection=5000  # Adjust as needed
    )