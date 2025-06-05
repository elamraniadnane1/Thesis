#!/usr/bin/env python3
"""
Qdrant to MongoDB Backup System
Creates comprehensive backups of Qdrant collections and stores them in MongoDB
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
import gzip
import base64

from qdrant_client import QdrantClient
from qdrant_client.models import ScrollRequest
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, BulkWriteError
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Connection details
QDRANT_URL = "http://154.44.186.241:6333"
MONGODB_HOST = "154.44.186.241"
MONGODB_PORT = 27017
MONGODB_DATABASE = "qdrant_backups"

class QdrantMongoBackup:
    def __init__(self, qdrant_url: str, mongodb_host: str, mongodb_port: int, db_name: str = "qdrant_backups"):
        """Initialize backup system with Qdrant and MongoDB clients."""
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.mongodb_client = None
        self.db = None
        self.mongodb_host = mongodb_host
        self.mongodb_port = mongodb_port
        self.db_name = db_name
        
        # Connect to MongoDB
        self._connect_mongodb()
        
    def _connect_mongodb(self):
        """Establish connection to MongoDB."""
        connection_string = f"mongodb://{self.mongodb_host}:{self.mongodb_port}/"
        
        try:
            self.mongodb_client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.mongodb_client.admin.command('ismaster')
            self.db = self.mongodb_client[self.db_name]
            logger.info(f"✅ Connected to MongoDB at {self.mongodb_host}:{self.mongodb_port}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            sys.exit(1)
    
    def _compress_vector(self, vector: List[float]) -> str:
        """Compress vector data for storage efficiency."""
        # Convert to numpy array, then to bytes, compress, and encode to base64
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        compressed = gzip.compress(vector_bytes)
        return base64.b64encode(compressed).decode('utf-8')
    
    def _decompress_vector(self, compressed_vector: str, size: int) -> List[float]:
        """Decompress vector data."""
        compressed = base64.b64decode(compressed_vector.encode('utf-8'))
        vector_bytes = gzip.decompress(compressed)
        return np.frombuffer(vector_bytes, dtype=np.float32).tolist()
    
    def backup_collection(self, collection_name: str) -> Dict[str, Any]:
        """Backup a single Qdrant collection to MongoDB."""
        logger.info(f"Starting backup of collection: {collection_name}")
        
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection(collection_name)
            
            # Create backup metadata
            backup_metadata = {
                "collection_name": collection_name,
                "timestamp": datetime.now(),
                "points_count": collection_info.points_count,
                "vectors_config": str(collection_info.config.params.vectors),
                "shard_number": collection_info.config.params.shard_number,
                "replication_factor": collection_info.config.params.replication_factor,
                "on_disk_payload": collection_info.config.params.on_disk_payload
            }
            
            # Create collection in MongoDB
            mongo_collection = self.db[f"backup_{collection_name}"]
            
            # Create indexes for efficient queries
            mongo_collection.create_index([("point_id", ASCENDING)], unique=True)
            mongo_collection.create_index([("backup_timestamp", DESCENDING)])
            
            # Backup points in batches
            offset = None
            batch_size = 100
            total_backed_up = 0
            
            while True:
                # Scroll through points
                result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                points, next_offset = result
                
                if not points:
                    break
                
                # Prepare batch for MongoDB
                batch_docs = []
                
                for point in points:
                    doc = {
                        "point_id": str(point.id),
                        "backup_timestamp": backup_metadata["timestamp"],
                        "payload": point.payload if point.payload else {}
                    }
                    
                    # Handle vectors
                    if point.vector:
                        if isinstance(point.vector, dict):
                            # Named vectors
                            doc["vectors"] = {}
                            for vec_name, vec_data in point.vector.items():
                                if isinstance(vec_data, list):
                                    doc["vectors"][vec_name] = {
                                        "data": self._compress_vector(vec_data),
                                        "size": len(vec_data)
                                    }
                        else:
                            # Single vector
                            doc["vector"] = {
                                "data": self._compress_vector(point.vector),
                                "size": len(point.vector)
                            }
                    
                    batch_docs.append(doc)
                
                # Insert batch into MongoDB
                if batch_docs:
                    try:
                        mongo_collection.insert_many(batch_docs, ordered=False)
                        total_backed_up += len(batch_docs)
                    except BulkWriteError as e:
                        # Handle duplicate key errors
                        successful = e.details.get('nInserted', 0)
                        total_backed_up += successful
                        logger.warning(f"Partial batch insert: {successful} out of {len(batch_docs)} documents")
                
                logger.info(f"Backed up {total_backed_up} points so far...")
                
                offset = next_offset
                if offset is None:
                    break
            
            # Store backup metadata
            metadata_collection = self.db["backup_metadata"]
            metadata_collection.insert_one(backup_metadata)
            
            logger.info(f"✅ Successfully backed up {total_backed_up} points from {collection_name}")
            
            return {
                "collection_name": collection_name,
                "points_backed_up": total_backed_up,
                "timestamp": backup_metadata["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error backing up collection {collection_name}: {e}")
            return {
                "collection_name": collection_name,
                "error": str(e)
            }
    
    def backup_all_collections(self) -> List[Dict[str, Any]]:
        """Backup all Qdrant collections to MongoDB."""
        logger.info("Starting backup of all collections...")
        
        # Get all collections
        collections = self.qdrant_client.get_collections().collections
        results = []
        
        for collection in collections:
            result = self.backup_collection(collection.name)
            results.append(result)
        
        # Create summary
        summary = {
            "backup_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now(),
            "total_collections": len(collections),
            "successful_backups": sum(1 for r in results if "error" not in r),
            "failed_backups": sum(1 for r in results if "error" in r),
            "collections": results
        }
        
        # Store summary
        self.db["backup_summaries"].insert_one(summary)
        
        return results
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        summaries = list(self.db["backup_summaries"].find().sort("timestamp", -1))
        
        for summary in summaries:
            summary["_id"] = str(summary["_id"])
        
        return summaries
    
    def restore_collection(self, collection_name: str, backup_timestamp: Optional[datetime] = None) -> bool:
        """Restore a collection from MongoDB backup."""
        logger.info(f"Starting restore of collection: {collection_name}")
        
        try:
            # Get backup metadata
            metadata_query = {"collection_name": collection_name}
            if backup_timestamp:
                metadata_query["timestamp"] = backup_timestamp
            
            metadata = self.db["backup_metadata"].find_one(
                metadata_query,
                sort=[("timestamp", -1)]
            )
            
            if not metadata:
                logger.error(f"No backup found for collection {collection_name}")
                return False
            
            # Delete existing collection if it exists
            try:
                self.qdrant_client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except:
                pass
            
            # Parse vectors config
            import ast
            vectors_config = ast.literal_eval(metadata["vectors_config"])
            
            # Recreate collection
            # Note: This is simplified - you may need to handle complex vector configs
            from qdrant_client.models import Distance, VectorParams
            
            if isinstance(vectors_config, dict):
                # Named vectors
                vector_params = {}
                for name, config in vectors_config.items():
                    vector_params[name] = VectorParams(
                        size=config['size'],
                        distance=Distance.COSINE
                    )
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vector_params,
                    on_disk_payload=metadata.get("on_disk_payload", True)
                )
            else:
                # Single vector
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vectors_config['size'],
                        distance=Distance.COSINE
                    ),
                    on_disk_payload=metadata.get("on_disk_payload", True)
                )
            
            # Restore points
            mongo_collection = self.db[f"backup_{collection_name}"]
            query = {"backup_timestamp": metadata["timestamp"]}
            
            batch_size = 100
            batch = []
            restored_count = 0
            
            for doc in mongo_collection.find(query):
                # Prepare point
                from qdrant_client.models import PointStruct
                
                # Decompress vectors
                vectors = None
                if "vectors" in doc:
                    # Named vectors
                    vectors = {}
                    for vec_name, vec_info in doc["vectors"].items():
                        vectors[vec_name] = self._decompress_vector(
                            vec_info["data"],
                            vec_info["size"]
                        )
                elif "vector" in doc:
                    # Single vector
                    vectors = self._decompress_vector(
                        doc["vector"]["data"],
                        doc["vector"]["size"]
                    )
                
                point = PointStruct(
                    id=doc["point_id"],
                    vector=vectors,
                    payload=doc["payload"]
                )
                batch.append(point)
                
                # Insert batch when full
                if len(batch) >= batch_size:
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                    restored_count += len(batch)
                    logger.info(f"Restored {restored_count} points...")
                    batch = []
            
            # Insert remaining points
            if batch:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                restored_count += len(batch)
            
            logger.info(f"✅ Successfully restored {restored_count} points to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error restoring collection {collection_name}: {e}")
            return False
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get statistics about backups."""
        stats = {
            "total_backups": self.db["backup_summaries"].count_documents({}),
            "collections_backed_up": len(self.db.list_collection_names()),
            "total_size_mb": 0,
            "collections": {}
        }
        
        # Get size of each collection
        for collection_name in self.db.list_collection_names():
            if collection_name.startswith("backup_"):
                collection_stats = self.db.command("collStats", collection_name)
                size_mb = collection_stats.get("size", 0) / (1024 * 1024)
                stats["total_size_mb"] += size_mb
                stats["collections"][collection_name] = {
                    "size_mb": round(size_mb, 2),
                    "count": collection_stats.get("count", 0)
                }
        
        return stats

def main():
    """Main function to run backup operations."""
    backup_system = QdrantMongoBackup(
        qdrant_url=QDRANT_URL,
        mongodb_host=MONGODB_HOST,
        mongodb_port=MONGODB_PORT
    )
    
    # Create a full backup
    logger.info("=== Starting Full Qdrant Backup ===")
    results = backup_system.backup_all_collections()
    
    # Print results
    logger.info("\n=== Backup Results ===")
    successful = sum(1 for r in results if "error" not in r)
    failed = sum(1 for r in results if "error" in r)
    
    logger.info(f"Total collections: {len(results)}")
    logger.info(f"Successful backups: {successful}")
    logger.info(f"Failed backups: {failed}")
    
    # Print individual results
    for result in results:
        if "error" in result:
            logger.error(f"❌ {result['collection_name']}: {result['error']}")
        else:
            logger.info(f"✅ {result['collection_name']}: {result['points_backed_up']} points backed up")
    
    # Get backup statistics
    stats = backup_system.get_backup_stats()
    logger.info(f"\n=== Backup Statistics ===")
    logger.info(f"Total backup size: {stats['total_size_mb']:.2f} MB")
    logger.info(f"Number of backup runs: {stats['total_backups']}")
    
    # Example: List available backups
    logger.info("\n=== Available Backups ===")
    backups = backup_system.list_backups()
    for backup in backups[:5]:  # Show last 5 backups
        logger.info(f"Backup ID: {backup['backup_id']} - Time: {backup['timestamp']}")

if __name__ == "__main__":
    main()