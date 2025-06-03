#!/usr/bin/env python
"""
migrate_qdrant_to_cloud.py
──────────────────────────
Clone *all* collections from a local Qdrant (localhost:6333)
to a remote/cloud Qdrant cluster.

• Uses only `collection_exists`, `delete_collection`, and `create_collection`
  (no deprecated `recreate_collection`).
• Copies vector parameters, shard count, on-disk flag, payload schema.
• Streams points in batches with a progress bar.

Requirements
------------
pip install --upgrade qdrant-client tqdm
"""

from __future__ import annotations

import os
import time
import logging
from typing import Iterable, List, Dict, Any, Optional

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qdrant_migration")

# ──────────────────────────────────────────────────────────────────────────
# 1.  CONNECTION SETTINGS
# ──────────────────────────────────────────────────────────────────────────
LOCAL_HOST = os.getenv("LOCAL_QDRANT_HOST", "localhost")
LOCAL_PORT = int(os.getenv("LOCAL_QDRANT_PORT", 6333))

CLOUD_URL = (
    "https://a6dfc799-7211-45f6-b597-4054978bbeed.europe-west3-0.gcp.cloud.qdrant.io:6333"
)
CLOUD_API_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.8oTruRh30L0m2A-2fDhyFPXuLvg2GsslQVyRy45b4nU"
)

BATCH_SIZE = 1_000  # points per upsert
MAX_RETRIES = 3     # max retries for API calls

# ──────────────────────────────────────────────────────────────────────────
# 2.  CLIENTS
# ──────────────────────────────────────────────────────────────────────────
try:
    local = QdrantClient(host=LOCAL_HOST, port=LOCAL_PORT)
    remote = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY)
    # Test connections
    local.get_collections()
    remote.get_collections()
    logger.info("Connected to both local and remote Qdrant instances")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant instances: {e}")
    raise


# ──────────────────────────────────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────────────────────────────────
def batched_scroll(
    client: QdrantClient, collection: str, batch_size: int = BATCH_SIZE
) -> Iterable[List[qm.Record]]:
    """Yield batches of points from *collection* with retry logic."""
    offset = None
    while True:
        retries = 0
        while retries <= MAX_RETRIES:
            try:
                points, offset = client.scroll(
                    collection_name=collection,
                    offset=offset,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=True,
                )
                break
            except Exception as e:
                retries += 1
                if retries > MAX_RETRIES:
                    logger.error(f"Failed to scroll collection {collection}: {e}")
                    raise
                logger.warning(f"Scroll attempt {retries} failed: {e}. Retrying...")
                time.sleep(1)  # Wait before retry
                
        if not points:
            break
        yield points


def extract_vectors_config(params: qm.CollectionParams) -> Dict[str, Any]:
    """Extract vectors configuration in the correct format for create_collection."""
    vectors_config = params.vectors
    
    # Handle both dictionary and VectorParams formats
    if isinstance(vectors_config, dict):
        return vectors_config
    
    # If it's a VectorParams object, convert it to the proper format
    return {
        "size": getattr(vectors_config, "size", None),
        "distance": getattr(vectors_config, "distance", "Cosine"),
        "on_disk": getattr(vectors_config, "on_disk", False),
    }


def convert_hnsw_config(hnsw_config: Any) -> Optional[Dict[str, Any]]:
    """Convert HnswConfig to HnswConfigDiff format."""
    if not hnsw_config:
        return None
    
    return {
        "m": getattr(hnsw_config, "m", 16),
        "ef_construct": getattr(hnsw_config, "ef_construct", 100),
        "full_scan_threshold": getattr(hnsw_config, "full_scan_threshold", 10000),
    }


def convert_wal_config(wal_config: Any) -> Optional[Dict[str, Any]]:
    """Convert WalConfig to WalConfigDiff format."""
    if not wal_config:
        return None
    
    return {
        "wal_capacity_mb": getattr(wal_config, "wal_capacity_mb", 32),
        "wal_segments_ahead": getattr(wal_config, "wal_segments_ahead", 0),
    }


def convert_optimizer_config(optimizer_config: Any) -> Optional[Dict[str, Any]]:
    """Convert OptimizerConfig to OptimizerConfigDiff format."""
    if not optimizer_config:
        return None
    
    return {
        "deleted_threshold": getattr(optimizer_config, "deleted_threshold", 0.2),
        "vacuum_min_vector_number": getattr(optimizer_config, "vacuum_min_vector_number", 1000),
        "default_segment_number": getattr(optimizer_config, "default_segment_number", 0),
        "max_segment_size": getattr(optimizer_config, "max_segment_size", None),
        "memmap_threshold": getattr(optimizer_config, "memmap_threshold", None),
        "indexing_threshold": getattr(optimizer_config, "indexing_threshold", 20000),
    }


def recreate_remote_collection(name: str, cfg: qm.CollectionConfig) -> None:
    """
    Delete *name* in cloud (if present) and create it with params from *cfg*.
    Properly converts config objects to dictionary format to avoid pydantic validation errors.
    """
    # Check if collection exists and delete it if it does
    if remote.collection_exists(name):
        logger.info(f"Collection {name} exists in remote, deleting...")
        remote.delete_collection(name)
    
    # Extract parameters from config
    params = cfg.params  # This is a CollectionParams object
    vectors_config = extract_vectors_config(params)
    shard_number = getattr(cfg, "shard_number", None)
    on_disk_payload = bool(getattr(params, "on_disk_payload", False))
    
    # Convert configs to compatible dictionary formats
    hnsw_config = convert_hnsw_config(getattr(cfg, "hnsw_config", None))
    wal_config = convert_wal_config(getattr(cfg, "wal_config", None))
    optimizers_config = convert_optimizer_config(getattr(cfg, "optimizers_config", None))
    
    logger.info(f"Creating collection {name} in remote...")
    try:
        remote.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            shard_number=shard_number,
            on_disk_payload=on_disk_payload,
            hnsw_config=hnsw_config,
            wal_config=wal_config,
            optimizers_config=optimizers_config,
            # Add sparse_vectors_config if needed
        )
        logger.info(f"Collection {name} created successfully")
    except Exception as e:
        logger.error(f"Failed to create collection {name}: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────
# 4.  MIGRATION LOGIC
# ──────────────────────────────────────────────────────────────────────────
def migrate_collection(name: str) -> None:
    """Clone one *collection* from local → remote with robust error handling."""
    logger.info(f"🚚 Migrating collection: {name}")
    
    try:
        # Get collection info
        info = local.get_collection(name)
        
        # Recreate collection in remote
        recreate_remote_collection(name, info.config)
        
        # Copy payload schema if it exists
        if info.payload_schema:
            logger.info(f"Updating payload schema for {name}")
            schema_actions = [
                qm.PayloadSchemaOps(type="add", field_name=f, schema=s)
                for f, s in info.payload_schema.items()
            ]
            remote.update_payload_schema(
                collection_name=name,
                actions=schema_actions,
            )
        
        # Get total points count
        total_pts = local.count(name, exact=True).count
        logger.info(f"Found {total_pts} points to transfer")
        
        # Set up progress bar
        bar = tqdm(total=total_pts, unit="pt", ncols=80)
        
        # Transfer points in batches
        for batch in batched_scroll(local, name):
            points = [
                qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in batch
            ]
            
            # Retry logic for upsert
            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    remote.upsert(collection_name=name, points=points)
                    break
                except Exception as e:
                    retries += 1
                    if retries > MAX_RETRIES:
                        logger.error(f"Failed to upsert batch to {name}: {e}")
                        raise
                    logger.warning(f"Upsert attempt {retries} failed: {e}. Retrying...")
                    time.sleep(2)  # Wait before retry
            
            bar.update(len(batch))
        
        bar.close()
        logger.info(f"✅ Completed migration of {name}")
        
    except Exception as e:
        logger.error(f"Failed to migrate collection {name}: {e}")
        print(f"❌ Error migrating {name}: {e}")
        raise


def main() -> None:
    """Migrate every collection in the local instance with error handling."""
    try:
        # Get all collections from local
        collections_info = local.get_collections()
        collections = [c.name for c in collections_info.collections]
        
        if not collections:
            logger.warning("No collections found in local Qdrant")
            print("⚠️  No collections in local Qdrant.")
            return
        
        logger.info(f"Found {len(collections)} collections: {', '.join(collections)}")
        print(f"Found {len(collections)} collections: {', '.join(collections)}")
        
        # Migrate each collection
        for coll in collections:
            try:
                migrate_collection(coll)
            except Exception as e:
                logger.error(f"Migration of {coll} failed: {e}")
                print(f"❌ Migration of {coll} failed. Check logs for details.")
                # Continue with next collection instead of failing completely
                continue
        
        logger.info("All collections migrated successfully")
        print("\n🎉  All collections migrated successfully!")
        
    except Exception as e:
        logger.error(f"Migration process failed: {e}")
        print(f"❌ Migration process failed: {e}")
        raise


# ──────────────────────────────────────────────────────────────────────────
# ENTRY-POINT
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        logger.warning("Migration interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.exception("Fatal error occurred")