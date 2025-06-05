#!/usr/bin/env python3
"""
Qdrant Backup Diagnostics Tool
Diagnose and fix issues with backups in MongoDB
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import json

# Connection details
MONGODB_HOST = "154.44.186.241"
MONGODB_PORT = 27017
MONGODB_DATABASE = "qdrant_backups"

class QdrantBackupDiagnostics:
    def __init__(self, mongodb_host: str, mongodb_port: int, db_name: str = "qdrant_backups"):
        """Initialize diagnostics tool."""
        self.mongodb_host = mongodb_host
        self.mongodb_port = mongodb_port
        self.db_name = db_name
        self.mongodb_client = None
        self.db = None
        
        # Connect to MongoDB
        self._connect_mongodb()
    
    def _connect_mongodb(self):
        """Establish connection to MongoDB."""
        connection_string = f"mongodb://{self.mongodb_host}:{self.mongodb_port}/"
        
        try:
            print(f"Connecting to MongoDB at {self.mongodb_host}:{self.mongodb_port}...")
            self.mongodb_client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.mongodb_client.admin.command('ismaster')
            self.db = self.mongodb_client[self.db_name]
            print("✅ Connected to MongoDB successfully!\n")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            sys.exit(1)
    
    def check_database_status(self):
        """Check overall database status."""
        print("=" * 60)
        print("DATABASE STATUS")
        print("=" * 60)
        
        # List all collections
        collections = self.db.list_collection_names()
        print(f"Total collections: {len(collections)}")
        
        # Separate backup collections from metadata
        backup_collections = [c for c in collections if c.startswith("backup_")]
        metadata_collections = [c for c in collections if not c.startswith("backup_")]
        
        print(f"Backup collections: {len(backup_collections)}")
        print(f"Metadata collections: {len(metadata_collections)}")
        
        # Check database size
        db_stats = self.db.command("dbStats")
        print(f"Database size: {db_stats.get('dataSize', 0) / (1024*1024):.2f} MB")
        print(f"Storage size: {db_stats.get('storageSize', 0) / (1024*1024):.2f} MB")
        print()
    
    def analyze_backup_summaries(self):
        """Analyze backup summaries."""
        print("=" * 60)
        print("BACKUP SUMMARIES ANALYSIS")
        print("=" * 60)
        
        summaries = list(self.db["backup_summaries"].find().sort("timestamp", -1))
        
        if not summaries:
            print("❌ No backup summaries found!")
            return
        
        print(f"Total backup runs: {len(summaries)}")
        
        # Show recent backups
        print("\nRecent backups:")
        for i, summary in enumerate(summaries[:5]):
            print(f"\n{i+1}. Backup ID: {summary['backup_id']}")
            print(f"   Timestamp: {summary['timestamp']}")
            print(f"   Collections: {summary.get('successful_backups', 0)} successful, "
                  f"{summary.get('failed_backups', 0)} failed")
            
            # Check timestamp types in collections
            if 'collections' in summary:
                for col in summary['collections'][:3]:  # Show first 3
                    if 'timestamp' in col:
                        print(f"   - {col['collection_name']}: timestamp type = {type(col['timestamp'])}")
    
    def diagnose_collection(self, collection_name: str):
        """Diagnose a specific backup collection."""
        print(f"\n{'='*60}")
        print(f"DIAGNOSING: backup_{collection_name}")
        print("="*60)
        
        mongo_collection = self.db[f"backup_{collection_name}"]
        
        # Get total documents
        total_docs = mongo_collection.count_documents({})
        print(f"Total documents: {total_docs}")
        
        if total_docs == 0:
            print("❌ Collection is empty!")
            return
        
        # Analyze timestamps
        print("\nTimestamp Analysis:")
        
        # Get sample documents
        samples = list(mongo_collection.find().limit(10))
        
        # Check timestamp field existence and types
        timestamp_types = defaultdict(int)
        timestamp_values = []
        
        for doc in samples:
            if "backup_timestamp" in doc:
                ts = doc["backup_timestamp"]
                timestamp_types[type(ts).__name__] += 1
                timestamp_values.append(ts)
        
        print(f"Timestamp field types found:")
        for ts_type, count in timestamp_types.items():
            print(f"  - {ts_type}: {count}")
        
        # Show sample timestamps
        if timestamp_values:
            print(f"\nSample timestamps:")
            for i, ts in enumerate(timestamp_values[:3]):
                print(f"  {i+1}. {ts} (type: {type(ts).__name__})")
        
        # Group by timestamp
        print("\nDocuments grouped by timestamp:")
        pipeline = [
            {"$group": {"_id": "$backup_timestamp", "count": {"$sum": 1}}},
            {"$sort": {"_id": -1}},
            {"$limit": 5}
        ]
        
        groups = list(mongo_collection.aggregate(pipeline))
        for group in groups:
            print(f"  - {group['_id']}: {group['count']} documents")
    
    def fix_timestamp_types(self):
        """Fix timestamp type mismatches."""
        print("\n" + "="*60)
        print("FIXING TIMESTAMP TYPES")
        print("="*60)
        
        fixed_count = 0
        
        # Fix backup summaries
        print("\nFixing backup_summaries collection...")
        summaries = self.db["backup_summaries"].find({})
        
        for summary in summaries:
            updates = {}
            
            # Check main timestamp
            if "timestamp" in summary and isinstance(summary["timestamp"], str):
                try:
                    updates["timestamp"] = datetime.fromisoformat(summary["timestamp"])
                except:
                    pass
            
            # Check collection timestamps
            if "collections" in summary:
                new_collections = []
                for col in summary["collections"]:
                    if "timestamp" in col and isinstance(col["timestamp"], str):
                        try:
                            col["timestamp"] = datetime.fromisoformat(col["timestamp"])
                        except:
                            pass
                    new_collections.append(col)
                updates["collections"] = new_collections
            
            if updates:
                self.db["backup_summaries"].update_one(
                    {"_id": summary["_id"]},
                    {"$set": updates}
                )
                fixed_count += 1
        
        print(f"✅ Fixed {fixed_count} summary documents")
        
        # Fix metadata
        print("\nFixing backup_metadata collection...")
        fixed_count = 0
        metadata_docs = self.db["backup_metadata"].find({})
        
        for doc in metadata_docs:
            if "timestamp" in doc and isinstance(doc["timestamp"], str):
                try:
                    new_timestamp = datetime.fromisoformat(doc["timestamp"])
                    self.db["backup_metadata"].update_one(
                        {"_id": doc["_id"]},
                        {"$set": {"timestamp": new_timestamp}}
                    )
                    fixed_count += 1
                except:
                    pass
        
        print(f"✅ Fixed {fixed_count} metadata documents")
    
    def verify_backup_integrity(self, backup_id: str):
        """Verify integrity of a specific backup."""
        print(f"\n{'='*60}")
        print(f"VERIFYING BACKUP: {backup_id}")
        print("="*60)
        
        # Find backup summary
        summary = self.db["backup_summaries"].find_one({"backup_id": backup_id})
        
        if not summary:
            print(f"❌ Backup {backup_id} not found!")
            return
        
        print(f"Backup timestamp: {summary['timestamp']}")
        print(f"Expected collections: {summary.get('successful_backups', 0)}")
        
        # Check each collection
        issues = []
        
        for col_info in summary.get('collections', []):
            if 'error' in col_info:
                continue
            
            collection_name = col_info['collection_name']
            expected_points = col_info.get('points_backed_up', 0)
            
            # Count actual points
            mongo_collection = self.db[f"backup_{collection_name}"]
            
            # Try different timestamp queries
            timestamp = col_info.get('timestamp', summary['timestamp'])
            
            # Exact match
            actual_count = mongo_collection.count_documents({"backup_timestamp": timestamp})
            
            # Range match (within same day)
            if actual_count == 0 and isinstance(timestamp, datetime):
                start_of_day = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = timestamp.replace(hour=23, minute=59, second=59, microsecond=999999)
                actual_count = mongo_collection.count_documents({
                    "backup_timestamp": {"$gte": start_of_day, "$lte": end_of_day}
                })
            
            print(f"\n{collection_name}:")
            print(f"  Expected: {expected_points} points")
            print(f"  Actual: {actual_count} points")
            
            if actual_count != expected_points:
                issues.append(f"{collection_name}: expected {expected_points}, found {actual_count}")
                print(f"  ❌ Mismatch detected!")
            else:
                print(f"  ✅ OK")
        
        if issues:
            print(f"\n⚠️  Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ Backup integrity verified!")
    
    def run_full_diagnostics(self):
        """Run complete diagnostics."""
        print("\n🔍 RUNNING FULL DIAGNOSTICS\n")
        
        # Check database status
        self.check_database_status()
        
        # Analyze backup summaries
        self.analyze_backup_summaries()
        
        # Get sample collections to diagnose
        backup_collections = [c for c in self.db.list_collection_names() if c.startswith("backup_")]
        
        if backup_collections:
            print(f"\nDiagnosing sample collections...")
            for collection in backup_collections[:3]:  # Diagnose first 3
                self.diagnose_collection(collection.replace("backup_", ""))
        
        # Check latest backup
        latest_summary = self.db["backup_summaries"].find_one(sort=[("timestamp", -1)])
        if latest_summary:
            self.verify_backup_integrity(latest_summary["backup_id"])
        
        print("\n" + "="*60)
        print("DIAGNOSTICS COMPLETE")
        print("="*60)

def main():
    """Main function."""
    diagnostics = QdrantBackupDiagnostics(
        mongodb_host=MONGODB_HOST,
        mongodb_port=MONGODB_PORT
    )
    
    print("🔧 Qdrant Backup Diagnostics Tool")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Run full diagnostics")
        print("2. Check database status")
        print("3. Analyze backup summaries")
        print("4. Diagnose specific collection")
        print("5. Verify backup integrity")
        print("6. Fix timestamp types")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            diagnostics.run_full_diagnostics()
        elif choice == '2':
            diagnostics.check_database_status()
        elif choice == '3':
            diagnostics.analyze_backup_summaries()
        elif choice == '4':
            collection = input("Enter collection name (without 'backup_' prefix): ").strip()
            if collection:
                diagnostics.diagnose_collection(collection)
        elif choice == '5':
            backup_id = input("Enter backup ID: ").strip()
            if backup_id:
                diagnostics.verify_backup_integrity(backup_id)
        elif choice == '6':
            confirm = input("Fix timestamp type mismatches? (yes/no): ").strip().lower()
            if confirm == 'yes':
                diagnostics.fix_timestamp_types()
        elif choice == '7':
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()