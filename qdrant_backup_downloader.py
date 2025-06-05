#!/usr/bin/env python3
"""
Qdrant Backup Downloader
Browse and download Qdrant backups from MongoDB as zip files with progress indication
"""

import os
import json
import zipfile
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
from tqdm import tqdm
import gzip
import base64

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Connection details
MONGODB_HOST = "154.44.186.241"
MONGODB_PORT = 27017
MONGODB_DATABASE = "qdrant_backups"
DOWNLOAD_DIR = os.path.expanduser("~/qdrant_backup_downloads")

# Ensure download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

class QdrantBackupDownloader:
    def __init__(self, mongodb_host: str, mongodb_port: int, db_name: str = "qdrant_backups"):
        """Initialize downloader with MongoDB connection."""
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
            # Test connection
            self.mongodb_client.admin.command('ismaster')
            self.db = self.mongodb_client[self.db_name]
            print("✅ Connected to MongoDB successfully!")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            sys.exit(1)
    
    def format_bytes(self, bytes_size: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} TB"
    
    def list_available_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with details."""
        print("\n📋 Fetching available backups...")
        
        # Get backup summaries
        summaries = list(self.db["backup_summaries"].find().sort("timestamp", -1))
        
        if not summaries:
            print("No backups found in MongoDB!")
            return []
        
        print(f"\nFound {len(summaries)} backup(s):\n")
        
        backups = []
        for i, summary in enumerate(summaries):
            backup_info = {
                "index": i + 1,
                "backup_id": summary["backup_id"],
                "timestamp": summary["timestamp"],
                "total_collections": summary["total_collections"],
                "successful_backups": summary["successful_backups"],
                "failed_backups": summary.get("failed_backups", 0),
                "collections": summary.get("collections", [])
            }
            
            # Calculate approximate size
            total_points = sum(col.get("points_backed_up", 0) for col in backup_info["collections"] if "error" not in col)
            
            print(f"{backup_info['index']}. Backup ID: {backup_info['backup_id']}")
            print(f"   📅 Date: {backup_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   📊 Collections: {backup_info['successful_backups']} successful, {backup_info['failed_backups']} failed")
            print(f"   📈 Total Points: {total_points:,}")
            print()
            
            backups.append(backup_info)
        
        return backups
    
    def get_collection_size(self, collection_name: str, timestamp: datetime) -> int:
        """Get the number of documents for a collection at specific backup time."""
        mongo_collection = self.db[f"backup_{collection_name}"]
        return mongo_collection.count_documents({"backup_timestamp": timestamp})
    
    def export_backup_to_zip(self, backup_info: Dict[str, Any], output_path: str):
        """Export a backup to a zip file with progress indication."""
        print(f"\n📦 Preparing to export backup: {backup_info['backup_id']}")
        
        # Get metadata for all collections in this backup
        # Use date range query to handle timestamp precision issues
        timestamp = backup_info["timestamp"]
        timestamp_start = timestamp.replace(microsecond=0)
        timestamp_end = timestamp.replace(microsecond=999999)
        
        metadata_docs = list(self.db["backup_metadata"].find({
            "timestamp": {
                "$gte": timestamp_start,
                "$lte": timestamp_end
            }
        }))
        
        # Debug: Check what we found
        print(f"Found {len(metadata_docs)} metadata documents for this backup")
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add backup summary
            summary_data = {
                "backup_id": backup_info["backup_id"],
                "timestamp": backup_info["timestamp"].isoformat(),
                "total_collections": backup_info["total_collections"],
                "successful_backups": backup_info["successful_backups"],
                "export_date": datetime.now().isoformat()
            }
            zipf.writestr("backup_summary.json", json.dumps(summary_data, indent=2))
            
            # Process each collection
            for col_info in backup_info["collections"]:
                if "error" in col_info:
                    continue
                
                collection_name = col_info["collection_name"]
                points_count = col_info.get("points_backed_up", 0)
                
                print(f"\n📂 Processing collection: {collection_name} (expected {points_count} points)")
                
                # Find metadata for this collection
                metadata = next((m for m in metadata_docs if m["collection_name"] == collection_name), None)
                if metadata:
                    # Save metadata
                    metadata_clean = {k: v for k, v in metadata.items() if k != "_id"}
                    metadata_clean["timestamp"] = metadata_clean["timestamp"].isoformat()
                    zipf.writestr(
                        f"{collection_name}/metadata.json",
                        json.dumps(metadata_clean, indent=2)
                    )
                
                # Export points
                mongo_collection = self.db[f"backup_{collection_name}"]
                
                # Use timestamp from collection info if available, otherwise use backup timestamp
                if "timestamp" in col_info:
                    query_timestamp = col_info["timestamp"]
                else:
                    query_timestamp = backup_info["timestamp"]
                
                # Try exact match first
                query = {"backup_timestamp": query_timestamp}
                total_docs = mongo_collection.count_documents(query)
                
                # If no documents found, try range query
                if total_docs == 0:
                    print(f"  ⚠️  No exact timestamp match, trying range query...")
                    query = {
                        "backup_timestamp": {
                            "$gte": timestamp_start,
                            "$lte": timestamp_end
                        }
                    }
                    total_docs = mongo_collection.count_documents(query)
                
                # If still no documents, check if collection has any documents
                if total_docs == 0:
                    total_in_collection = mongo_collection.count_documents({})
                    print(f"  ⚠️  No documents found for this backup timestamp")
                    print(f"  📊 Total documents in collection: {total_in_collection}")
                    
                    # If collection has documents, show sample timestamps
                    if total_in_collection > 0:
                        sample = list(mongo_collection.find().limit(1))
                        if sample and "backup_timestamp" in sample[0]:
                            print(f"  🕐 Sample timestamp from collection: {sample[0]['backup_timestamp']}")
                
                # Create progress bar
                with tqdm(total=total_docs, desc=f"Exporting {collection_name}", unit="points") as pbar:
                    # Process in batches
                    batch_size = 1000
                    batch_num = 0
                    
                    cursor = mongo_collection.find(query).batch_size(batch_size)
                    current_batch = []
                    
                    for doc in cursor:
                        # Remove MongoDB-specific fields
                        doc.pop("_id", None)
                        if isinstance(doc.get("backup_timestamp"), datetime):
                            doc["backup_timestamp"] = doc["backup_timestamp"].isoformat()
                        current_batch.append(doc)
                        
                        if len(current_batch) >= batch_size:
                            # Write batch to zip
                            batch_data = json.dumps(current_batch, indent=2)
                            zipf.writestr(
                                f"{collection_name}/batch_{batch_num:04d}.json",
                                batch_data
                            )
                            
                            pbar.update(len(current_batch))
                            batch_num += 1
                            current_batch = []
                    
                    # Write remaining documents
                    if current_batch:
                        batch_data = json.dumps(current_batch, indent=2)
                        zipf.writestr(
                            f"{collection_name}/batch_{batch_num:04d}.json",
                            batch_data
                        )
                        pbar.update(len(current_batch))
                
                print(f"✅ Exported {total_docs} points from {collection_name}")
    
    def download_backup(self, backup_index: int, backups: List[Dict[str, Any]]) -> Optional[str]:
        """Download a specific backup."""
        if backup_index < 1 or backup_index > len(backups):
            print("❌ Invalid backup number!")
            return None
        
        backup_info = backups[backup_index - 1]
        
        # Generate output filename
        timestamp_str = backup_info["timestamp"].strftime("%Y%m%d_%H%M%S")
        output_filename = f"qdrant_backup_{backup_info['backup_id']}_{timestamp_str}.zip"
        output_path = os.path.join(DOWNLOAD_DIR, output_filename)
        
        print(f"\n💾 Downloading backup to: {output_path}")
        
        try:
            # Export backup to zip
            self.export_backup_to_zip(backup_info, output_path)
            
            # Get file size
            file_size = os.path.getsize(output_path)
            print(f"\n✅ Backup downloaded successfully!")
            print(f"📁 File: {output_path}")
            print(f"📊 Size: {self.format_bytes(file_size)}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error downloading backup: {e}")
            # Clean up partial file
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
    
    def interactive_download(self):
        """Interactive interface for browsing and downloading backups."""
        print("="*60)
        print("🗄️  Qdrant Backup Downloader")
        print("="*60)
        
        while True:
            # List available backups
            backups = self.list_available_backups()
            
            if not backups:
                print("No backups available to download.")
                return
            
            print("\n" + "-"*60)
            print("Options:")
            print("  - Enter backup number (1-{}) to download".format(len(backups)))
            print("  - Enter 'q' to quit")
            print("  - Enter 'r' to refresh list")
            print("-"*60)
            
            choice = input("\n🔸 Your choice: ").strip().lower()
            
            if choice == 'q':
                print("\n👋 Goodbye!")
                break
            elif choice == 'r':
                print("\n🔄 Refreshing backup list...")
                continue
            else:
                try:
                    backup_num = int(choice)
                    
                    # Confirm download
                    if 1 <= backup_num <= len(backups):
                        backup = backups[backup_num - 1]
                        total_points = sum(col.get("points_backed_up", 0) 
                                         for col in backup["collections"] 
                                         if "error" not in col)
                        
                        print(f"\n📋 Selected: Backup {backup['backup_id']}")
                        print(f"   Date: {backup['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Collections: {backup['successful_backups']}")
                        print(f"   Total Points: {total_points:,}")
                        
                        confirm = input("\n⚠️  Download this backup? (yes/no): ").strip().lower()
                        
                        if confirm == 'yes':
                            downloaded_path = self.download_backup(backup_num, backups)
                            
                            if downloaded_path:
                                open_folder = input("\n📂 Open download folder? (yes/no): ").strip().lower()
                                if open_folder == 'yes':
                                    # Open folder in file explorer (cross-platform)
                                    import subprocess
                                    import platform
                                    
                                    system = platform.system()
                                    if system == 'Darwin':  # macOS
                                        subprocess.run(['open', DOWNLOAD_DIR])
                                    elif system == 'Windows':
                                        subprocess.run(['explorer', DOWNLOAD_DIR])
                                    else:  # Linux and others
                                        subprocess.run(['xdg-open', DOWNLOAD_DIR])
                        else:
                            print("❌ Download cancelled.")
                    else:
                        print("❌ Invalid backup number!")
                        
                except ValueError:
                    print("❌ Please enter a valid number or command!")
                except KeyboardInterrupt:
                    print("\n\n❌ Download cancelled by user.")
                    break
            
            input("\n🔸 Press Enter to continue...")

def main():
    """Main function."""
    try:
        # Check if tqdm is installed
        import tqdm
    except ImportError:
        print("❌ Please install tqdm for progress bars: pip install tqdm")
        sys.exit(1)
    
    downloader = QdrantBackupDownloader(
        mongodb_host=MONGODB_HOST,
        mongodb_port=MONGODB_PORT
    )
    
    downloader.interactive_download()

if __name__ == "__main__":
    main()