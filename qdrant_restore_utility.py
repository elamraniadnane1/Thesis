#!/usr/bin/env python3
"""
Qdrant Restore Utility
Interactive script to restore Qdrant collections from MongoDB backups
"""

import sys
from datetime import datetime
from qdrant_mongodb_backup import QdrantMongoBackup, QDRANT_URL, MONGODB_HOST, MONGODB_PORT

def print_menu():
    """Print the main menu."""
    print("\n" + "="*50)
    print("Qdrant Restore Utility")
    print("="*50)
    print("1. List all backups")
    print("2. Restore specific collection")
    print("3. Restore all collections")
    print("4. View backup statistics")
    print("5. Exit")
    print("="*50)

def list_backups(backup_system):
    """List all available backups."""
    backups = backup_system.list_backups()
    
    if not backups:
        print("No backups found!")
        return
    
    print("\nAvailable Backups:")
    print("-" * 80)
    
    for i, backup in enumerate(backups):
        print(f"\nBackup #{i+1}")
        print(f"  ID: {backup['backup_id']}")
        print(f"  Timestamp: {backup['timestamp']}")
        print(f"  Total Collections: {backup['total_collections']}")
        print(f"  Successful: {backup['successful_backups']}")
        print(f"  Failed: {backup['failed_backups']}")
        
        if backup.get('collections'):
            print("  Collections:")
            for col in backup['collections']:
                if 'error' not in col:
                    print(f"    - {col['collection_name']}: {col['points_backed_up']} points")

def restore_specific_collection(backup_system):
    """Restore a specific collection."""
    # Get collection name
    collection_name = input("\nEnter collection name to restore: ").strip()
    
    if not collection_name:
        print("Invalid collection name!")
        return
    
    # Check if backups exist for this collection
    metadata = backup_system.db["backup_metadata"].find(
        {"collection_name": collection_name}
    ).sort("timestamp", -1)
    
    backups = list(metadata)
    
    if not backups:
        print(f"No backups found for collection '{collection_name}'!")
        return
    
    print(f"\nFound {len(backups)} backup(s) for '{collection_name}':")
    for i, backup in enumerate(backups):
        print(f"{i+1}. Backup from {backup['timestamp']} - {backup['points_count']} points")
    
    # Select backup
    try:
        choice = int(input("\nSelect backup number (or 0 for latest): "))
        if choice == 0:
            selected_backup = backups[0]
        elif 1 <= choice <= len(backups):
            selected_backup = backups[choice - 1]
        else:
            print("Invalid choice!")
            return
    except ValueError:
        print("Invalid input!")
        return
    
    # Confirm restoration
    confirm = input(f"\nRestore '{collection_name}' from {selected_backup['timestamp']}? (yes/no): ")
    
    if confirm.lower() == 'yes':
        print(f"\nRestoring collection '{collection_name}'...")
        success = backup_system.restore_collection(
            collection_name,
            selected_backup['timestamp']
        )
        
        if success:
            print(f"✅ Successfully restored collection '{collection_name}'!")
        else:
            print(f"❌ Failed to restore collection '{collection_name}'!")

def restore_all_collections(backup_system):
    """Restore all collections from a specific backup."""
    backups = backup_system.list_backups()
    
    if not backups:
        print("No backups found!")
        return
    
    print("\nAvailable backup runs:")
    for i, backup in enumerate(backups[:10]):  # Show last 10
        print(f"{i+1}. {backup['backup_id']} - {backup['timestamp']} ({backup['successful_backups']} collections)")
    
    try:
        choice = int(input("\nSelect backup run number: "))
        if 1 <= choice <= len(backups):
            selected_backup = backups[choice - 1]
        else:
            print("Invalid choice!")
            return
    except ValueError:
        print("Invalid input!")
        return
    
    # Confirm restoration
    confirm = input(f"\nRestore ALL collections from backup {selected_backup['backup_id']}? (yes/no): ")
    
    if confirm.lower() == 'yes':
        print(f"\nRestoring all collections from backup {selected_backup['backup_id']}...")
        
        restored = 0
        failed = 0
        
        for col_info in selected_backup.get('collections', []):
            if 'error' not in col_info:
                print(f"\nRestoring {col_info['collection_name']}...")
                success = backup_system.restore_collection(
                    col_info['collection_name'],
                    selected_backup['timestamp']
                )
                
                if success:
                    restored += 1
                    print(f"✅ Restored {col_info['collection_name']}")
                else:
                    failed += 1
                    print(f"❌ Failed to restore {col_info['collection_name']}")
        
        print(f"\n=== Restoration Complete ===")
        print(f"Successfully restored: {restored} collections")
        print(f"Failed: {failed} collections")

def view_backup_statistics(backup_system):
    """View backup statistics."""
    stats = backup_system.get_backup_stats()
    
    print("\n=== Backup Statistics ===")
    print(f"Total backup runs: {stats['total_backups']}")
    print(f"Total backup size: {stats['total_size_mb']:.2f} MB")
    print(f"Collections backed up: {len(stats['collections'])}")
    
    if stats['collections']:
        print("\nBackup sizes by collection:")
        sorted_collections = sorted(
            stats['collections'].items(),
            key=lambda x: x[1]['size_mb'],
            reverse=True
        )
        
        for col_name, col_stats in sorted_collections[:10]:  # Top 10
            clean_name = col_name.replace('backup_', '')
            print(f"  {clean_name}: {col_stats['size_mb']:.2f} MB ({col_stats['count']} documents)")

def main():
    """Main function."""
    print("Connecting to Qdrant and MongoDB...")
    
    try:
        backup_system = QdrantMongoBackup(
            qdrant_url=QDRANT_URL,
            mongodb_host=MONGODB_HOST,
            mongodb_port=MONGODB_PORT
        )
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)
    
    while True:
        print_menu()
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                list_backups(backup_system)
            elif choice == '2':
                restore_specific_collection(backup_system)
            elif choice == '3':
                restore_all_collections(backup_system)
            elif choice == '4':
                view_backup_statistics(backup_system)
            elif choice == '5':
                print("\nExiting...")
                break
            else:
                print("Invalid choice! Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()