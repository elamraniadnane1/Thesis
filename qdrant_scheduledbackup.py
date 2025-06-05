#!/usr/bin/env python3
"""
Qdrant Scheduled Backup Script
Can be run via cron for automatic backups
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from qdrant_mongodb_backup import QdrantMongoBackup, QDRANT_URL, MONGODB_HOST, MONGODB_PORT

# Configure logging to file
LOG_DIR = os.path.expanduser("~/qdrant_backup_logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RETENTION_DAYS = 7  # Keep backups for 7 days
ALERT_EMAIL = None  # Set to your email for alerts (requires additional setup)

def cleanup_old_backups(backup_system, retention_days):
    """Remove backups older than retention_days."""
    logger.info(f"Cleaning up backups older than {retention_days} days...")
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    # Find old backup summaries
    old_summaries = list(backup_system.db["backup_summaries"].find({
        "timestamp": {"$lt": cutoff_date}
    }))
    
    if not old_summaries:
        logger.info("No old backups to clean up.")
        return
    
    logger.info(f"Found {len(old_summaries)} old backup(s) to remove.")
    
    for summary in old_summaries:
        # Remove backup data for each collection
        for col_info in summary.get('collections', []):
            if 'error' not in col_info:
                collection_name = f"backup_{col_info['collection_name']}"
                # Remove documents from this backup
                result = backup_system.db[collection_name].delete_many({
                    "backup_timestamp": summary['timestamp']
                })
                logger.info(f"Removed {result.deleted_count} documents from {collection_name}")
        
        # Remove metadata
        backup_system.db["backup_metadata"].delete_many({
            "timestamp": summary['timestamp']
        })
        
        # Remove summary
        backup_system.db["backup_summaries"].delete_one({
            "_id": summary['_id']
        })
        
        logger.info(f"Removed backup from {summary['timestamp']}")

def verify_backup(backup_system, results):
    """Verify backup integrity."""
    logger.info("Verifying backup integrity...")
    
    issues = []
    
    for result in results:
        if "error" in result:
            issues.append(f"Failed to backup {result['collection_name']}: {result['error']}")
        else:
            # Verify point count
            collection_name = result['collection_name']
            expected_count = result['points_backed_up']
            
            # Count documents in MongoDB
            mongo_collection = backup_system.db[f"backup_{collection_name}"]
            actual_count = mongo_collection.count_documents({
                "backup_timestamp": result['timestamp']
            })
            
            if actual_count != expected_count:
                issues.append(
                    f"Point count mismatch for {collection_name}: "
                    f"expected {expected_count}, found {actual_count}"
                )
            else:
                logger.info(f"✅ Verified {collection_name}: {actual_count} points")
    
    return issues

def generate_backup_report(backup_system, results, issues):
    """Generate a detailed backup report."""
    report = []
    report.append("="*60)
    report.append("QDRANT BACKUP REPORT")
    report.append("="*60)
    report.append(f"Backup Date: {datetime.now()}")
    report.append("")
    
    # Summary
    successful = sum(1 for r in results if "error" not in r)
    failed = sum(1 for r in results if "error" in r)
    total_points = sum(r.get('points_backed_up', 0) for r in results if "error" not in r)
    
    report.append("SUMMARY:")
    report.append(f"  Total Collections: {len(results)}")
    report.append(f"  Successful: {successful}")
    report.append(f"  Failed: {failed}")
    report.append(f"  Total Points Backed Up: {total_points}")
    report.append("")
    
    # Collection Details
    report.append("COLLECTION DETAILS:")
    for result in results:
        if "error" in result:
            report.append(f"  ❌ {result['collection_name']}: FAILED - {result['error']}")
        else:
            report.append(f"  ✅ {result['collection_name']}: {result['points_backed_up']} points")
    report.append("")
    
    # Storage Statistics
    stats = backup_system.get_backup_stats()
    report.append("STORAGE STATISTICS:")
    report.append(f"  Total Backup Size: {stats['total_size_mb']:.2f} MB")
    report.append(f"  Number of Backup Runs: {stats['total_backups']}")
    report.append("")
    
    # Issues
    if issues:
        report.append("ISSUES FOUND:")
        for issue in issues:
            report.append(f"  ⚠️  {issue}")
    else:
        report.append("✅ No issues found - backup verified successfully!")
    
    report.append("")
    report.append("="*60)
    
    return "\n".join(report)

def main():
    """Main backup function."""
    logger.info("="*60)
    logger.info("Starting Qdrant Scheduled Backup")
    logger.info("="*60)
    
    try:
        # Initialize backup system
        backup_system = QdrantMongoBackup(
            qdrant_url=QDRANT_URL,
            mongodb_host=MONGODB_HOST,
            mongodb_port=MONGODB_PORT
        )
        
        # Perform backup
        logger.info("Starting backup process...")
        start_time = datetime.now()
        results = backup_system.backup_all_collections()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Backup completed in {duration:.2f} seconds")
        
        # Verify backup
        issues = verify_backup(backup_system, results)
        
        # Generate report
        report = generate_backup_report(backup_system, results, issues)
        logger.info("\n" + report)
        
        # Save report to file
        report_filename = os.path.join(
            LOG_DIR, 
            f"backup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_filename, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_filename}")
        
        # Cleanup old backups
        if RETENTION_DAYS > 0:
            cleanup_old_backups(backup_system, RETENTION_DAYS)
        
        # Exit with appropriate code
        if issues or any("error" in r for r in results):
            logger.error("Backup completed with errors!")
            sys.exit(1)
        else:
            logger.info("Backup completed successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Critical error during backup: {e}", exc_info=True)
        sys.exit(2)

if __name__ == "__main__":
    main()

# CRON SETUP INSTRUCTIONS:
# To run this script daily at 2 AM, add this to your crontab (crontab -e):
# 0 2 * * * /usr/bin/python3 /path/to/qdrant_scheduled_backup.py >> /var/log/qdrant_backup.log 2>&1
#
# For hourly backups:
# 0 * * * * /usr/bin/python3 /path/to/qdrant_scheduled_backup.py >> /var/log/qdrant_backup.log 2>&1
#
# For weekly backups (Sunday at 3 AM):
# 0 3 * * 0 /usr/bin/python3 /path/to/qdrant_scheduled_backup.py >> /var/log/qdrant_backup.log 2>&1