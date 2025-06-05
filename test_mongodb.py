from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import sys

def test_mongodb_connection(host, port, timeout=5000):
    """Test connection to MongoDB server"""
    connection_string = f"mongodb://{host}:{port}/"
    
    try:
        # Set a timeout to avoid hanging if server is unreachable
        client = MongoClient(connection_string, serverSelectionTimeoutMS=timeout)
        
        # Force a connection by requesting server info
        client.admin.command('ismaster')
        
        print(f"✅ Successfully connected to MongoDB at {host}:{port}")
        
        # Optional: List available databases
        print("\nAvailable databases:")
        for db_name in client.list_database_names():
            print(f"- {db_name}")
            
        return True
        
    except ConnectionFailure:
        print(f"❌ Failed to connect to MongoDB at {host}:{port}")
        return False
    except ServerSelectionTimeoutError:
        print(f"❌ Server selection timeout: MongoDB at {host}:{port} is unreachable")
        return False
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return False

if __name__ == "__main__":
    # You can replace these with your actual MongoDB server details
    mongodb_host = "154.44.186.241"
    mongodb_port = 27017
    
    test_mongodb_connection(mongodb_host, mongodb_port)