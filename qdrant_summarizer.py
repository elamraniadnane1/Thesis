#!/usr/bin/env python3
"""
Qdrant Collection Properties Extractor (Perfected Version)

This script connects to a Qdrant database, extracts all collections and their properties,
and saves the information to a text file that can be used in future prompts.
"""

import json
import argparse
import sys
import logging
from datetime import datetime
import traceback
import os
from typing import Dict, Any, Optional, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qdrant_extractor")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    import qdrant_client
    QDRANT_AVAILABLE = True
    QDRANT_VERSION = getattr(qdrant_client, "__version__", "unknown")
    logger.info(f"Using qdrant-client version: {QDRANT_VERSION}")
except ImportError:
    logger.warning("Qdrant client not installed. Run 'pip install qdrant-client' to install.")
    QDRANT_AVAILABLE = False
    QDRANT_VERSION = None

# Define default Qdrant values
QDRANT_DEFAULTS = {
    "distance": "Cosine",  # Default distance metric in Qdrant
    "on_disk": False,      # Default storage location
    "hnsw_config": {       # Default HNSW parameters
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000
    }
}

# Custom JSON encoder to handle Qdrant objects
class QdrantEncoder(json.JSONEncoder):
    def default(self, obj):
        # Try to convert Pydantic models to dict
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        # For any other objects, use their __dict__ if available
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        # Otherwise use the default behavior
        return super().default(obj)

def connect_to_qdrant(host="localhost", port=6333, api_key=None, url=None, timeout=10.0, verify=True):
    """Connect to Qdrant instance using either local or remote configuration."""
    if not QDRANT_AVAILABLE:
        logger.error("Cannot connect: qdrant-client package is not installed")
        return None
    
    try:
        if url:
            # Connect to remote Qdrant instance
            logger.info(f"Connecting to remote Qdrant at {url}")
            client = QdrantClient(url=url, api_key=api_key, timeout=timeout, verify=verify)
        else:
            # Connect to local Qdrant instance
            logger.info(f"Connecting to local Qdrant at {host}:{port}")
            client = QdrantClient(host=host, port=port, timeout=timeout)
        
        # Test connection by getting collections (this is more reliable than get_cluster)
        test_collections = client.get_collections()
        logger.info(f"Successfully connected to Qdrant, found {len(test_collections.collections)} collections")
        return client
    except UnexpectedResponse as e:
        if hasattr(e, 'status_code') and e.status_code == 401:
            logger.error("Authentication failed. Check your API key.")
        else:
            logger.error(f"Unexpected response from Qdrant: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def to_dict(obj):
    """Safe converter from Pydantic model to dict, with proper deprecation handling."""
    if obj is None:
        return {}
    
    # Try model_dump first (Pydantic v2)
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return obj.model_dump()
        except Exception:
            pass
    
    # Fall back to dict (Pydantic v1)
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    
    # If it's already a dict, return it
    if isinstance(obj, dict):
        return obj
    
    # Last resort: try to get __dict__
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    
    # If all else fails, return empty dict
    return {}

def get_qdrant_version_info(client):
    """Get Qdrant server version information."""
    try:
        telemetry = client._client.telemetry()
        if hasattr(telemetry, 'status') and telemetry.status == "ok":
            if hasattr(telemetry, 'version'):
                return telemetry.version
            elif hasattr(telemetry, 'server') and hasattr(telemetry.server, 'version'):
                return telemetry.server.version
    except Exception:
        pass
    
    # Fallback method
    try:
        info = client._client.root()
        if hasattr(info, 'version'):
            return info.version
    except Exception:
        pass
    
    return "Unknown"

def get_vector_size_from_points(client, collection_name, vector_name):
    """Try to determine vector size by fetching a sample point."""
    try:
        # Scroll to get one point
        points = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_vectors=True
        )
        
        if points and points[0] and len(points[0]) > 0:
            points_data = points[0]
            if len(points_data) > 0:
                point = points_data[0]
                if hasattr(point, "vector") and point.vector:
                    vectors = to_dict(point.vector)
                    if vector_name in vectors:
                        vector_data = vectors[vector_name]
                        if isinstance(vector_data, list):
                            return len(vector_data)
    except Exception:
        pass
    
    return None

def extract_vector_configs(client, collection_name, vectors_param):
    """Safely extract vector configurations from collection info."""
    vector_configs = {}
    
    try:
        # Get Qdrant server version for context
        server_version = get_qdrant_version_info(client)
        
        # First convert to dictionary using the safe converter
        vectors_dict = to_dict(vectors_param)
        
        # Process each vector configuration
        for vector_name, vector_config in vectors_dict.items():
            if isinstance(vector_config, dict):
                # Standard dictionary processing
                size = vector_config.get("size")
                distance = vector_config.get("distance", QDRANT_DEFAULTS["distance"])
                on_disk = vector_config.get("on_disk", QDRANT_DEFAULTS["on_disk"])
                hnsw_config = vector_config.get("hnsw_config", QDRANT_DEFAULTS["hnsw_config"])
                quantization_config = vector_config.get("quantization_config")
                
                vector_configs[vector_name] = {
                    "size": size,
                    "distance": distance,
                    "on_disk": on_disk,
                    "hnsw_config": hnsw_config,
                    "quantization_config": quantization_config,
                }
            elif hasattr(vector_config, "size") and hasattr(vector_config, "distance"):
                # Object with attributes
                vector_configs[vector_name] = {
                    "size": getattr(vector_config, "size"),
                    "distance": getattr(vector_config, "distance", QDRANT_DEFAULTS["distance"]),
                    "on_disk": getattr(vector_config, "on_disk", QDRANT_DEFAULTS["on_disk"]),
                    "hnsw_config": getattr(vector_config, "hnsw_config", QDRANT_DEFAULTS["hnsw_config"]),
                    "quantization_config": getattr(vector_config, "quantization_config"),
                }
            elif isinstance(vector_config, int):
                # Just a dimension size (simplified config)
                # For this case, we'll get the distance metric from the server version
                if server_version and "2." in str(server_version):
                    # Qdrant 2.x uses Cosine by default
                    default_distance = "Cosine"
                else:
                    default_distance = QDRANT_DEFAULTS["distance"]
                
                vector_configs[vector_name] = {
                    "size": vector_config,
                    "distance": default_distance,
                    "on_disk": QDRANT_DEFAULTS["on_disk"],
                    "hnsw_config": QDRANT_DEFAULTS["hnsw_config"],
                    "quantization_config": None,
                }
            else:
                # Try to infer size by reading a point
                size = get_vector_size_from_points(client, collection_name, vector_name)
                
                vector_configs[vector_name] = {
                    "size": size if size else "Inferred from data",
                    "distance": QDRANT_DEFAULTS["distance"],
                    "on_disk": QDRANT_DEFAULTS["on_disk"],
                    "hnsw_config": QDRANT_DEFAULTS["hnsw_config"],
                    "quantization_config": None,
                    "note": f"Configuration inferred, type: {type(vector_config).__name__}"
                }
    except Exception as e:
        logger.warning(f"Error in extract_vector_configs: {e}")
        vector_configs = {"error": f"Could not extract vector configurations: {str(e)}"}
    
    return vector_configs

def get_collection_direct_info(client, name):
    """Get collection info using direct REST API calls if possible."""
    try:
        # Try to get collection info via REST API directly
        collection_info = client.get_collection(collection_name=name)
        
        # Extract basic configuration
        config = {
            "name": name,
        }
        
        # Try to extract other configuration parameters safely
        try:
            # Convert to dict first
            params = to_dict(collection_info.config.params)
            
            # Add all parameters to config
            for key, value in params.items():
                if key != "vectors" and key != "schema":  # These are handled separately
                    config[key] = value
            
            # Ensure we have the critical parameters
            if "shard_number" not in config:
                config["shard_number"] = 1  # Default in Qdrant
            
            if "replication_factor" not in config:
                config["replication_factor"] = 1  # Default in Qdrant
            
            if "write_consistency_factor" not in config:
                config["write_consistency_factor"] = 1  # Default in Qdrant
            
            if "on_disk_payload" not in config:
                config["on_disk_payload"] = False  # Default in Qdrant
        except AttributeError:
            logger.warning(f"Could not access all config parameters for {name}")
        
        return collection_info, config
    except Exception as e:
        logger.error(f"Error getting collection info for {name}: {e}")
        return None, {"name": name, "error": str(e)}

def extract_schema_info(schema_param):
    """Safely extract schema information."""
    try:
        if schema_param is None:
            return "No schema defined"
        
        # Use the safe converter
        schema_dict = to_dict(schema_param)
        
        if schema_dict:
            return schema_dict
        
        # If we got an empty dict but have a schema_param object
        if hasattr(schema_param, "__dict__"):
            return schema_param.__dict__
        
        return "No schema defined (empty schema)"
    except Exception as e:
        logger.warning(f"Error extracting schema: {e}")
        return f"Error extracting schema: {str(e)}"

def get_all_collections_info(client, verbose=False):
    """Get a list of all collections and their detailed properties."""
    if client is None:
        logger.error("Cannot get collections: No valid client connection")
        return {}
    
    collections_info = {}
    
    try:
        # Get list of all collections
        collections_list = client.get_collections().collections
        collection_names = [collection.name for collection in collections_list]
        logger.info(f"Found {len(collection_names)} collections: {', '.join(collection_names)}")
        
        # For each collection, get its detailed information
        for name in collection_names:
            logger.info(f"Processing collection: {name}")
            try:
                # Get collection info
                collection_info, config = get_collection_direct_info(client, name)
                
                if collection_info is None:
                    collections_info[name] = {"error": "Failed to retrieve collection information"}
                    continue
                
                if verbose:
                    logger.info(f"Raw collection info type for {name}: {type(collection_info)}")
                
                # Get vector dimensions and distance metric
                vector_configs = {}
                try:
                    if hasattr(collection_info.config.params, "vectors"):
                        vectors_param = collection_info.config.params.vectors
                        vector_configs = extract_vector_configs(client, name, vectors_param)
                    else:
                        logger.warning(f"Collection {name} does not have 'vectors' attribute")
                        vector_configs = {"error": "Collection does not have vectors attribute"}
                except Exception as e:
                    logger.warning(f"Error extracting vector configs for {name}: {e}")
                    vector_configs = {"error": f"Could not extract vector configurations: {str(e)}"}
                
                # Get schema information (payload fields)
                try:
                    if hasattr(collection_info.config.params, "schema"):
                        collection_schema = collection_info.config.params.schema
                        schema_info = extract_schema_info(collection_schema)
                    else:
                        schema_info = "No schema defined"
                except Exception as e:
                    logger.warning(f"Error extracting schema for {name}: {e}")
                    schema_info = f"Schema information not available: {str(e)}"
                
                # Count points in the collection
                try:
                    count_result = client.count(collection_name=name, exact=True)
                    point_count = count_result.count
                except Exception as e:
                    logger.warning(f"Error counting points for {name}: {e}")
                    
                    # Try alternative count method if exact count fails
                    try:
                        count_result = client.count(collection_name=name, exact=False)
                        point_count = f"{count_result.count} (approximate)"
                    except Exception as e2:
                        logger.warning(f"Alternative count also failed for {name}: {e2}")
                        point_count = "Unable to retrieve count"
                
                # Compile all information
                collections_info[name] = {
                    "config": config,
                    "vector_configs": vector_configs,
                    "schema": schema_info,
                    "point_count": point_count,
                }
                
            except Exception as e:
                logger.error(f"Error processing collection {name}: {e}")
                collections_info[name] = {
                    "error": f"Failed to extract information: {str(e)}"
                }
        
    except Exception as e:
        logger.error(f"Failed to get collections list: {e}")
        logger.debug(traceback.format_exc())
    
    return collections_info

def safe_serialize(obj):
    """Convert complex objects to serializable dictionaries."""
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return obj.model_dump()
        except Exception:
            pass
    
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_serialize(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return {k: safe_serialize(v) for k, v in obj.__dict__.items() 
                if not k.startswith('_')}
    else:
        # For primitive types
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def format_for_prompt(collections_info):
    """Format the collections information in a way that's useful for prompts."""
    formatted_text = "# Qdrant Collections Information\n\n"
    formatted_text += f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if QDRANT_VERSION:
        formatted_text += f"Qdrant Client Version: {QDRANT_VERSION}\n\n"
    else:
        formatted_text += "\n"
    
    if not collections_info:
        formatted_text += "**No collections found or unable to connect to Qdrant server.**\n\n"
        return formatted_text
    
    for collection_name, info in collections_info.items():
        formatted_text += f"## Collection: {collection_name}\n\n"
        
        # Check if there was an error processing this collection
        if "error" in info:
            formatted_text += f"**Error**: {info['error']}\n\n"
            formatted_text += "-" * 80 + "\n\n"
            continue
        
        # Add point count
        formatted_text += f"**Points Count**: {info['point_count']}\n\n"
        
        # Add vector configurations
        formatted_text += "### Vector Configurations\n\n"
        if isinstance(info['vector_configs'], dict) and "error" in info['vector_configs']:
            formatted_text += f"**Error**: {info['vector_configs']['error']}\n\n"
        elif isinstance(info['vector_configs'], dict):
            for vector_name, vector_config in info['vector_configs'].items():
                formatted_text += f"**Vector Name**: {vector_name}\n"
                
                if isinstance(vector_config, dict):
                    # Standard vector config display
                    if 'size' in vector_config and vector_config['size'] is not None:
                        formatted_text += f"- Dimensions: {vector_config['size']}\n"
                    else:
                        formatted_text += f"- Dimensions: Not specified (using default)\n"
                        
                    if 'distance' in vector_config and vector_config['distance'] is not None:
                        formatted_text += f"- Distance Metric: {vector_config['distance']}\n"
                    else:
                        formatted_text += f"- Distance Metric: Cosine (default)\n"
                    
                    if 'on_disk' in vector_config:
                        formatted_text += f"- Stored On Disk: {vector_config['on_disk']}\n"
                    else:
                        formatted_text += f"- Stored On Disk: False (default)\n"
                    
                    # Add HNSW configuration if available
                    if vector_config.get('hnsw_config'):
                        formatted_text += "- HNSW Configuration:\n"
                        hnsw_config = vector_config['hnsw_config']
                        if isinstance(hnsw_config, dict):
                            for k, v in hnsw_config.items():
                                formatted_text += f"  - {k}: {v}\n"
                        else:
                            formatted_text += f"  - Raw config: {hnsw_config}\n"
                    else:
                        formatted_text += "- HNSW Configuration: Using defaults\n"
                    
                    # Add quantization configuration if available
                    if vector_config.get('quantization_config'):
                        formatted_text += "- Quantization Configuration:\n"
                        quant_config = vector_config['quantization_config']
                        if isinstance(quant_config, dict):
                            for k, v in quant_config.items():
                                formatted_text += f"  - {k}: {v}\n"
                        else:
                            formatted_text += f"  - Raw config: {quant_config}\n"
                    else:
                        formatted_text += "- Quantization: None (not using quantization)\n"
                    
                    # Add any notes
                    if 'note' in vector_config:
                        formatted_text += f"- Note: {vector_config['note']}\n"
                else:
                    # Unknown format
                    formatted_text += f"- Raw configuration: {vector_config}\n"
                
                formatted_text += "\n"
        else:
            formatted_text += f"Vector configurations not available in expected format.\n\n"
        
        # Add schema information if available
        formatted_text += "### Schema Information\n\n"
        if isinstance(info['schema'], dict):
            if not info['schema']:
                formatted_text += "No schema defined (empty schema)\n\n"
            else:
                for field_name, field_schema in info['schema'].items():
                    formatted_text += f"**Field**: {field_name}\n"
                    if isinstance(field_schema, dict):
                        formatted_text += f"- Type: {field_schema.get('type', 'Not specified')}\n"
                        if 'properties' in field_schema and isinstance(field_schema['properties'], dict):
                            formatted_text += "- Properties:\n"
                            for prop_name, prop_details in field_schema['properties'].items():
                                formatted_text += f"  - {prop_name}: {prop_details}\n"
                    else:
                        formatted_text += f"- Schema: {field_schema}\n"
                    formatted_text += "\n"
        else:
            formatted_text += f"{info['schema']}\n\n"
        
        # Add collection configuration
        formatted_text += "### Collection Configuration\n\n"
        if "error" in info['config']:
            formatted_text += f"**Error**: {info['config']['error']}\n\n"
        else:
            for key, value in info['config'].items():
                if key != "name" and key != "vectors_config" and key != "error":
                    formatted_text += f"- {key}: {value}\n"
            formatted_text += "\n"
        
        formatted_text += "-" * 80 + "\n\n"
    
    return formatted_text

def save_to_file(text, filename="qdrant_collections_info.txt"):
    """Save the formatted text to a file."""
    try:
        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Successfully saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save to {filename}: {e}")
        logger.debug(traceback.format_exc())
        return None

def generate_sample_data():
    """Generate sample data if connection fails, for testing purposes."""
    logger.info("Generating sample data for demonstration")
    return {
        "sample_collection_1": {
            "config": {
                "name": "sample_collection_1",
                "shard_number": 1,
                "replication_factor": 1,
                "write_consistency_factor": 1,
                "on_disk_payload": True,
            },
            "vector_configs": {
                "default": {
                    "size": 1536,
                    "distance": "Cosine",
                    "on_disk": True,
                    "hnsw_config": {
                        "m": 16,
                        "ef_construct": 100
                    },
                    "quantization_config": None
                }
            },
            "schema": {
                "text": {
                    "type": "text",
                    "properties": {}
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "keyword"},
                        "date": {"type": "keyword"}
                    }
                }
            },
            "point_count": 5000,
        },
        "sample_collection_2": {
            "config": {
                "name": "sample_collection_2",
                "shard_number": 2,
                "replication_factor": 1,
                "write_consistency_factor": 1,
                "on_disk_payload": False,
            },
            "vector_configs": {
                "image": {
                    "size": 512,
                    "distance": "Dot",
                    "on_disk": False,
                    "hnsw_config": None,
                    "quantization_config": {
                        "scalar": {
                            "type": "int8",
                            "quantile": 0.99,
                            "always_ram": True
                        }
                    }
                }
            },
            "schema": "No schema defined",
            "point_count": 1200,
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Extract Qdrant collections information')
    parser.add_argument('--host', default='localhost', help='Qdrant server host (default: localhost)')
    parser.add_argument('--port', default=6333, type=int, help='Qdrant server port (default: 6333)')
    parser.add_argument('--url', help='Qdrant server URL for cloud deployments')
    parser.add_argument('--api-key', help='API key for cloud deployments')
    parser.add_argument('--output', default='qdrant_collections_info.txt', help='Output filename')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--timeout', type=float, default=10.0, help='Connection timeout in seconds')
    parser.add_argument('--no-verify', action='store_true', help='Disable SSL verification')
    parser.add_argument('--sample', action='store_true', help='Generate sample data instead of connecting')
    parser.add_argument('--no-json', action='store_true', help='Skip generating JSON output')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    try:
        collections_info = {}
        
        if args.sample:
            # Generate sample data for testing
            collections_info = generate_sample_data()
        else:
            # Connect to Qdrant
            logger.info(f"Connecting to Qdrant {'at URL ' + args.url if args.url else 'at ' + args.host + ':' + str(args.port)}...")
            client = connect_to_qdrant(
                host=args.host, 
                port=args.port, 
                api_key=args.api_key, 
                url=args.url,
                timeout=args.timeout,
                verify=not args.no_verify
            )
            
            if client is None:
                logger.error("Failed to establish connection to Qdrant. Exiting.")
                logger.info("You can run with --sample to generate sample data for testing.")
                sys.exit(1)
            
            # Get collections information
            logger.info("Getting collections information...")
            collections_info = get_all_collections_info(client, verbose=args.verbose)
        
        if not collections_info:
            logger.warning("No collections found in the Qdrant database.")
        else:
            logger.info(f"Found {len(collections_info)} collections.")
        
        # Format the information
        formatted_text = format_for_prompt(collections_info)
        
        # Save to text file
        filename = save_to_file(formatted_text, args.output)
        if filename:
            logger.info(f"Collections information saved to {filename}")
        
        # Also save a JSON version for programmatic access
        if collections_info and not args.no_json:
            json_filename = args.output.replace('.txt', '.json')
            try:
                # Convert all objects to serializable form
                serializable_info = safe_serialize(collections_info)
                
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(serializable_info, f, indent=2)
                logger.info(f"JSON format saved to {json_filename}")
            except Exception as e:
                logger.error(f"Failed to save JSON: {e}")
                logger.debug(traceback.format_exc())
                logger.info("Run with --no-json to skip JSON output")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()