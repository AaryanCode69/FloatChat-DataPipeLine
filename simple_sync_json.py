"""
Simple Supabase to ChromaDB Sync Script
=======================================

This script connects to Supabase, fetches JSON from the "properties" column 
of the "floats" table, and inserts each JSON object exactly as-is into 
ChromaDB collection "float_embeddings".

Example JSON format:
{
  "date_range": {...},
  "measurements": {...},
  "location_range": {...},
  "total_profiles": 1
}
"""

import os
import json
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from ingest.db_handler import SupabaseHandler, ChromaDBHandler

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def connect_to_databases():
    """Initialize connections to Supabase and ChromaDB."""
    try:
        # Connect to Supabase
        logger.info("Connecting to Supabase...")
        supabase_handler = SupabaseHandler()
        logger.info("✓ Supabase connected")
        
        # Connect to ChromaDB
        logger.info("Connecting to ChromaDB...")
        chromadb_handler = ChromaDBHandler()
        logger.info("✓ ChromaDB connected")
        
        return supabase_handler, chromadb_handler
        
    except Exception as e:
        logger.error(f"Failed to connect to databases: {e}")
        raise


def fetch_json_from_supabase(supabase_handler):
    """Fetch JSON data from Supabase floats table."""
    try:
        logger.info("Fetching JSON data from Supabase floats table...")
        
        with supabase_handler.engine.connect() as conn:
            from sqlalchemy import text
            
            # Query to get properties JSON
            query = text("""
                SELECT float_id, properties 
                FROM floats 
                WHERE properties IS NOT NULL
            """)
            
            result = conn.execute(query)
            records = result.fetchall()
            
            # Parse JSON properties
            json_data = []
            for record in records:
                try:
                    if isinstance(record.properties, str):
                        properties_json = json.loads(record.properties)
                    else:
                        properties_json = record.properties
                    
                    json_data.append({
                        "float_id": record.float_id,
                        "json_data": properties_json
                    })
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON for {record.float_id}: {e}")
                    continue
            
            logger.info(f"✓ Fetched {len(json_data)} JSON records")
            return json_data
            
    except Exception as e:
        logger.error(f"Failed to fetch data from Supabase: {e}")
        raise


def insert_json_to_chromadb(chromadb_handler, json_data):
    """Insert JSON objects into ChromaDB collection."""
    try:
        logger.info("Inserting JSON data into ChromaDB...")
        
        # Get or create the collection
        collection = chromadb_handler.client.get_or_create_collection(
            name="float_embeddings",
            metadata={"description": "Float profile JSON embeddings"}
        )
        
        # Prepare data for insertion
        ids = []
        documents = []
        metadatas = []
        
        for i, record in enumerate(json_data):
            # Create unique ID as requested
            doc_id = f"profile_{i+1}"
            
            # Use the JSON as the document (convert to string for ChromaDB)
            json_document = json.dumps(record["json_data"], ensure_ascii=False)
            
            # Store only simple metadata (ChromaDB doesn't support nested objects in metadata)
            metadata = {
                "float_id": record["float_id"],
                "source": "supabase_sync",
                "data_type": "json_properties",
                "total_profiles": record["json_data"].get("total_profiles", 0) if isinstance(record["json_data"], dict) else 0
            }
            
            # Add location info if available (as simple values)
            if isinstance(record["json_data"], dict):
                location_range = record["json_data"].get("location_range", {})
                if location_range:
                    metadata.update({
                        "lat_min": location_range.get("lat_min", 0.0),
                        "lat_max": location_range.get("lat_max", 0.0),
                        "lon_min": location_range.get("lon_min", 0.0),
                        "lon_max": location_range.get("lon_max", 0.0)
                    })
                
                # Add measurement counts if available
                measurements = record["json_data"].get("measurements", {})
                if measurements:
                    metadata.update({
                        "has_temperature": "temperature" in measurements,
                        "has_salinity": "salinity" in measurements,
                        "has_pressure": "pressure" in measurements
                    })
            
            ids.append(doc_id)
            documents.append(json_document)
            metadatas.append(metadata)
        
        # Insert into ChromaDB
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"✓ Successfully inserted {len(ids)} JSON documents into ChromaDB")
        return len(ids)
        
    except Exception as e:
        logger.error(f"Failed to insert into ChromaDB: {e}")
        raise


def main():
    """Main execution function."""
    try:
        print("🚀 Starting Supabase to ChromaDB JSON sync...")
        
        # Connect to databases
        supabase_handler, chromadb_handler = connect_to_databases()
        
        # Fetch JSON data from Supabase
        json_data = fetch_json_from_supabase(supabase_handler)
        
        if not json_data:
            print("⚠️  No JSON data found in Supabase")
            return
        
        # Insert JSON into ChromaDB
        inserted_count = insert_json_to_chromadb(chromadb_handler, json_data)
        
        print(f"\n✅ SUCCESS!")
        print(f"📊 Total JSON records processed: {len(json_data)}")
        print(f"📥 Records inserted into ChromaDB: {inserted_count}")
        print(f"🎯 Collection: float_embeddings")
        print(f"🔑 Document IDs: profile_1, profile_2, ..., profile_{inserted_count}")
        
        # Show sample of what was inserted
        if json_data:
            print(f"\n📋 Sample JSON inserted:")
            sample_json = json_data[0]["json_data"]
            print(json.dumps(sample_json, indent=2))
        
    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        logger.error(f"Script execution failed: {e}")


if __name__ == "__main__":
    main()