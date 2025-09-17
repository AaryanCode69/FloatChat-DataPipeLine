"""
Sync Supabase floats data to ChromaDB
=====================================

This script connects to Supabase, fetches JSON stored in the "properties" column 
of the "floats" table, and inserts each JSON object as a document into the 
existing ChromaDB collection named "float_embeddings".

Author: FloatChat Data Pipeline
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from ingest.db_handler import SupabaseHandler, ChromaDBHandler
from sentence_transformers import SentenceTransformer

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SupabaseToChromaDBSync:
    """Synchronize data from Supabase to ChromaDB."""
    
    def __init__(self):
        """Initialize the sync service."""
        self.supabase_handler = None
        self.chromadb_handler = None
        self.embedding_model = None
        self.collection_name = "float_embeddings"
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections and embedding model."""
        try:
            # Initialize Supabase connection
            logger.info("Connecting to Supabase...")
            self.supabase_handler = SupabaseHandler()
            logger.info("✓ Supabase connection established")
            
            # Initialize ChromaDB connection
            logger.info("Connecting to ChromaDB...")
            self.chromadb_handler = ChromaDBHandler()
            logger.info("✓ ChromaDB connection established")
            
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Embedding model loaded")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    def fetch_supabase_data(self) -> List[Dict[str, Any]]:
        """
        Fetch all records from Supabase floats table.
        
        Returns:
            List of dictionaries containing float_id and properties
        """
        try:
            logger.info("Fetching data from Supabase floats table...")
            
            # Query to get all records from floats table
            with self.supabase_handler.engine.connect() as conn:
                from sqlalchemy import text
                
                query = text("""
                    SELECT float_id, properties, platform_number, deploy_date
                    FROM floats
                    WHERE properties IS NOT NULL
                    ORDER BY float_id
                """)
                
                result = conn.execute(query)
                records = result.fetchall()
                
                # Convert to list of dictionaries
                data = []
                for record in records:
                    try:
                        # Parse the JSON properties
                        properties_json = json.loads(record.properties) if isinstance(record.properties, str) else record.properties
                        
                        data.append({
                            "float_id": record.float_id,
                            "platform_number": record.platform_number,
                            "deploy_date": str(record.deploy_date) if record.deploy_date else None,
                            "properties": properties_json
                        })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON for float_id {record.float_id}: {e}")
                        continue
                
                logger.info(f"✓ Fetched {len(data)} records from Supabase")
                return data
                
        except Exception as e:
            logger.error(f"Failed to fetch data from Supabase: {e}")
            raise
    
    def create_document_text(self, properties: Dict[str, Any], float_id: str, platform_number: str = None) -> str:
        """
        Create a descriptive text document from the JSON properties.
        
        Args:
            properties: The JSON properties from Supabase
            float_id: The float identifier
            platform_number: Platform number if available
            
        Returns:
            Descriptive text for embedding
        """
        try:
            # Extract key information from the JSON
            total_profiles = properties.get("total_profiles", "unknown")
            
            # Date range
            date_range = properties.get("date_range", {})
            start_date = date_range.get("start", "unknown")
            end_date = date_range.get("end", "unknown")
            
            # Location range
            location_range = properties.get("location_range", {})
            lat_min = location_range.get("lat_min", "unknown")
            lat_max = location_range.get("lat_max", "unknown")
            lon_min = location_range.get("lon_min", "unknown")
            lon_max = location_range.get("lon_max", "unknown")
            
            # Measurements summary
            measurements = properties.get("measurements", {})
            measurement_summary = []
            
            for measure_type, data in measurements.items():
                if isinstance(data, dict):
                    min_val = data.get("min", "N/A")
                    max_val = data.get("max", "N/A")
                    mean_val = data.get("mean", "N/A")
                    count = data.get("count", "N/A")
                    measurement_summary.append(f"{measure_type}: min={min_val}, max={max_val}, mean={mean_val}, count={count}")
            
            # Create descriptive document
            document_text = f"""
Float Profile Data - ID: {float_id}
Platform Number: {platform_number or 'N/A'}
Total Profiles: {total_profiles}
Date Range: {start_date} to {end_date}
Location: Latitude {lat_min}° to {lat_max}°, Longitude {lon_min}° to {lon_max}°
Measurements: {'; '.join(measurement_summary)}
Full JSON Data: {json.dumps(properties)}
            """.strip()
            
            return document_text
            
        except Exception as e:
            logger.warning(f"Failed to create document text for {float_id}: {e}")
            # Fallback to just JSON dump
            return f"Float {float_id}: {json.dumps(properties)}"
    
    def insert_to_chromadb(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert data into ChromaDB collection.
        
        Args:
            data: List of float data dictionaries
            
        Returns:
            Summary of insertion results
        """
        try:
            logger.info(f"Inserting {len(data)} records into ChromaDB...")
            
            # Get or create the collection
            collection = self.chromadb_handler.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Argo float profile embeddings"}
            )
            
            # Prepare data for batch insertion
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for i, record in enumerate(data):
                try:
                    # Create unique ID
                    doc_id = f"profile_{i+1}_{record['float_id']}"
                    
                    # Create document text for embedding
                    document_text = self.create_document_text(
                        properties=record['properties'],
                        float_id=record['float_id'],
                        platform_number=record['platform_number']
                    )
                    
                    # Generate embedding
                    embedding = self.embedding_model.encode(document_text).tolist()
                    
                    # Prepare metadata (include the exact JSON as requested)
                    metadata = {
                        "float_id": record['float_id'],
                        "platform_number": record['platform_number'],
                        "deploy_date": record['deploy_date'],
                        "source": "supabase_sync",
                        "json_properties": json.dumps(record['properties']),  # Store exact JSON
                        "total_profiles": record['properties'].get('total_profiles', None)
                    }
                    
                    ids.append(doc_id)
                    documents.append(document_text)
                    metadatas.append(metadata)
                    embeddings.append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare record {record.get('float_id', 'unknown')}: {e}")
                    continue
            
            if not ids:
                logger.warning("No valid records to insert")
                return {"success": False, "message": "No valid records"}
            
            # Batch insert into ChromaDB
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"✓ Successfully inserted {len(ids)} records into ChromaDB collection '{self.collection_name}'")
            
            return {
                "success": True,
                "inserted_count": len(ids),
                "collection_name": self.collection_name,
                "total_records_processed": len(data)
            }
            
        except Exception as e:
            logger.error(f"Failed to insert data into ChromaDB: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the ChromaDB collection."""
        try:
            collection = self.chromadb_handler.client.get_collection(name=self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "exists": True
            }
        except Exception as e:
            logger.warning(f"Collection info error: {e}")
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "exists": False
            }
    
    def sync_data(self) -> Dict[str, Any]:
        """
        Perform complete synchronization from Supabase to ChromaDB.
        
        Returns:
            Summary of sync operation
        """
        try:
            logger.info("🚀 Starting Supabase to ChromaDB synchronization...")
            
            # Check initial state
            initial_info = self.get_collection_info()
            logger.info(f"Initial ChromaDB state: {initial_info['total_documents']} documents")
            
            # Fetch data from Supabase
            supabase_data = self.fetch_supabase_data()
            
            if not supabase_data:
                logger.warning("No data found in Supabase")
                return {"success": False, "message": "No data in Supabase"}
            
            # Insert into ChromaDB
            result = self.insert_to_chromadb(supabase_data)
            
            # Check final state
            final_info = self.get_collection_info()
            logger.info(f"Final ChromaDB state: {final_info['total_documents']} documents")
            
            logger.info("✅ Synchronization completed successfully!")
            
            return {
                "success": True,
                "supabase_records": len(supabase_data),
                "chromadb_inserted": result.get("inserted_count", 0),
                "initial_chromadb_count": initial_info['total_documents'],
                "final_chromadb_count": final_info['total_documents']
            }
            
        except Exception as e:
            logger.error(f"❌ Synchronization failed: {e}")
            return {"success": False, "error": str(e)}
    
    def close(self):
        """Close database connections."""
        try:
            if self.supabase_handler:
                self.supabase_handler.engine.dispose()
            logger.info("✓ Connections closed")
        except Exception as e:
            logger.warning(f"Error closing connections: {e}")


def main():
    """Main function to run the synchronization."""
    sync_service = None
    try:
        # Create sync service
        sync_service = SupabaseToChromaDBSync()
        
        # Perform synchronization
        result = sync_service.sync_data()
        
        # Print results
        print("\n" + "="*50)
        print("SYNCHRONIZATION RESULTS")
        print("="*50)
        
        if result["success"]:
            print(f"✅ SUCCESS!")
            print(f"📊 Supabase records found: {result.get('supabase_records', 0)}")
            print(f"📥 ChromaDB records inserted: {result.get('chromadb_inserted', 0)}")
            print(f"📈 Initial ChromaDB count: {result.get('initial_chromadb_count', 0)}")
            print(f"📊 Final ChromaDB count: {result.get('final_chromadb_count', 0)}")
        else:
            print(f"❌ FAILED: {result.get('error', result.get('message', 'Unknown error'))}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        print(f"\n❌ Script failed: {e}")
    
    finally:
        if sync_service:
            sync_service.close()


if __name__ == "__main__":
    main()