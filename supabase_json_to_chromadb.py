"""
Supabase JSON to ChromaDB Embedder
==================================

This script pulls JSON data from the 'properties' column of the 'floats' table 
in Supabase and embeds it into ChromaDB exactly as JSON strings.

Author: FloatChat Data Pipeline
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from supabase import create_client, Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SupabaseToChromaDBEmbedder:
    """Handles embedding Supabase JSON data into ChromaDB."""
    
    def __init__(self):
        """Initialize the embedder with database connections."""
        self.supabase_client: Optional[Client] = None
        self.chromadb_client: Optional[chromadb.Client] = None
        self.collection = None
        self.collection_name = "float_profiles"
        self._load_environment()
        self._setup_connections()
    
    def _load_environment(self):
        """Load environment variables."""
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("✓ Environment variables loaded")
        else:
            logger.warning("⚠️  .env file not found")
    
    def _setup_connections(self):
        """Setup connections to Supabase and ChromaDB."""
        try:
            # Setup Supabase connection
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_ANON_KEY')
            
            if not supabase_url or not supabase_key:
                # Fallback: construct URL from individual components
                db_host = os.getenv('SUPABASE_DB_HOST')
                db_port = os.getenv('SUPABASE_DB_PORT', '5432')
                db_name = os.getenv('SUPABASE_DB_NAME', 'postgres')
                db_user = os.getenv('SUPABASE_DB_USER')
                db_password = os.getenv('SUPABASE_DB_PASSWORD')
                
                if not all([db_host, db_user, db_password]):
                    raise ValueError("Missing required Supabase credentials")
                
                # For direct database access, we'll use a different approach
                logger.info("Using direct database connection approach")
                self._setup_direct_db_connection()
            else:
                self.supabase_client = create_client(supabase_url, supabase_key)
                logger.info("✓ Connected to Supabase")
            
            # Setup ChromaDB connection
            chromadb_host = os.getenv('CHROMADB_HOST', 'localhost')
            chromadb_port = os.getenv('CHROMADB_PORT', '8000')
            
            self.chromadb_client = chromadb.HttpClient(
                host=chromadb_host,
                port=int(chromadb_port)
            )
            
            # Test ChromaDB connection
            self.chromadb_client.heartbeat()
            logger.info(f"✓ Connected to ChromaDB at {chromadb_host}:{chromadb_port}")
            
            # Create or get collection
            self._setup_collection()
            
        except Exception as e:
            logger.error(f"❌ Failed to setup connections: {e}")
            raise
    
    def _setup_direct_db_connection(self):
        """Setup direct database connection for Supabase."""
        import psycopg2
        import psycopg2.extras
        
        try:
            self.db_connection = psycopg2.connect(
                host=os.getenv('SUPABASE_DB_HOST'),
                port=os.getenv('SUPABASE_DB_PORT', '5432'),
                database=os.getenv('SUPABASE_DB_NAME', 'postgres'),
                user=os.getenv('SUPABASE_DB_USER'),
                password=os.getenv('SUPABASE_DB_PASSWORD')
            )
            logger.info("✓ Connected to Supabase via direct database connection")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def _setup_collection(self):
        """Create or get the ChromaDB collection."""
        try:
            # Try to get existing collection
            self.collection = self.chromadb_client.get_collection(self.collection_name)
            logger.info(f"✓ Using existing collection: {self.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            self.collection = self.chromadb_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Supabase floats JSON data embeddings"}
            )
            logger.info(f"✓ Created new collection: {self.collection_name}")
    
    def fetch_float_data_from_supabase(self) -> List[Dict[str, Any]]:
        """
        Fetch float data from Supabase floats table.
        
        Returns:
            List of dictionaries containing float data with properties JSON
        """
        try:
            if hasattr(self, 'db_connection'):
                # Direct database query
                return self._fetch_via_direct_db()
            else:
                # Supabase client query
                return self._fetch_via_supabase_client()
        except Exception as e:
            logger.error(f"❌ Failed to fetch data from Supabase: {e}")
            raise
    
    def _fetch_via_direct_db(self) -> List[Dict[str, Any]]:
        """Fetch data using direct database connection."""
        import psycopg2.extras
        
        cursor = self.db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
        SELECT id, float_id, properties 
        FROM floats 
        WHERE properties IS NOT NULL
        ORDER BY id
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        
        logger.info(f"✓ Fetched {len(results)} float records from Supabase")
        return [dict(row) for row in results]
    
    def _fetch_via_supabase_client(self) -> List[Dict[str, Any]]:
        """Fetch data using Supabase client."""
        response = self.supabase_client.table("floats").select("id, float_id, properties").execute()
        
        if response.data:
            logger.info(f"✓ Fetched {len(response.data)} float records from Supabase")
            return response.data
        else:
            logger.warning("⚠️  No data found in floats table")
            return []
    
    def embed_json_data(self, float_data: List[Dict[str, Any]], batch_size: int = 50):
        """
        Embed JSON data into ChromaDB.
        
        Args:
            float_data: List of float records from Supabase
            batch_size: Number of records to process in each batch
        """
        if not float_data:
            logger.warning("⚠️  No data to embed")
            return
        
        logger.info(f"🔄 Starting to embed {len(float_data)} JSON records...")
        
        # Process in batches
        for i in range(0, len(float_data), batch_size):
            batch = float_data[i:i + batch_size]
            self._process_batch(batch, i // batch_size + 1, len(float_data), batch_size)
        
        logger.info("✅ Completed embedding all JSON data into ChromaDB")
    
    def _process_batch(self, batch: List[Dict[str, Any]], batch_num: int, total_records: int, batch_size: int):
        """Process a single batch of records."""
        try:
            ids = []
            documents = []
            metadatas = []
            
            for record in batch:
                # Extract data
                record_id = record.get('id')
                float_id = record.get('float_id')
                properties_json = record.get('properties')
                
                if not properties_json:
                    logger.warning(f"⚠️  Skipping record {record_id}: No properties data")
                    continue
                
                # Create unique ID for ChromaDB
                unique_id = f"float_{record_id}_{float_id}" if float_id else f"float_{record_id}"
                
                # Convert JSON to string for embedding
                json_document = json.dumps(properties_json, ensure_ascii=False)
                
                # Prepare metadata
                metadata = {
                    "source": "supabase_floats",
                    "supabase_id": record_id,
                    "float_id": str(float_id) if float_id else None,
                    "data_type": "json_properties"
                }
                
                ids.append(unique_id)
                documents.append(json_document)
                metadatas.append(metadata)
            
            if ids:
                # Add to ChromaDB collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
                processed = len(ids)
                total_processed = min(batch_num * batch_size, total_records)
                
                logger.info(f"✓ Processed batch {batch_num}: {processed} records | Total: {total_processed}/{total_records}")
            else:
                logger.warning(f"⚠️  Batch {batch_num} had no valid records to process")
                
        except Exception as e:
            logger.error(f"❌ Failed to process batch {batch_num}: {e}")
            raise
    
    def verify_embeddings(self) -> Dict[str, Any]:
        """
        Verify that embeddings were created successfully.
        
        Returns:
            Dictionary with verification results
        """
        try:
            # Get collection count
            count = self.collection.count()
            
            # Get a sample of embedded data
            sample = self.collection.get(
                limit=3,
                include=["documents", "metadatas", "embeddings"]
            )
            
            results = {
                "total_embedded": count,
                "has_embeddings": bool(sample.get("embeddings")),
                "sample_data": {
                    "ids": sample.get("ids", [])[:3],
                    "documents_preview": [doc[:100] + "..." if len(doc) > 100 else doc 
                                        for doc in (sample.get("documents", []))[:3]],
                    "metadatas": sample.get("metadatas", [])[:3]
                }
            }
            
            logger.info(f"✓ Verification complete: {count} records embedded")
            
            if sample.get("embeddings"):
                embedding_dim = len(sample["embeddings"][0]) if sample["embeddings"] else 0
                logger.info(f"✓ Embeddings generated successfully (dimension: {embedding_dim})")
                results["embedding_dimension"] = embedding_dim
            else:
                logger.warning("⚠️  No embeddings found in sample")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return {"error": str(e)}
    
    def test_semantic_search(self, query: str = "temperature measurements", n_results: int = 3):
        """
        Test semantic search functionality.
        
        Args:
            query: Search query
            n_results: Number of results to return
        """
        try:
            logger.info(f"🔍 Testing semantic search with query: '{query}'")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if results["ids"][0]:
                logger.info(f"✓ Found {len(results['ids'][0])} results")
                
                for i, (doc_id, distance, metadata) in enumerate(zip(
                    results["ids"][0],
                    results["distances"][0],
                    results["metadatas"][0]
                )):
                    logger.info(f"  {i+1}. ID: {doc_id} | Distance: {distance:.4f} | Float: {metadata.get('float_id', 'N/A')}")
                
                return results
            else:
                logger.warning("⚠️  No search results found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Semantic search test failed: {e}")
            return None
    
    def run_embedding_pipeline(self):
        """Run the complete embedding pipeline."""
        try:
            logger.info("🚀 Starting Supabase JSON to ChromaDB embedding pipeline")
            
            # Step 1: Fetch data from Supabase
            logger.info("📥 Step 1: Fetching data from Supabase...")
            float_data = self.fetch_float_data_from_supabase()
            
            if not float_data:
                logger.error("❌ No data found in Supabase. Exiting.")
                return False
            
            # Step 2: Embed JSON data into ChromaDB
            logger.info("🔄 Step 2: Embedding JSON data into ChromaDB...")
            self.embed_json_data(float_data)
            
            # Step 3: Verify embeddings
            logger.info("✅ Step 3: Verifying embeddings...")
            verification = self.verify_embeddings()
            
            if verification.get("error"):
                logger.error(f"❌ Verification failed: {verification['error']}")
                return False
            
            # Step 4: Test semantic search
            logger.info("🔍 Step 4: Testing semantic search...")
            self.test_semantic_search()
            
            logger.info("🎉 Pipeline completed successfully!")
            logger.info(f"📊 Summary: {verification['total_embedded']} JSON records embedded with {verification.get('embedding_dimension', 'unknown')} dimensional embeddings")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False
    
    def close_connections(self):
        """Close database connections."""
        try:
            if hasattr(self, 'db_connection'):
                self.db_connection.close()
                logger.info("✓ Closed database connection")
        except Exception as e:
            logger.warning(f"⚠️  Error closing connections: {e}")


def main():
    """Main function to run the embedding pipeline."""
    embedder = None
    try:
        # Create embedder instance
        embedder = SupabaseToChromaDBEmbedder()
        
        # Run the pipeline
        success = embedder.run_embedding_pipeline()
        
        if success:
            logger.info("✅ All operations completed successfully!")
        else:
            logger.error("❌ Pipeline execution failed!")
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        if embedder:
            embedder.close_connections()


if __name__ == "__main__":
    main()