"""
Test file to verify vector embeddings in ChromaDB and create them if missing.
This script checks if embeddings exist and generates them if needed.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
from ingest.db_handler import ChromaDBHandler

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChromaDBEmbeddingTester:
    """Test and verify vector embeddings in ChromaDB."""
    
    def __init__(self):
        """Initialize the tester."""
        self.chromadb_handler = None
        self.embedding_model = None
        self.collection_name = "float_embeddings"
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ChromaDB connection and embedding model."""
        try:
            # Initialize ChromaDB handler
            logger.info("Connecting to ChromaDB...")
            self.chromadb_handler = ChromaDBHandler()
            
            # Get the collection
            try:
                self.collection = self.chromadb_handler.client.get_collection(self.collection_name)
                logger.info(f"✓ Connected to collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"✗ Failed to get collection '{self.collection_name}': {e}")
                raise
            
            # Initialize embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def check_collection_info(self):
        """Check basic collection information."""
        try:
            count = self.collection.count()
            logger.info(f"Collection '{self.collection_name}' contains {count} entries")
            
            if count == 0:
                logger.warning("⚠️  Collection is empty!")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking collection info: {e}")
            return False
    
    def check_embeddings_exist(self, sample_size: int = 5) -> Dict[str, Any]:
        """
        Check if embeddings exist in the collection.
        
        Args:
            sample_size: Number of entries to sample for testing
            
        Returns:
            Dictionary with embedding check results
        """
        try:
            logger.info(f"Checking embeddings for {sample_size} sample entries...")
            
            # Get sample entries
            results = self.collection.get(
                limit=sample_size,
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not results['ids']:
                return {
                    'has_embeddings': False,
                    'total_entries': 0,
                    'entries_with_embeddings': 0,
                    'embedding_dimension': None,
                    'sample_embeddings': []
                }
            
            total_entries = len(results['ids'])
            entries_with_embeddings = 0
            embedding_dimension = None
            sample_embeddings = []
            
            # Check each entry for embeddings
            for i, (doc_id, embedding) in enumerate(zip(results['ids'], results.get('embeddings', []))):
                if embedding is not None and len(embedding) > 0:
                    entries_with_embeddings += 1
                    if embedding_dimension is None:
                        embedding_dimension = len(embedding)
                    
                    # Store first few for inspection
                    if len(sample_embeddings) < 3:
                        sample_embeddings.append({
                            'id': doc_id,
                            'embedding_length': len(embedding),
                            'embedding_sample': embedding[:5]  # First 5 values
                        })
                    
                    logger.info(f"✓ Entry {i+1}: ID='{doc_id}' has embedding (dim={len(embedding)})")
                else:
                    logger.warning(f"✗ Entry {i+1}: ID='{doc_id}' missing embedding")
            
            has_embeddings = entries_with_embeddings == total_entries
            
            result = {
                'has_embeddings': has_embeddings,
                'total_entries': total_entries,
                'entries_with_embeddings': entries_with_embeddings,
                'embedding_dimension': embedding_dimension,
                'sample_embeddings': sample_embeddings
            }
            
            if has_embeddings:
                logger.info(f"✓ All {total_entries} entries have embeddings (dimension: {embedding_dimension})")
            else:
                logger.warning(f"⚠️  Only {entries_with_embeddings}/{total_entries} entries have embeddings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking embeddings: {e}")
            return {
                'has_embeddings': False,
                'total_entries': 0,
                'entries_with_embeddings': 0,
                'embedding_dimension': None,
                'sample_embeddings': [],
                'error': str(e)
            }
    
    def test_vector_similarity_search(self, query_text: str = "Antarctic ocean temperature measurements") -> Dict[str, Any]:
        """
        Test vector similarity search functionality.
        
        Args:
            query_text: Text to search for
            
        Returns:
            Search results and performance metrics
        """
        try:
            logger.info(f"Testing vector similarity search with query: '{query_text}'")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['ids'][0]:
                return {
                    'success': False,
                    'error': 'No results returned from similarity search'
                }
            
            search_results = []
            for i, (doc_id, document, metadata, distance) in enumerate(zip(
                results['ids'][0], 
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                search_results.append({
                    'rank': i + 1,
                    'id': doc_id,
                    'distance': distance,
                    'float_id': metadata.get('float_id', 'N/A'),
                    'location': f"{metadata.get('lat', 'N/A')}°, {metadata.get('lon', 'N/A')}°",
                    'document_preview': document[:100] + "..." if len(document) > 100 else document
                })
                
                logger.info(f"Result {i+1}: Float {metadata.get('float_id')} (distance={distance:.4f})")
            
            return {
                'success': True,
                'query': query_text,
                'num_results': len(search_results),
                'results': search_results
            }
            
        except Exception as e:
            logger.error(f"Error in similarity search test: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def regenerate_missing_embeddings(self, batch_size: int = 50) -> bool:
        """
        Regenerate embeddings for entries that are missing them.
        
        Args:
            batch_size: Number of entries to process at once
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Checking for entries missing embeddings...")
            
            # Get all entries
            all_results = self.collection.get(include=["documents", "metadatas", "embeddings"])
            
            if not all_results['ids']:
                logger.warning("No entries found in collection")
                return False
            
            # Find entries missing embeddings
            missing_embeddings = []
            for i, (doc_id, document, metadata, embedding) in enumerate(zip(
                all_results['ids'],
                all_results['documents'],
                all_results['metadatas'],
                all_results.get('embeddings', [])
            )):
                if embedding is None or len(embedding) == 0:
                    missing_embeddings.append({
                        'id': doc_id,
                        'document': document,
                        'metadata': metadata
                    })
            
            if not missing_embeddings:
                logger.info("✓ All entries already have embeddings")
                return True
            
            logger.info(f"Found {len(missing_embeddings)} entries missing embeddings")
            
            # Process in batches
            for i in range(0, len(missing_embeddings), batch_size):
                batch = missing_embeddings[i:i + batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(missing_embeddings) + batch_size - 1)//batch_size}")
                
                # Generate embeddings for batch
                documents = [entry['document'] for entry in batch]
                embeddings = self.embedding_model.encode(documents)
                
                # Update entries with new embeddings
                for j, entry in enumerate(batch):
                    try:
                        # Update the entry with embedding
                        self.collection.update(
                            ids=[entry['id']],
                            embeddings=[embeddings[j].tolist()],
                            documents=[entry['document']],
                            metadatas=[entry['metadata']]
                        )
                        
                        logger.debug(f"Updated embedding for entry: {entry['id']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to update embedding for {entry['id']}: {e}")
                        continue
                
                logger.info(f"Completed batch {i//batch_size + 1}")
            
            logger.info(f"✓ Successfully regenerated embeddings for {len(missing_embeddings)} entries")
            return True
            
        except Exception as e:
            logger.error(f"Error regenerating embeddings: {e}")
            return False
    
    def run_full_test(self):
        """Run complete embedding verification and repair."""
        try:
            logger.info("=" * 60)
            logger.info("CHROMADB EMBEDDING VERIFICATION TEST")
            logger.info("=" * 60)
            
            # Step 1: Check collection info
            logger.info("\n1. Checking collection information...")
            if not self.check_collection_info():
                logger.error("Collection check failed")
                return False
            
            # Step 2: Check embeddings
            logger.info("\n2. Checking embeddings...")
            embedding_status = self.check_embeddings_exist(sample_size=10)
            
            # Step 3: Regenerate if needed
            if not embedding_status['has_embeddings']:
                logger.info("\n3. Regenerating missing embeddings...")
                if self.regenerate_missing_embeddings():
                    logger.info("✓ Embeddings regenerated successfully")
                    # Re-check after regeneration
                    embedding_status = self.check_embeddings_exist(sample_size=5)
                else:
                    logger.error("✗ Failed to regenerate embeddings")
                    return False
            else:
                logger.info("\n3. ✓ All embeddings present, skipping regeneration")
            
            # Step 4: Test similarity search
            logger.info("\n4. Testing vector similarity search...")
            search_results = self.test_vector_similarity_search()
            
            if search_results['success']:
                logger.info("✓ Vector similarity search working correctly")
                logger.info(f"Found {search_results['num_results']} relevant results")
            else:
                logger.error(f"✗ Vector similarity search failed: {search_results.get('error')}")
                return False
            
            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✓ Collection: {self.collection_name}")
            logger.info(f"✓ Total entries: {embedding_status['total_entries']}")
            logger.info(f"✓ Entries with embeddings: {embedding_status['entries_with_embeddings']}")
            logger.info(f"✓ Embedding dimension: {embedding_status['embedding_dimension']}")
            logger.info(f"✓ Vector search: Working")
            logger.info("✓ ChromaDB embedding system is fully functional!")
            
            return True
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            return False
    
    def close(self):
        """Close connections."""
        if self.chromadb_handler:
            self.chromadb_handler.close()


def main():
    """Main function to run the embedding test."""
    tester = None
    try:
        tester = ChromaDBEmbeddingTester()
        success = tester.run_full_test()
        
        if success:
            logger.info("\n🎉 All tests passed! ChromaDB embeddings are working correctly.")
        else:
            logger.error("\n❌ Some tests failed. Please check the logs above.")
            
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
    finally:
        if tester:
            tester.close()


if __name__ == "__main__":
    main()