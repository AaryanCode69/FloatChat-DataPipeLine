"""
ChromaDB Cleanup Script
======================

This script removes newly added JSON embedded data from ChromaDB and restores
the original 155 semantic descriptions. It identifies and deletes documents
that contain raw JSON data while preserving the original descriptive summaries.

Author: FloatChat Data Pipeline
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaDBCleanup:
    """Clean up ChromaDB by removing JSON embedded data."""
    
    def __init__(self):
        """Initialize the cleanup service."""
        self.chromadb_handler = None
        self.collection_name = "float_embeddings"
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize ChromaDB connection."""
        try:
            logger.info("Connecting to ChromaDB...")
            self.chromadb_handler = ChromaDBHandler()
            logger.info("✓ ChromaDB connection established")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def analyze_collection(self) -> Dict[str, Any]:
        """Analyze the current collection to identify different types of documents."""
        try:
            logger.info("Analyzing ChromaDB collection...")
            
            collection = self.chromadb_handler.client.get_collection(self.collection_name)
            results = collection.get(include=["documents", "metadatas"])
            
            total_docs = len(results['documents'])
            logger.info(f"Found {total_docs} total documents")
            
            # Categorize documents
            semantic_docs = []
            json_docs = []
            
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                doc_id = f"doc_{i+1}"  # Default ID
                
                # Try to get the actual document ID
                all_results = collection.get()
                if 'ids' in all_results and i < len(all_results['ids']):
                    doc_id = all_results['ids'][i]
                
                # Check if document is JSON format
                try:
                    parsed_json = json.loads(doc)
                    # If it parses successfully and has expected JSON structure, it's a JSON doc
                    if isinstance(parsed_json, dict) and any(key in parsed_json for key in ['date_range', 'measurements', 'location_range']):
                        json_docs.append({
                            'id': doc_id,
                            'index': i,
                            'metadata': metadata,
                            'type': 'json_embedded'
                        })
                    else:
                        semantic_docs.append({
                            'id': doc_id,
                            'index': i,
                            'metadata': metadata,
                            'type': 'semantic_description'
                        })
                except (json.JSONDecodeError, TypeError):
                    # If it doesn't parse as JSON, it's likely a semantic description
                    semantic_docs.append({
                        'id': doc_id,
                        'index': i,
                        'metadata': metadata,
                        'type': 'semantic_description'
                    })
            
            logger.info(f"✓ Analysis complete:")
            logger.info(f"  - Semantic descriptions: {len(semantic_docs)}")
            logger.info(f"  - JSON embedded docs: {len(json_docs)}")
            
            return {
                'total_documents': total_docs,
                'semantic_docs': semantic_docs,
                'json_docs': json_docs
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze collection: {e}")
            raise
    
    def remove_json_documents(self, json_docs: List[Dict[str, Any]]) -> bool:
        """Remove JSON embedded documents from ChromaDB."""
        try:
            if not json_docs:
                logger.info("No JSON documents to remove")
                return True
            
            logger.info(f"Removing {len(json_docs)} JSON embedded documents...")
            
            collection = self.chromadb_handler.client.get_collection(self.collection_name)
            
            # Get all document IDs to remove
            ids_to_remove = [doc['id'] for doc in json_docs]
            
            logger.info(f"Document IDs to remove: {ids_to_remove[:5]}..." if len(ids_to_remove) > 5 else f"Document IDs to remove: {ids_to_remove}")
            
            # Remove documents by ID
            collection.delete(ids=ids_to_remove)
            
            logger.info(f"✓ Successfully removed {len(ids_to_remove)} JSON documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove JSON documents: {e}")
            return False
    
    def verify_cleanup(self) -> Dict[str, Any]:
        """Verify the cleanup operation was successful."""
        try:
            logger.info("Verifying cleanup results...")
            
            collection = self.chromadb_handler.client.get_collection(self.collection_name)
            results = collection.get(include=["documents", "metadatas"])
            
            remaining_docs = len(results['documents'])
            
            # Quick check - count how many are JSON vs semantic
            json_count = 0
            semantic_count = 0
            
            for doc in results['documents']:
                try:
                    parsed_json = json.loads(doc)
                    if isinstance(parsed_json, dict) and any(key in parsed_json for key in ['date_range', 'measurements', 'location_range']):
                        json_count += 1
                    else:
                        semantic_count += 1
                except (json.JSONDecodeError, TypeError):
                    semantic_count += 1
            
            logger.info(f"✓ Cleanup verification:")
            logger.info(f"  - Total remaining documents: {remaining_docs}")
            logger.info(f"  - Semantic descriptions: {semantic_count}")
            logger.info(f"  - JSON documents: {json_count}")
            
            return {
                'total_remaining': remaining_docs,
                'semantic_count': semantic_count,
                'json_count': json_count,
                'cleanup_successful': json_count == 0 and semantic_count > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to verify cleanup: {e}")
            return {'cleanup_successful': False, 'error': str(e)}
    
    def cleanup_chromadb(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Perform complete cleanup of ChromaDB.
        
        Args:
            confirm: If True, perform the cleanup. If False, just analyze.
            
        Returns:
            Summary of cleanup operation
        """
        try:
            logger.info("🧹 Starting ChromaDB cleanup operation...")
            
            # Step 1: Analyze current state
            analysis = self.analyze_collection()
            
            if not confirm:
                logger.info("🔍 ANALYSIS MODE - No changes made")
                return {
                    'analysis_only': True,
                    'total_documents': analysis['total_documents'],
                    'semantic_docs': len(analysis['semantic_docs']),
                    'json_docs_to_remove': len(analysis['json_docs']),
                    'would_remain': len(analysis['semantic_docs'])
                }
            
            # Step 2: Remove JSON documents
            if analysis['json_docs']:
                success = self.remove_json_documents(analysis['json_docs'])
                if not success:
                    return {'success': False, 'error': 'Failed to remove JSON documents'}
            else:
                logger.info("No JSON documents found to remove")
            
            # Step 3: Verify cleanup
            verification = self.verify_cleanup()
            
            logger.info("✅ ChromaDB cleanup completed!")
            
            return {
                'success': True,
                'original_total': analysis['total_documents'],
                'json_removed': len(analysis['json_docs']),
                'semantic_preserved': verification['semantic_count'],
                'final_total': verification['total_remaining'],
                'cleanup_successful': verification['cleanup_successful']
            }
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Close database connections."""
        try:
            if self.chromadb_handler:
                # ChromaDB doesn't require explicit closing
                pass
            logger.info("✓ Connections closed")
        except Exception as e:
            logger.warning(f"Error closing connections: {e}")


def main():
    """Main function to run the cleanup."""
    cleanup_service = None
    try:
        print("🧹 ChromaDB Cleanup Tool")
        print("=" * 50)
        print("This tool will remove JSON embedded data and keep only semantic descriptions.")
        print()
        
        # Create cleanup service
        cleanup_service = ChromaDBCleanup()
        
        # First, analyze without making changes
        print("🔍 Analyzing current ChromaDB state...")
        analysis_result = cleanup_service.cleanup_chromadb(confirm=False)
        
        print("\n📊 ANALYSIS RESULTS:")
        print(f"  Total documents: {analysis_result['total_documents']}")
        print(f"  Semantic descriptions: {analysis_result['semantic_docs']}")
        print(f"  JSON documents to remove: {analysis_result['json_docs_to_remove']}")
        print(f"  Will remain after cleanup: {analysis_result['would_remain']}")
        
        if analysis_result['json_docs_to_remove'] == 0:
            print("\n✅ No JSON documents found to remove. Collection is already clean!")
            return
        
        # Ask for confirmation
        print(f"\n⚠️  This will PERMANENTLY DELETE {analysis_result['json_docs_to_remove']} JSON documents")
        print(f"   and keep {analysis_result['semantic_docs']} semantic descriptions.")
        print()
        
        confirm = input("Do you want to proceed with cleanup? (yes/no): ").strip().lower()
        
        if confirm not in ['yes', 'y']:
            print("❌ Cleanup cancelled by user")
            return
        
        # Perform actual cleanup
        print("\n🗑️  Performing cleanup...")
        cleanup_result = cleanup_service.cleanup_chromadb(confirm=True)
        
        # Display results
        print("\n" + "=" * 50)
        print("CLEANUP RESULTS")
        print("=" * 50)
        
        if cleanup_result['success']:
            print("✅ Cleanup completed successfully!")
            print(f"📊 Original total: {cleanup_result['original_total']} documents")
            print(f"🗑️  JSON removed: {cleanup_result['json_removed']} documents")
            print(f"📚 Semantic preserved: {cleanup_result['semantic_preserved']} documents")
            print(f"📋 Final total: {cleanup_result['final_total']} documents")
            
            if cleanup_result['cleanup_successful']:
                print("✅ All JSON documents successfully removed!")
            else:
                print("⚠️  Some JSON documents may still remain")
        else:
            print(f"❌ Cleanup failed: {cleanup_result.get('error', 'Unknown error')}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Cleanup cancelled by user")
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        logger.error(f"Script execution failed: {e}")
    
    finally:
        if cleanup_service:
            cleanup_service.close()


if __name__ == "__main__":
    main()