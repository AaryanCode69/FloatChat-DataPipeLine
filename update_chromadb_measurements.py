"""
Update ChromaDB with Measurements from Supabase
===============================================

This script extracts temperature, pressure, and salinity measurements from 
the JSON properties column in Supabase floats table and adds them as metadata
to the existing ChromaDB documents by matching float IDs.

Author: FloatChat Data Pipeline
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaDBMeasurementUpdater:
    """Update ChromaDB documents with measurement data from Supabase."""
    
    def __init__(self):
        """Initialize the updater service."""
        self.supabase_handler = None
        self.chromadb_handler = None
        self.collection_name = "float_embeddings"
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections."""
        try:
            # Initialize Supabase connection
            logger.info("Connecting to Supabase...")
            self.supabase_handler = SupabaseHandler()
            logger.info("✓ Supabase connection established")
            
            # Initialize ChromaDB connection
            logger.info("Connecting to ChromaDB...")
            self.chromadb_handler = ChromaDBHandler()
            logger.info("✓ ChromaDB connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    def extract_measurements_from_supabase(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract measurements from Supabase floats table.
        
        Returns:
            Dictionary mapping float_id to measurement data
        """
        try:
            logger.info("Extracting measurements from Supabase...")
            
            with self.supabase_handler.engine.connect() as conn:
                from sqlalchemy import text
                
                # Query to get float_id and properties JSON
                query = text("""
                    SELECT float_id, properties 
                    FROM floats 
                    WHERE properties IS NOT NULL
                """)
                
                result = conn.execute(query)
                records = result.fetchall()
                
                measurements_map = {}
                
                for record in records:
                    try:
                        # Parse JSON properties
                        if isinstance(record.properties, str):
                            properties_json = json.loads(record.properties)
                        else:
                            properties_json = record.properties
                        
                        # Extract measurements
                        measurements = properties_json.get('measurements', {})
                        
                        if measurements:
                            # Extract the three parameters
                            extracted_measurements = {}
                            
                            for param in ['temperature', 'pressure', 'salinity']:
                                if param in measurements and isinstance(measurements[param], dict):
                                    param_data = measurements[param]
                                    extracted_measurements[param] = {
                                        'min': param_data.get('min'),
                                        'max': param_data.get('max'),
                                        'mean': param_data.get('mean'),
                                        'count': param_data.get('count')
                                    }
                            
                            if extracted_measurements:
                                measurements_map[record.float_id] = extracted_measurements
                        
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Failed to parse JSON for float_id {record.float_id}: {e}")
                        continue
                
                logger.info(f"✓ Extracted measurements for {len(measurements_map)} floats")
                return measurements_map
                
        except Exception as e:
            logger.error(f"Failed to extract measurements from Supabase: {e}")
            raise
    
    def get_chromadb_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents from ChromaDB with their IDs and metadata.
        
        Returns:
            List of document information
        """
        try:
            logger.info("Retrieving ChromaDB documents...")
            
            collection = self.chromadb_handler.client.get_collection(self.collection_name)
            
            # Get all documents with metadata
            results = collection.get(include=["documents", "metadatas"])
            
            # Get IDs separately
            all_results = collection.get()
            document_ids = all_results.get('ids', [])
            
            documents = []
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                doc_id = document_ids[i] if i < len(document_ids) else f"doc_{i+1}"
                
                documents.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': metadata,
                    'float_id': metadata.get('float_id', '')
                })
            
            logger.info(f"✓ Retrieved {len(documents)} ChromaDB documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve ChromaDB documents: {e}")
            raise
    
    def clean_float_id(self, float_id: str) -> str:
        """Clean float ID by removing prefixes and byte string markers."""
        if not float_id:
            return ""
        
        # Remove byte string markers like b'...'
        if float_id.startswith("b'") and float_id.endswith("'"):
            float_id = float_id[2:-1]
        
        # Remove common prefixes
        float_id = float_id.strip()
        
        return float_id
    
    def match_float_ids(self, chromadb_docs: List[Dict[str, Any]], 
                       measurements_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Match ChromaDB float IDs with Supabase float IDs.
        
        Args:
            chromadb_docs: List of ChromaDB documents
            measurements_map: Map of Supabase float_id to measurements
            
        Returns:
            Dictionary mapping ChromaDB doc_id to measurements
        """
        try:
            logger.info("Matching float IDs between ChromaDB and Supabase...")
            
            matches = {}
            unmatched_chromadb = []
            unmatched_supabase = list(measurements_map.keys())
            
            for doc in chromadb_docs:
                chromadb_float_id = self.clean_float_id(doc['float_id'])
                
                # Try to find a match in Supabase data
                matched = False
                for supabase_float_id in measurements_map.keys():
                    supabase_clean_id = self.clean_float_id(supabase_float_id)
                    
                    # Check various matching patterns
                    if (chromadb_float_id == supabase_clean_id or
                        chromadb_float_id == supabase_float_id or
                        chromadb_float_id in supabase_float_id or
                        supabase_float_id in chromadb_float_id):
                        
                        matches[doc['id']] = measurements_map[supabase_float_id]
                        if supabase_float_id in unmatched_supabase:
                            unmatched_supabase.remove(supabase_float_id)
                        matched = True
                        break
                
                if not matched:
                    unmatched_chromadb.append(chromadb_float_id)
            
            logger.info(f"✓ Matching results:")
            logger.info(f"  - Matched: {len(matches)} documents")
            logger.info(f"  - Unmatched ChromaDB: {len(unmatched_chromadb)}")
            logger.info(f"  - Unmatched Supabase: {len(unmatched_supabase)}")
            
            if len(unmatched_chromadb) > 0:
                logger.info(f"  - Sample unmatched ChromaDB IDs: {unmatched_chromadb[:5]}")
            if len(unmatched_supabase) > 0:
                logger.info(f"  - Sample unmatched Supabase IDs: {unmatched_supabase[:5]}")
            
            return matches
            
        except Exception as e:
            logger.error(f"Failed to match float IDs: {e}")
            raise
    
    def update_chromadb_metadata(self, matches: Dict[str, Dict[str, Any]]) -> bool:
        """
        Update ChromaDB documents with measurement metadata.
        
        Args:
            matches: Dictionary mapping doc_id to measurements
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not matches:
                logger.info("No matches found to update")
                return True
            
            logger.info(f"Updating {len(matches)} ChromaDB documents with measurements...")
            
            collection = self.chromadb_handler.client.get_collection(self.collection_name)
            
            # Get current documents to preserve existing data
            results = collection.get(include=["documents", "metadatas"])
            all_results = collection.get()
            document_ids = all_results.get('ids', [])
            
            updates_made = 0
            
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                doc_id = document_ids[i] if i < len(document_ids) else f"doc_{i+1}"
                
                if doc_id in matches:
                    # Create updated metadata with measurements
                    updated_metadata = metadata.copy()
                    measurements = matches[doc_id]
                    
                    # Add measurements as flattened metadata fields
                    for param, data in measurements.items():
                        if data and isinstance(data, dict):
                            updated_metadata[f"{param}_min"] = data.get('min')
                            updated_metadata[f"{param}_max"] = data.get('max')
                            updated_metadata[f"{param}_mean"] = data.get('mean')
                            updated_metadata[f"{param}_count"] = data.get('count')
                    
                    # Also store a flag indicating measurements are present
                    updated_metadata['has_measurements'] = True
                    updated_metadata['measurements_updated'] = True
                    
                    # Update the document (we need to delete and re-add with new metadata)
                    try:
                        # Delete the old document
                        collection.delete(ids=[doc_id])
                        
                        # Add back with updated metadata
                        collection.add(
                            ids=[doc_id],
                            documents=[doc],
                            metadatas=[updated_metadata]
                        )
                        
                        updates_made += 1
                        
                        if updates_made % 10 == 0:
                            logger.info(f"  Progress: {updates_made}/{len(matches)} documents updated")
                        
                    except Exception as e:
                        logger.error(f"Failed to update document {doc_id}: {e}")
                        continue
            
            logger.info(f"✓ Successfully updated {updates_made} ChromaDB documents with measurements")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update ChromaDB metadata: {e}")
            return False
    
    def verify_updates(self) -> Dict[str, Any]:
        """Verify that measurements were added successfully."""
        try:
            logger.info("Verifying measurement updates...")
            
            collection = self.chromadb_handler.client.get_collection(self.collection_name)
            results = collection.get(include=["metadatas"])
            
            total_docs = len(results['metadatas'])
            docs_with_measurements = 0
            measurement_types = {'temperature': 0, 'pressure': 0, 'salinity': 0}
            
            for metadata in results['metadatas']:
                if metadata.get('has_measurements', False):
                    docs_with_measurements += 1
                
                # Count each measurement type
                for param in ['temperature', 'pressure', 'salinity']:
                    if f"{param}_min" in metadata:
                        measurement_types[param] += 1
            
            logger.info(f"✓ Verification results:")
            logger.info(f"  - Total documents: {total_docs}")
            logger.info(f"  - Documents with measurements: {docs_with_measurements}")
            logger.info(f"  - Temperature measurements: {measurement_types['temperature']}")
            logger.info(f"  - Pressure measurements: {measurement_types['pressure']}")
            logger.info(f"  - Salinity measurements: {measurement_types['salinity']}")
            
            return {
                'total_documents': total_docs,
                'documents_with_measurements': docs_with_measurements,
                'measurement_types': measurement_types,
                'success': docs_with_measurements > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to verify updates: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_measurements(self) -> Dict[str, Any]:
        """
        Perform complete measurement update process.
        
        Returns:
            Summary of update operation
        """
        try:
            logger.info("🔄 Starting ChromaDB measurement update process...")
            
            # Step 1: Extract measurements from Supabase
            measurements_map = self.extract_measurements_from_supabase()
            
            if not measurements_map:
                logger.warning("No measurements found in Supabase")
                return {'success': False, 'message': 'No measurements found in Supabase'}
            
            # Step 2: Get ChromaDB documents
            chromadb_docs = self.get_chromadb_documents()
            
            if not chromadb_docs:
                logger.warning("No documents found in ChromaDB")
                return {'success': False, 'message': 'No documents found in ChromaDB'}
            
            # Step 3: Match float IDs
            matches = self.match_float_ids(chromadb_docs, measurements_map)
            
            if not matches:
                logger.warning("No matching float IDs found")
                return {'success': False, 'message': 'No matching float IDs found'}
            
            # Step 4: Update ChromaDB metadata
            update_success = self.update_chromadb_metadata(matches)
            
            if not update_success:
                return {'success': False, 'message': 'Failed to update ChromaDB metadata'}
            
            # Step 5: Verify updates
            verification = self.verify_updates()
            
            logger.info("✅ Measurement update process completed!")
            
            return {
                'success': True,
                'supabase_measurements': len(measurements_map),
                'chromadb_documents': len(chromadb_docs),
                'matched_documents': len(matches),
                'updated_documents': verification.get('documents_with_measurements', 0),
                'measurement_counts': verification.get('measurement_types', {}),
                'verification_success': verification.get('success', False)
            }
            
        except Exception as e:
            logger.error(f"❌ Measurement update failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Close database connections."""
        try:
            if self.supabase_handler:
                self.supabase_handler.engine.dispose()
            logger.info("✓ Connections closed")
        except Exception as e:
            logger.warning(f"Error closing connections: {e}")


def main():
    """Main function to run the measurement update."""
    updater = None
    try:
        print("🔄 ChromaDB Measurement Update Tool")
        print("=" * 50)
        print("This tool extracts measurements from Supabase and adds them to ChromaDB metadata.")
        print()
        
        # Create updater service
        updater = ChromaDBMeasurementUpdater()
        
        # Perform update
        result = updater.update_measurements()
        
        # Display results
        print("\n" + "=" * 50)
        print("UPDATE RESULTS")
        print("=" * 50)
        
        if result['success']:
            print("✅ Update completed successfully!")
            print(f"📊 Supabase measurements found: {result.get('supabase_measurements', 0)}")
            print(f"📚 ChromaDB documents: {result.get('chromadb_documents', 0)}")
            print(f"🔗 Matched documents: {result.get('matched_documents', 0)}")
            print(f"📝 Updated documents: {result.get('updated_documents', 0)}")
            
            measurement_counts = result.get('measurement_counts', {})
            print(f"\n📏 Measurement counts:")
            print(f"  - Temperature: {measurement_counts.get('temperature', 0)}")
            print(f"  - Pressure: {measurement_counts.get('pressure', 0)}")
            print(f"  - Salinity: {measurement_counts.get('salinity', 0)}")
            
            if result.get('verification_success'):
                print("✅ All measurements verified successfully!")
            else:
                print("⚠️  Some verification issues detected")
        else:
            print(f"❌ Update failed: {result.get('error', result.get('message', 'Unknown error'))}")
        
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n⚠️  Update cancelled by user")
    except Exception as e:
        print(f"\n❌ Update failed: {e}")
        logger.error(f"Script execution failed: {e}")
    
    finally:
        if updater:
            updater.close()


if __name__ == "__main__":
    main()