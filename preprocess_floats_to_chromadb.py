"""
Preprocessing script to convert JSON float data from Supabase floats table 
into semantic text entries and store them in ChromaDB.

This script:
1. Fetches rows from the floats table in Supabase
2. Extracts relevant fields from the properties JSONB column
3. Generates semantic natural language summaries
4. Creates ChromaDB entries with embeddings
5. Stores them in a ChromaDB collection named 'float_embeddings'
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

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


class FloatsToChromaDBProcessor:
    """Processes float data from Supabase and stores semantic summaries in ChromaDB."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the processor.
        
        Args:
            embedding_model: Name of the sentence transformer model for embeddings
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.supabase_handler = None
        self.chromadb_handler = None
        self.collection_name = "float_embeddings"
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize database handlers and embedding model."""
        try:
            # Initialize Supabase handler
            logger.info("Initializing Supabase connection...")
            self.supabase_handler = SupabaseHandler()
            
            # Initialize ChromaDB handler
            logger.info("Initializing ChromaDB connection...")
            self.chromadb_handler = ChromaDBHandler()
            
            # Create or get the float_embeddings collection
            try:
                self.chromadb_handler.collection = self.chromadb_handler.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.chromadb_handler.collection = self.chromadb_handler.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Semantic summaries of Argo float data from Supabase"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def fetch_floats_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch float data from Supabase floats table.
        
        Args:
            limit: Optional limit on number of rows to fetch
            
        Returns:
            List of float records with properties
        """
        try:
            from sqlalchemy import text
            
            query = text("""
            SELECT float_id, platform_number, deploy_date, properties
            FROM floats
            ORDER BY deploy_date DESC
            """)
            
            if limit:
                query = text(f"""
                SELECT float_id, platform_number, deploy_date, properties
                FROM floats
                ORDER BY deploy_date DESC
                LIMIT {limit}
                """)
            
            with self.supabase_handler.engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()
                
                # Convert to list of dicts
                floats_data = []
                for row in rows:
                    properties = row[3] if isinstance(row[3], dict) else json.loads(row[3]) if row[3] else {}
                    
                    float_record = {
                        'float_id': row[0],
                        'platform_number': row[1],
                        'deploy_date': row[2],
                        'properties': properties,
                        'total_profiles': properties.get('total_profiles', 0)  # Extract from properties
                    }
                    floats_data.append(float_record)
                
                logger.info(f"Fetched {len(floats_data)} float records from Supabase")
                return floats_data
                
        except Exception as e:
            logger.error(f"Error fetching floats data: {e}")
            return []
    
    def generate_semantic_summary(self, float_record: Dict[str, Any]) -> str:
        """
        Generate a semantic natural language summary from float record.
        
        Args:
            float_record: Float record with properties
            
        Returns:
            Natural language summary text
        """
        try:
            float_id = float_record.get('float_id', 'Unknown')
            platform_number = float_record.get('platform_number', 'Unknown')
            deploy_date = float_record.get('deploy_date')
            properties = float_record.get('properties', {})
            total_profiles = float_record.get('total_profiles', 0)
            
            # Format deploy date
            date_str = "Unknown date"
            if deploy_date:
                if isinstance(deploy_date, str):
                    try:
                        deploy_date = datetime.fromisoformat(deploy_date.replace('Z', '+00:00'))
                    except:
                        pass
                if isinstance(deploy_date, datetime):
                    date_str = deploy_date.strftime("%dth %b %Y")
            
            # Extract location information
            location_range = properties.get('location_range', {})
            lat_min = location_range.get('lat_min')
            lat_max = location_range.get('lat_max')
            lon_min = location_range.get('lon_min')
            lon_max = location_range.get('lon_max')
            
            # Calculate location means
            lat_mean = None
            lon_mean = None
            if lat_min is not None and lat_max is not None:
                lat_mean = (lat_min + lat_max) / 2
            if lon_min is not None and lon_max is not None:
                lon_mean = (lon_min + lon_max) / 2
            
            # Start building summary
            summary_parts = []
            
            # Basic info
            summary_parts.append(f"On {date_str}, Argo float {float_id} (platform {platform_number})")
            
            if total_profiles:
                profile_text = "profile" if total_profiles == 1 else "profiles"
                summary_parts.append(f"recorded {total_profiles} {profile_text}")
            else:
                summary_parts.append("was deployed")
            
            # Location info
            if lat_mean is not None and lon_mean is not None:
                lat_dir = "N" if lat_mean >= 0 else "S"
                lon_dir = "E" if lon_mean >= 0 else "W"
                summary_parts.append(f"near latitude {abs(lat_mean):.3f}°{lat_dir} and longitude {abs(lon_mean):.3f}°{lon_dir}")
            
            # Join the basic info
            summary = " ".join(summary_parts) + "."
            
            # Add measurement details
            measurements = properties.get('measurements', {})
            measurement_details = []
            
            # Pressure/depth info
            if 'pressure' in measurements:
                pres = measurements['pressure']
                if 'min' in pres and 'max' in pres:
                    pres_min = pres.get('min', 0)
                    pres_max = pres.get('max', 0)
                    pres_mean = pres.get('mean', (pres_min + pres_max) / 2)
                    measurement_details.append(
                        f"Pressure ranged from {pres_min:.1f} dbar to {pres_max:.1f} dbar (mean ~{pres_mean:.0f} dbar)"
                    )
            
            # Temperature info
            if 'temperature' in measurements:
                temp = measurements['temperature']
                if 'min' in temp and 'max' in temp:
                    temp_min = temp.get('min', 0)
                    temp_max = temp.get('max', 0)
                    temp_mean = temp.get('mean', (temp_min + temp_max) / 2)
                    measurement_details.append(
                        f"Temperature ranged from {temp_min:.1f}°C to {temp_max:.1f}°C (mean {temp_mean:.1f}°C)"
                    )
            
            # Salinity info
            if 'salinity' in measurements:
                sal = measurements['salinity']
                if 'min' in sal and 'max' in sal:
                    sal_min = sal.get('min', 0)
                    sal_max = sal.get('max', 0)
                    sal_mean = sal.get('mean', (sal_min + sal_max) / 2)
                    measurement_details.append(
                        f"Salinity ranged from {sal_min:.2f} PSU to {sal_max:.2f} PSU (mean {sal_mean:.2f} PSU)"
                    )
            
            # Add measurement count
            total_measurements = 0
            for param, data in measurements.items():
                if isinstance(data, dict) and 'count' in data:
                    total_measurements += data.get('count', 0)
            
            if total_measurements > 0:
                measurement_details.append(f"A total of {total_measurements} measurements were taken")
            
            # Combine everything
            if measurement_details:
                summary += " " + ". ".join(measurement_details) + "."
            
            # Add date range if available
            date_range = properties.get('date_range', {})
            if date_range and 'start' in date_range and 'end' in date_range:
                start_date = date_range['start']
                end_date = date_range['end']
                if start_date != end_date:
                    summary += f" Data collection spanned from {start_date} to {end_date}."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary for float {float_record.get('float_id')}: {e}")
            return f"Argo float {float_record.get('float_id', 'Unknown')} with oceanographic measurements."
    
    def create_chromadb_entry(self, float_record: Dict[str, Any], summary: str) -> Dict[str, Any]:
        """
        Create a ChromaDB entry for a float record.
        
        Args:
            float_record: Float record with properties
            summary: Generated semantic summary
            
        Returns:
            ChromaDB entry dictionary
        """
        try:
            float_id = float_record.get('float_id', 'unknown')
            deploy_date = float_record.get('deploy_date', '')
            properties = float_record.get('properties', {})
            total_profiles = float_record.get('total_profiles', 0)
            
            # Create unique ID
            date_suffix = ""
            if deploy_date:
                try:
                    if isinstance(deploy_date, str):
                        deploy_date_obj = datetime.fromisoformat(deploy_date.replace('Z', '+00:00'))
                    else:
                        deploy_date_obj = deploy_date
                    date_suffix = deploy_date_obj.strftime("%Y%m%d")
                except:
                    date_suffix = "unknown"
            
            entry_id = f"float_{float_id}_{date_suffix}"
            
            # Calculate location means
            location_range = properties.get('location_range', {})
            lat_mean = None
            lon_mean = None
            
            if location_range.get('lat_min') is not None and location_range.get('lat_max') is not None:
                lat_mean = (location_range['lat_min'] + location_range['lat_max']) / 2
            if location_range.get('lon_min') is not None and location_range.get('lon_max') is not None:
                lon_mean = (location_range['lon_min'] + location_range['lon_max']) / 2
            
            # Create metadata
            metadata = {
                'float_id': str(float_id),
                'date': str(deploy_date) if deploy_date else '',
                'profiles': int(total_profiles) if total_profiles else 0
            }
            
            # Add location if available
            if lat_mean is not None:
                metadata['lat'] = float(lat_mean)
            if lon_mean is not None:
                metadata['lon'] = float(lon_mean)
            
            # Generate embedding
            embedding = self.embedding_model.encode(summary).tolist()
            
            return {
                'id': entry_id,
                'document': summary,
                'metadata': metadata,
                'embedding': embedding
            }
            
        except Exception as e:
            logger.error(f"Error creating ChromaDB entry for float {float_record.get('float_id')}: {e}")
            return None
    
    def process_and_store_floats(self, limit: Optional[int] = None, batch_size: int = 50):
        """
        Process floats from Supabase and store them in ChromaDB.
        
        Args:
            limit: Optional limit on number of floats to process
            batch_size: Number of entries to process in each batch
        """
        try:
            # Fetch float data
            logger.info("Fetching float data from Supabase...")
            floats_data = self.fetch_floats_data(limit=limit)
            
            if not floats_data:
                logger.warning("No float data found")
                return
            
            # Process in batches
            total_processed = 0
            total_successful = 0
            
            for i in range(0, len(floats_data), batch_size):
                batch = floats_data[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(floats_data) + batch_size - 1)//batch_size}")
                
                # Prepare batch data for ChromaDB
                batch_ids = []
                batch_documents = []
                batch_metadatas = []
                batch_embeddings = []
                
                for float_record in batch:
                    try:
                        # Generate semantic summary
                        summary = self.generate_semantic_summary(float_record)
                        
                        # Create ChromaDB entry
                        entry = self.create_chromadb_entry(float_record, summary)
                        
                        if entry:
                            batch_ids.append(entry['id'])
                            batch_documents.append(entry['document'])
                            batch_metadatas.append(entry['metadata'])
                            batch_embeddings.append(entry['embedding'])
                            
                            logger.debug(f"Prepared entry for float {float_record.get('float_id')}")
                        
                    except Exception as e:
                        logger.error(f"Error processing float {float_record.get('float_id')}: {e}")
                        continue
                
                # Store batch in ChromaDB
                if batch_ids:
                    try:
                        self.chromadb_handler.collection.add(
                            ids=batch_ids,
                            documents=batch_documents,
                            metadatas=batch_metadatas,
                            embeddings=batch_embeddings
                        )
                        
                        batch_success = len(batch_ids)
                        total_successful += batch_success
                        logger.info(f"Successfully stored {batch_success} entries in ChromaDB")
                        
                    except Exception as e:
                        logger.error(f"Error storing batch in ChromaDB: {e}")
                
                total_processed += len(batch)
                logger.info(f"Progress: {total_processed}/{len(floats_data)} floats processed")
            
            # Final summary
            logger.info(f"Processing complete: {total_successful}/{len(floats_data)} floats successfully stored in ChromaDB")
            
            # Get collection info
            collection_info = self.chromadb_handler.get_collection_info()
            logger.info(f"ChromaDB collection '{self.collection_name}' now contains {collection_info.get('count', 0)} entries")
            
        except Exception as e:
            logger.error(f"Error in process_and_store_floats: {e}")
            raise
    
    def close(self):
        """Close database connections."""
        if self.supabase_handler:
            self.supabase_handler.close()
        if self.chromadb_handler:
            self.chromadb_handler.close()


def main():
    """Main function to run the preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Supabase float data to ChromaDB semantic entries")
    parser.add_argument('--limit', type=int, help='Limit number of floats to process')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2', help='Embedding model to use')
    
    args = parser.parse_args()
    
    processor = None
    try:
        # Initialize processor
        logger.info("Starting float data preprocessing to ChromaDB...")
        processor = FloatsToChromaDBProcessor(embedding_model=args.embedding_model)
        
        # Process and store floats
        processor.process_and_store_floats(
            limit=args.limit, 
            batch_size=args.batch_size
        )
        
        logger.info("Float data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
    finally:
        if processor:
            processor.close()


if __name__ == "__main__":
    main()