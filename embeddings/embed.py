"""
Embeddings generation module for Argo float metadata.
Creates vector embeddings for semantic search and retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArgoEmbeddingsGenerator:
    """Generates embeddings for Argo float metadata."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings generator.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_float_metadata_summary(self, float_data: Dict[str, Any]) -> str:
        """
        Generate a natural language summary of float metadata.
        
        Args:
            float_data: Float metadata dictionary
            
        Returns:
            str: Natural language summary
        """
        try:
            float_id = float_data.get('float_id', 'Unknown')
            properties = float_data.get('properties', {})
            
            # Extract key information
            total_profiles = properties.get('total_profiles', 0)
            date_range = properties.get('date_range', {})
            location_range = properties.get('location_range', {})
            measurements = properties.get('measurements', {})
            
            # Build summary components
            summary_parts = []
            
            # Basic info
            summary_parts.append(f"Argo float {float_id}")
            
            # Time range
            if date_range:
                start_date = date_range.get('start', '').split('T')[0]
                end_date = date_range.get('end', '').split('T')[0]
                summary_parts.append(f"operated from {start_date} to {end_date}")
            
            # Location
            if location_range:
                lat_center = (location_range.get('lat_min', 0) + location_range.get('lat_max', 0)) / 2
                lon_center = (location_range.get('lon_min', 0) + location_range.get('lon_max', 0)) / 2
                
                # Determine ocean region
                region = self._determine_ocean_region(lat_center, lon_center)
                summary_parts.append(f"in the {region}")
                
                # Add coordinate details
                summary_parts.append(f"(latitude {location_range.get('lat_min', 0):.1f} to {location_range.get('lat_max', 0):.1f}, longitude {location_range.get('lon_min', 0):.1f} to {location_range.get('lon_max', 0):.1f})")
            
            # Profiles
            summary_parts.append(f"with {total_profiles} profiles")
            
            # Measurements
            measurement_summary = []
            if 'temperature' in measurements:
                temp_stats = measurements['temperature']
                measurement_summary.append(f"temperature ranging from {temp_stats.get('min', 0):.1f}°C to {temp_stats.get('max', 0):.1f}°C")
            
            if 'salinity' in measurements:
                sal_stats = measurements['salinity']
                measurement_summary.append(f"salinity from {sal_stats.get('min', 0):.1f} to {sal_stats.get('max', 0):.1f} PSU")
            
            if 'depth' in measurements or 'pressure' in measurements:
                if 'depth' in measurements:
                    depth_stats = measurements['depth']
                    measurement_summary.append(f"depths up to {depth_stats.get('max', 0):.0f} meters")
                elif 'pressure' in measurements:
                    pres_stats = measurements['pressure']
                    measurement_summary.append(f"pressures up to {pres_stats.get('max', 0):.0f} dbar")
            
            if measurement_summary:
                summary_parts.append(f"measuring {', '.join(measurement_summary)}")
            
            # Join all parts
            summary = ' '.join(summary_parts) + '.'
            
            # Add oceanographic context
            context_parts = []
            if lat_center < -30:
                context_parts.append("Southern Ocean region")
            elif lat_center > 30:
                context_parts.append("Northern hemisphere waters")
            else:
                context_parts.append("tropical and subtropical waters")
            
            # Add seasonal context if date range is available
            if date_range:
                try:
                    start_month = int(date_range.get('start', '2023-01-01').split('-')[1])
                    if 3 <= start_month <= 5:
                        context_parts.append("spring deployment")
                    elif 6 <= start_month <= 8:
                        context_parts.append("summer deployment")
                    elif 9 <= start_month <= 11:
                        context_parts.append("autumn deployment")
                    else:
                        context_parts.append("winter deployment")
                except:
                    pass
            
            if context_parts:
                summary += f" This float operated in {', '.join(context_parts)}."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating metadata summary: {e}")
            return f"Argo float {float_data.get('float_id', 'Unknown')} with oceanographic measurements."
    
    def _determine_ocean_region(self, lat: float, lon: float) -> str:
        """
        Determine ocean region based on coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            str: Ocean region name
        """
        # Simplified ocean region classification
        if 20 <= lon <= 120 and -60 <= lat <= 30:
            return "Indian Ocean"
        elif -180 <= lon <= -60:
            return "Pacific Ocean"
        elif -60 <= lon <= 20:
            return "Atlantic Ocean"
        elif lon >= 120:
            if lat >= 0:
                return "North Pacific Ocean"
            else:
                return "South Pacific Ocean"
        else:
            return "Global Ocean"
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            np.ndarray: Array of embeddings
        """
        try:
            if not texts:
                return np.empty((0, self.embedding_dim))
            
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.empty((0, self.embedding_dim))
    
    def process_float_embeddings(self, floats_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process floats DataFrame to generate embeddings.
        
        Args:
            floats_df: DataFrame containing float metadata
            
        Returns:
            List[Dict]: List of embedding records
        """
        try:
            if floats_df.empty:
                logger.warning("No float data to process for embeddings")
                return []
            
            # Generate metadata summaries
            summaries = []
            for _, row in floats_df.iterrows():
                summary = self.generate_float_metadata_summary(row.to_dict())
                summaries.append(summary)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(summaries)
            
            # Create embedding records
            embedding_records = []
            for i, (_, row) in enumerate(floats_df.iterrows()):
                if i < len(embeddings):
                    record = {
                        'float_id': row['float_id'],
                        'metadata_summary': summaries[i],
                        'embedding': embeddings[i].tolist(),  # Convert to list for JSON storage
                        'created_at': datetime.utcnow()
                    }
                    embedding_records.append(record)
            
            logger.info(f"Generated {len(embedding_records)} embedding records")
            return embedding_records
            
        except Exception as e:
            logger.error(f"Error processing float embeddings: {e}")
            return []
    
    def process_and_store_chromadb_embeddings(self, floats_df: pd.DataFrame, chromadb_handler) -> bool:
        """
        Process floats DataFrame and store embeddings directly in ChromaDB.
        
        Args:
            floats_df: DataFrame containing float metadata
            chromadb_handler: ChromaDB handler instance
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if floats_df.empty:
                logger.warning("No float data to process for ChromaDB embeddings")
                return False
            
            # Generate metadata summaries
            summaries = []
            metadatas = []
            ids = []
            
            for _, row in floats_df.iterrows():
                float_dict = row.to_dict()
                summary = self.generate_float_metadata_summary(float_dict)
                summaries.append(summary)
                
                # Create metadata for ChromaDB
                metadata = {
                    'float_id': str(float_dict.get('float_id', 'unknown')),
                    'latitude': float(float_dict.get('latitude', 0.0)) if pd.notna(float_dict.get('latitude')) else 0.0,
                    'longitude': float(float_dict.get('longitude', 0.0)) if pd.notna(float_dict.get('longitude')) else 0.0,
                    'timestamp': str(float_dict.get('time', '')),
                    'cycle_number': int(float_dict.get('cycle_number', 0)) if pd.notna(float_dict.get('cycle_number')) else 0,
                    'created_at': datetime.utcnow().isoformat()
                }
                metadatas.append(metadata)
                
                # Create unique ID for ChromaDB
                float_id = str(float_dict.get('float_id', 'unknown'))
                cycle = str(float_dict.get('cycle_number', 0))
                unique_id = f"{float_id}_{cycle}_{len(ids)}"
                ids.append(unique_id)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(summaries)
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            # Store in ChromaDB
            chromadb_handler.add_embeddings(
                embeddings=embeddings_list,
                documents=summaries,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(embeddings_list)} embeddings in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error processing and storing ChromaDB embeddings: {e}")
            return False
    
    def generate_profile_summaries(self, profiles_df: pd.DataFrame) -> List[str]:
        """
        Generate summaries for profile data (for future use).
        
        Args:
            profiles_df: DataFrame containing profile data
            
        Returns:
            List[str]: List of profile summaries
        """
        try:
            if profiles_df.empty:
                return []
            
            # Group profiles by float and time
            profile_groups = profiles_df.groupby(['float_id', 'profile_time'])
            
            summaries = []
            for (float_id, profile_time), group in profile_groups:
                
                # Create summary for this profile
                date_str = profile_time.strftime('%Y-%m-%d')
                lat = group['lat'].iloc[0]
                lon = group['lon'].iloc[0]
                
                # Get measurements summary
                temp_data = group[group['variable_name'] == 'TEMP']
                sal_data = group[group['variable_name'] == 'PSAL']
                
                summary_parts = [f"Argo float {float_id} profile from {date_str}"]
                summary_parts.append(f"at location {lat:.2f}°N, {lon:.2f}°E")
                
                if not temp_data.empty:
                    temp_range = f"{temp_data['variable_value'].min():.1f}°C to {temp_data['variable_value'].max():.1f}°C"
                    summary_parts.append(f"with temperature measurements {temp_range}")
                
                if not sal_data.empty:
                    sal_range = f"{sal_data['variable_value'].min():.1f} to {sal_data['variable_value'].max():.1f} PSU"
                    summary_parts.append(f"and salinity measurements {sal_range}")
                
                max_depth = group['depth'].max() if 'depth' in group.columns and not group['depth'].isna().all() else None
                max_pressure = group['pressure'].max() if 'pressure' in group.columns and not group['pressure'].isna().all() else None
                
                if max_depth:
                    summary_parts.append(f"down to {max_depth:.0f} meters depth")
                elif max_pressure:
                    summary_parts.append(f"down to {max_pressure:.0f} dbar pressure")
                
                summary = ' '.join(summary_parts) + '.'
                summaries.append(summary)
            
            logger.info(f"Generated {len(summaries)} profile summaries")
            return summaries
            
        except Exception as e:
            logger.error(f"Error generating profile summaries: {e}")
            return []
    
    def search_similar_floats(self, 
                             query_text: str, 
                             embeddings_data: List[Dict[str, Any]], 
                             top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for similar floats based on text query.
        
        Args:
            query_text: Search query
            embeddings_data: List of embedding records
            top_k: Number of top results to return
            
        Returns:
            List[Tuple]: List of (float_id, similarity_score, summary) tuples
        """
        try:
            if not embeddings_data:
                return []
            
            # Generate query embedding
            query_embedding = self.generate_embeddings([query_text])[0]
            
            # Calculate similarities
            similarities = []
            for record in embeddings_data:
                float_embedding = np.array(record['embedding'])
                similarity = np.dot(query_embedding, float_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(float_embedding)
                )
                similarities.append((
                    record['float_id'],
                    float(similarity),
                    record['metadata_summary']
                ))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar floats: {e}")
            return []


def create_embeddings_generator(model_name: str = "all-MiniLM-L6-v2") -> ArgoEmbeddingsGenerator:
    """
    Factory function to create an embeddings generator.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        ArgoEmbeddingsGenerator: Configured embeddings generator
    """
    return ArgoEmbeddingsGenerator(model_name)


if __name__ == "__main__":
    # Test the embeddings generator
    try:
        generator = create_embeddings_generator()
        
        # Create sample float data
        sample_float = {
            'float_id': '1234567',
            'properties': {
                'total_profiles': 150,
                'date_range': {
                    'start': '2023-01-15T12:00:00Z',
                    'end': '2023-06-20T18:30:00Z'
                },
                'location_range': {
                    'lat_min': -10.5,
                    'lat_max': -8.2,
                    'lon_min': 75.1,
                    'lon_max': 77.8
                },
                'measurements': {
                    'temperature': {'min': 22.1, 'max': 29.8, 'mean': 26.5, 'count': 1500},
                    'salinity': {'min': 34.8, 'max': 35.6, 'mean': 35.2, 'count': 1500}
                }
            }
        }
        
        # Generate summary and embedding
        summary = generator.generate_float_metadata_summary(sample_float)
        print(f"Generated summary: {summary}")
        
        embedding = generator.generate_embeddings([summary])
        print(f"Generated embedding with shape: {embedding.shape}")
        
        print("Embeddings generator test successful!")
        
    except Exception as e:
        print(f"Embeddings generator test failed: {e}")