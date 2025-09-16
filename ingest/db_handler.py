"""
Database handler for FloatChat Argo data pipeline.
Manages connections to Supabase PostgreSQL and handles data operations.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupabaseHandler:
    """Handles all database operations for the Argo data pipeline."""
    
    def __init__(self, 
                 db_user: str = None,
                 db_password: str = None, 
                 db_host: str = None,
                 db_port: str = None,
                 db_name: str = None):
        """
        Initialize Supabase connection using direct PostgreSQL credentials.
        
        Args:
            db_user: Database username
            db_password: Database password
            db_host: Database host
            db_port: Database port
            db_name: Database name
        """
        self.db_user = db_user or os.getenv("SUPABASE_DB_USER", "postgres")
        self.db_password = db_password or os.getenv("SUPABASE_DB_PASSWORD")
        self.db_host = db_host or os.getenv("SUPABASE_DB_HOST")
        self.db_port = db_port or os.getenv("SUPABASE_DB_PORT", "5432")
        self.db_name = db_name or os.getenv("SUPABASE_DB_NAME", "postgres")
        
        if not all([self.db_password, self.db_host]):
            raise ValueError("Database password and host must be provided via environment variables or parameters")
        
        self.engine = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Supabase PostgreSQL."""
        try:
            # Clean and validate connection parameters
            db_host = self.db_host.replace('http://', '').replace('https://', '')
            db_port = str(self.db_port).strip()
            
            # Validate port
            if not db_port or not db_port.isdigit():
                db_port = "5432"  # Default PostgreSQL port
                logger.warning(f"Invalid port '{self.db_port}', using default 5432")
            
            # Build PostgreSQL connection string
            db_url = f"postgresql://{self.db_user}:{self.db_password}@{db_host}:{db_port}/{self.db_name}"
            
            logger.info(f"Attempting to connect to: postgresql://{self.db_user}:***@{db_host}:{db_port}/{self.db_name}")
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                test_value = result.fetchone()[0]
                if test_value != 1:
                    raise Exception("Connection test failed")
                conn.commit()
            
            logger.info("Successfully connected to Supabase PostgreSQL")
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            logger.error(f"Connection details: host={self.db_host}, port={self.db_port}, user={self.db_user}, db={self.db_name}")
            raise
    
    def initialize_schema(self, schema_path: str = None):
        """
        Initialize database schema from schema.sql file.
        
        Args:
            schema_path: Path to schema.sql file
        """
        if schema_path is None:
            # Default to schema.sql in the ingest directory
            current_dir = Path(__file__).parent
            schema_path = current_dir / "schema.sql"
        
        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema using SQLAlchemy
            with self.engine.connect() as conn:
                # Split by statements and execute each one
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                
                for statement in statements:
                    if statement:
                        conn.execute(text(statement))
                        conn.commit()
            
            logger.info(f"Schema initialized successfully from {schema_path}")
            
        except FileNotFoundError:
            logger.error(f"Schema file not found: {schema_path}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Error initializing schema: {e}")
            raise
    
    def insert_float_data(self, float_data: Dict[str, Any]) -> bool:
        """
        Insert float metadata into the floats table.
        
        Args:
            float_data: Dictionary containing float information
            
        Returns:
            bool: Success status
        """
        try:
            with self.engine.connect() as conn:
                # Convert properties dict to JSON string if needed
                properties = float_data.get('properties', {})
                if isinstance(properties, dict):
                    import json
                    properties_json = json.dumps(properties)
                else:
                    properties_json = properties
                
                # Insert query
                query = text("""
                    INSERT INTO floats (float_id, platform_number, deploy_date, properties)
                    VALUES (:float_id, :platform_number, :deploy_date, :properties)
                    ON CONFLICT (float_id) DO UPDATE SET
                        platform_number = EXCLUDED.platform_number,
                        deploy_date = EXCLUDED.deploy_date,
                        properties = EXCLUDED.properties
                """)
                
                conn.execute(query, {
                    'float_id': float_data.get('float_id'),
                    'platform_number': float_data.get('platform_number'),
                    'deploy_date': float_data.get('deploy_date'),
                    'properties': properties_json
                })
                conn.commit()
                
                logger.info(f"Successfully inserted float: {float_data.get('float_id')}")
                return True
                
        except Exception as e:
            logger.error(f"Error inserting float data: {e}")
            return False
    
    def insert_profile_data(self, profile_data: List[Dict[str, Any]]) -> bool:
        """
        Insert profile data into the profiles table.
        
        Args:
            profile_data: List of profile dictionaries
            
        Returns:
            bool: Success status
        """
        try:
            if not profile_data:
                logger.warning("No profile data to insert")
                return True
            
            with self.engine.connect() as conn:
                # Prepare insert query
                query = text("""
                    INSERT INTO profiles 
                    (profile_id, float_id, profile_time, lat, lon, pressure, depth, 
                     variable_name, variable_value, level, raw_profile)
                    VALUES 
                    (:profile_id, :float_id, :profile_time, :lat, :lon, :pressure, :depth,
                     :variable_name, :variable_value, :level, :raw_profile)
                    ON CONFLICT (profile_id) DO NOTHING
                """)
                
                # Execute batch insert
                conn.execute(query, profile_data)
                conn.commit()
                
                logger.info(f"Successfully inserted {len(profile_data)} profiles")
                return True
                
        except Exception as e:
            logger.error(f"Error inserting profile data: {e}")
            return False
    
    def bulk_insert_profiles(self, df: pd.DataFrame) -> bool:
        """
        Bulk insert profile data using pandas to_sql for better performance.
        
        Args:
            df: DataFrame containing profile data
            
        Returns:
            bool: Success status
        """
        try:
            # Use pandas to_sql for efficient bulk insert
            df.to_sql(
                'profiles', 
                self.engine, 
                if_exists='append', 
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"Successfully bulk inserted {len(df)} profiles")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Error bulk inserting profiles: {e}")
            return False
    
    def insert_embedding_data(self, embedding_data: Dict[str, Any]) -> bool:
        """
        Insert embedding data into the float_embeddings table.
        
        Args:
            embedding_data: Dictionary containing embedding information
            
        Returns:
            bool: Success status
        """
        try:
            with self.engine.connect() as conn:
                # Convert embedding list to PostgreSQL array format
                embedding = embedding_data.get('embedding', [])
                if isinstance(embedding, list):
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                else:
                    embedding_str = str(embedding)
                
                query = text("""
                    INSERT INTO float_embeddings (float_id, metadata_summary, embedding, created_at)
                    VALUES (:float_id, :metadata_summary, :embedding::vector, :created_at)
                    ON CONFLICT DO NOTHING
                """)
                
                conn.execute(query, {
                    'float_id': embedding_data.get('float_id'),
                    'metadata_summary': embedding_data.get('metadata_summary'),
                    'embedding': embedding_str,
                    'created_at': embedding_data.get('created_at')
                })
                conn.commit()
                
                logger.info(f"Successfully inserted embedding for float: {embedding_data.get('float_id')}")
                return True
                
        except Exception as e:
            logger.error(f"Error inserting embedding data: {e}")
            return False
    
    def check_float_exists(self, float_id: str) -> bool:
        """
        Check if a float already exists in the database.
        
        Args:
            float_id: Float identifier
            
        Returns:
            bool: True if float exists
        """
        try:
            with self.engine.connect() as conn:
                query = text("SELECT 1 FROM floats WHERE float_id = :float_id LIMIT 1")
                result = conn.execute(query, {'float_id': float_id})
                return result.fetchone() is not None
                
        except Exception as e:
            logger.error(f"Error checking float existence: {e}")
            return False
    
    def get_float_profile_count(self, float_id: str) -> int:
        """
        Get the number of profiles for a given float.
        
        Args:
            float_id: Float identifier
            
        Returns:
            int: Number of profiles
        """
        try:
            with self.engine.connect() as conn:
                query = text("SELECT COUNT(*) FROM profiles WHERE float_id = :float_id")
                result = conn.execute(query, {'float_id': float_id})
                count = result.fetchone()[0]
                return count if count else 0
                
        except Exception as e:
            logger.error(f"Error getting profile count: {e}")
            return 0
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")


def create_db_handler() -> SupabaseHandler:
    """
    Factory function to create a database handler with environment variables.
    
    Returns:
        SupabaseHandler: Configured database handler
    """
    return SupabaseHandler()


if __name__ == "__main__":
    # Test the database handler
    try:
        db = create_db_handler()
        db.initialize_schema()
        print("Database handler test successful!")
        db.close()
        
    except Exception as e:
        print(f"Database handler test failed: {e}")