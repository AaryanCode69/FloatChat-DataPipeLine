"""
Main orchestration script for FloatChat Argo data preprocessing pipeline.
Coordinates the entire data pipeline from raw NetCDF to Supabase storage.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
import argparse
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path(__file__).parent / '.env'
try:
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from: {env_file}")
    else:
        print(f"Warning: .env file not found at: {env_file}")
        # Try loading from current directory
        load_dotenv()
except ImportError:
    # Fallback: manually load .env file if dotenv is not available
    print("python-dotenv not available, loading .env manually...")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"Manually loaded environment variables from: {env_file}")
except Exception as e:
    print(f"Error loading environment variables: {e}")

# Debug: Print loaded environment variables (without showing sensitive data)
print("Environment variables loaded:")
for key in ["SUPABASE_DB_USER", "SUPABASE_DB_HOST", "SUPABASE_DB_PORT", "SUPABASE_DB_NAME"]:
    value = os.getenv(key)
    if value:
        print(f"  {key}: {'*' * len(value) if 'PASSWORD' in key else value}")
    else:
        print(f"  {key}: NOT SET")

password_set = os.getenv("SUPABASE_DB_PASSWORD") is not None
print(f"  SUPABASE_DB_PASSWORD: {'SET' if password_set else 'NOT SET'}")

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from ingest.load_data import create_data_loader
from ingest.preprocess import create_preprocessor
from ingest.db_handler import create_db_handler, create_hybrid_db_handler
from embeddings.embed import create_embeddings_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FloatChatPipeline:
    """Main pipeline orchestrator for Argo data processing."""
    
    def __init__(self, 
                 cache_dir: str = None,
                 enable_embeddings: bool = True,
                 batch_size: int = 1000):
        """
        Initialize the pipeline.
        
        Args:
            cache_dir: Directory for caching downloaded files
            enable_embeddings: Whether to generate embeddings
            batch_size: Batch size for database operations
        """
        self.cache_dir = cache_dir
        self.enable_embeddings = enable_embeddings
        self.batch_size = batch_size
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.db_handler = None
        self.embeddings_generator = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            logger.info("Initializing pipeline components...")
            
            # Initialize data loader
            self.data_loader = create_data_loader(self.cache_dir)
            logger.info("Data loader initialized")
            
            # Initialize preprocessor
            self.preprocessor = create_preprocessor()
            logger.info("Preprocessor initialized")
            
            # Initialize hybrid database handler (Supabase + ChromaDB)
            try:
                self.db_handler = create_hybrid_db_handler()
                logger.info("Hybrid database handler initialized (Supabase + ChromaDB)")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid handler: {e}")
                logger.info("Falling back to Supabase-only handler")
                self.db_handler = create_db_handler()
                logger.info("Supabase database handler initialized")
            
            # Initialize embeddings generator if enabled
            if self.enable_embeddings:
                try:
                    self.embeddings_generator = create_embeddings_generator()
                    logger.info("Embeddings generator initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize embeddings generator: {e}")
                    logger.warning("Continuing without embeddings generation")
                    self.enable_embeddings = False
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def setup_database(self):
        """Initialize the database schema."""
        try:
            logger.info("Setting up database schema...")
            
            # Find schema.sql file
            schema_path = Path(__file__).parent / "ingest" / "schema.sql"
            if not schema_path.exists():
                schema_path = Path(__file__).parent.parent / "ingest" / "schema.sql"
            
            if schema_path.exists():
                # Initialize schema (use Supabase for structured data)
                db_schema = self.db_handler.supabase if hasattr(self.db_handler, 'supabase') else self.db_handler
                db_schema.initialize_schema(str(schema_path))
                logger.info("Database schema initialized successfully")
            else:
                logger.error("Could not find schema.sql file")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            return False
    
    def process_netcdf_file(self, file_path: str, skip_existing: bool = True) -> bool:
        """
        Process a single NetCDF file through the entire pipeline.
        
        Args:
            file_path: Path to NetCDF file
            skip_existing: Whether to skip floats that already exist in database
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Processing NetCDF file: {file_path}")
            
            # First inspect the file to understand its structure
            inspection = self.data_loader.inspect_netcdf_file(file_path)
            if 'error' in inspection:
                logger.error(f"Failed to inspect file: {inspection['error']}")
                return False
            
            # Load NetCDF data
            dataset = self.data_loader.load_netcdf_file(file_path)
            if dataset is None:
                logger.error("Failed to load NetCDF file")
                return False
            
            # Extract fields
            raw_data = self.data_loader.extract_argo_fields(dataset)
            dataset.close()  # Close dataset to free memory
            
            if not raw_data:
                logger.error("Failed to extract data from NetCDF file")
                return False
            
            # Preprocess data
            floats_df, profiles_df = self.preprocessor.process_raw_data(raw_data)
            
            if floats_df.empty and profiles_df.empty:
                logger.warning("No valid data found after preprocessing")
                return True
            
            # Validate data
            if not self.preprocessor.validate_dataframes(floats_df, profiles_df):
                logger.error("Data validation failed")
                return False
            
            # Store data in database
            success = self._store_data(floats_df, profiles_df, skip_existing)
            
            if success:
                logger.info(f"Successfully processed file: {file_path}")
            else:
                logger.error(f"Failed to store data from file: {file_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing NetCDF file {file_path}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _store_data(self, floats_df, profiles_df, skip_existing: bool = True) -> bool:
        """
        Store processed data in the database.
        
        Args:
            floats_df: Floats DataFrame
            profiles_df: Profiles DataFrame
            skip_existing: Whether to skip existing floats
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("Storing data in database...")
            
            # Insert float data
            float_success = self._insert_floats(floats_df, skip_existing)
            
            # Insert profile data
            profile_success = self._insert_profiles(profiles_df)
            
            # Generate and insert embeddings if enabled
            embeddings_success = True
            if self.enable_embeddings and not floats_df.empty:
                embeddings_success = self._insert_embeddings(floats_df)
            
            overall_success = float_success and profile_success and embeddings_success
            
            if overall_success:
                logger.info("All data stored successfully")
            else:
                logger.warning("Some data storage operations failed")
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return False
    
    def _insert_floats(self, floats_df, skip_existing: bool = True) -> bool:
        """Insert float metadata into database."""
        try:
            success_count = 0
            
            for _, row in floats_df.iterrows():
                float_id = row['float_id']
                
                # Check if float already exists (use Supabase for structured data)
                db_check = self.db_handler.supabase if hasattr(self.db_handler, 'supabase') else self.db_handler
                if skip_existing and db_check.check_float_exists(float_id):
                    logger.info(f"Skipping existing float: {float_id}")
                    continue
                
                # Insert float data (use Supabase for structured data)
                float_data = row.to_dict()
                # Convert properties to dict if it's not already
                if isinstance(float_data.get('properties'), str):
                    import json
                    float_data['properties'] = json.loads(float_data['properties'])
                
                db_insert = self.db_handler.supabase if hasattr(self.db_handler, 'supabase') else self.db_handler
                if db_insert.insert_float_data(float_data):
                    success_count += 1
                else:
                    logger.warning(f"Failed to insert float: {float_id}")
            
            logger.info(f"Successfully inserted {success_count}/{len(floats_df)} floats")
            return success_count > 0 or len(floats_df) == 0
            
        except Exception as e:
            logger.error(f"Error inserting floats: {e}")
            return False
    
    def _insert_profiles(self, profiles_df) -> bool:
        """Insert profile data into database."""
        try:
            if profiles_df.empty:
                return True
            
            # Use bulk insert for better performance (use Supabase for structured data)
            db_insert = self.db_handler.supabase if hasattr(self.db_handler, 'supabase') else self.db_handler
            success = db_insert.bulk_insert_profiles(profiles_df)
            
            if success:
                logger.info(f"Successfully inserted {len(profiles_df)} profiles")
            else:
                logger.error("Failed to insert profiles")
            
            return success
            
        except Exception as e:
            logger.error(f"Error inserting profiles: {e}")
            return False
    
    def _insert_embeddings(self, floats_df) -> bool:
        """Generate and insert embeddings."""
        try:
            if not self.embeddings_generator:
                return True
            
            # Check if we have ChromaDB available (hybrid handler)
            if hasattr(self.db_handler, 'chromadb'):
                # Use ChromaDB for embeddings storage
                logger.info("Storing embeddings in ChromaDB")
                success = self.embeddings_generator.process_and_store_chromadb_embeddings(
                    floats_df, self.db_handler.chromadb
                )
                if success:
                    logger.info("Successfully stored embeddings in ChromaDB")
                    return True
                else:
                    logger.warning("Failed to store embeddings in ChromaDB, falling back to traditional method")
            
            # Fallback to traditional embedding storage (Supabase)
            logger.info("Using traditional embedding storage method")
            embedding_records = self.embeddings_generator.process_float_embeddings(floats_df)
            
            if not embedding_records:
                logger.warning("No embeddings generated")
                return True
            
            # Insert embeddings using traditional method
            success_count = 0
            if hasattr(self.db_handler, 'supabase'):
                # Hybrid handler
                for record in embedding_records:
                    if self.db_handler.supabase.insert_embedding_data(record):
                        success_count += 1
            else:
                # Traditional handler
                for record in embedding_records:
                    if self.db_handler.insert_embedding_data(record):
                        success_count += 1
            
            logger.info(f"Successfully inserted {success_count}/{len(embedding_records)} embeddings")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error inserting embeddings: {e}")
            return False
    
    def process_data_folder(self, data_dir: str = None, skip_existing: bool = True) -> bool:
        """
        Process all NetCDF files in the data folder.
        
        Args:
            data_dir: Path to data directory (default: ./data)
            skip_existing: Whether to skip floats that already exist in database
            
        Returns:
            bool: Success status
        """
        try:
            # Set default data directory
            if data_dir is None:
                data_dir = Path(__file__).parent / "data"
            else:
                data_dir = Path(data_dir)
            
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                return False
            
            # Find all NetCDF files
            netcdf_patterns = ["*.nc", "*.netcdf", "*.NC", "*.NETCDF"]
            netcdf_files = []
            
            for pattern in netcdf_patterns:
                netcdf_files.extend(data_dir.glob(pattern))
            
            if not netcdf_files:
                logger.warning(f"No NetCDF files found in {data_dir}")
                return True
            
            logger.info(f"Found {len(netcdf_files)} NetCDF files to process")
            
            # Process each file
            successful_files = 0
            failed_files = 0
            
            for file_path in netcdf_files:
                logger.info(f"Processing file {successful_files + failed_files + 1}/{len(netcdf_files)}: {file_path.name}")
                
                try:
                    if self.process_netcdf_file(str(file_path), skip_existing):
                        successful_files += 1
                        logger.info(f"✓ Successfully processed: {file_path.name}")
                    else:
                        failed_files += 1
                        logger.error(f"✗ Failed to process: {file_path.name}")
                        
                except Exception as e:
                    failed_files += 1
                    logger.error(f"✗ Error processing {file_path.name}: {e}")
            
            # Summary
            logger.info(f"Processing complete: {successful_files} successful, {failed_files} failed")
            
            return failed_files == 0
            
        except Exception as e:
            logger.error(f"Error processing data folder: {e}")
            return False
        """
        Download and process a sample dataset for testing.
        
        Args:
            sample_name: Name of sample dataset
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Processing sample dataset: {sample_name}")
            
            # Download sample data
            sample_file = self.data_loader.download_sample_data(sample_name)
            
            if not sample_file:
                logger.error("Failed to download sample data")
                return False
            
            # Process the file
            return self.process_netcdf_file(sample_file)
            
        except Exception as e:
            logger.error(f"Error processing sample data: {e}")
            return False
    
    def download_and_process_argo_data(self, 
                                       time_range: Tuple[str, str] = None,
                                       region: str = "indian_ocean",
                                       source: str = "ifremer") -> bool:
        """
        Download and process Argo data from ERDDAP.
        
        Args:
            time_range: Tuple of (start_date, end_date)
            region: Region filter
            source: Data source
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Downloading Argo data from {source} for {region}")
            
            # Download data
            file_path = self.data_loader.download_argo_data(
                time_range=time_range,
                region=region,
                source=source
            )
            
            if not file_path:
                logger.error("Failed to download Argo data")
                return False
            
            # Process the file
            return self.process_netcdf_file(file_path)
            
        except Exception as e:
            logger.error(f"Error downloading and processing Argo data: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.db_handler:
                self.db_handler.close()
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="FloatChat Argo Data Pipeline")
    
    parser.add_argument(
        "--mode",
        choices=["sample", "download", "file", "data-folder"],
        default="sample",
        help="Processing mode: sample data, download from ERDDAP, process single file, or process all files in data folder"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to NetCDF file (for file mode)"
    )
    
    parser.add_argument(
        "--sample",
        type=str,
        default="small_test",
        help="Sample dataset name (for sample mode)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for data download (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for data download (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--region",
        type=str,
        default="indian_ocean",
        help="Region filter for data download"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="ifremer",
        choices=["ifremer", "ncei", "incois"],
        help="Data source for download"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory for caching files"
    )
    
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embeddings generation"
    )
    
    parser.add_argument(
        "--setup-db",
        action="store_true",
        help="Initialize database schema only"
    )
    
    args = parser.parse_args()
    
    try:
        # Check environment variables
        required_env_vars = ["SUPABASE_DB_PASSWORD", "SUPABASE_DB_HOST"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars and not args.setup_db:
            logger.error(f"Missing required environment variables: {missing_vars}")
            logger.error("Please set SUPABASE_DB_PASSWORD and SUPABASE_DB_HOST in your .env file")
            sys.exit(1)
        
        # Initialize pipeline
        pipeline = FloatChatPipeline(
            cache_dir=args.cache_dir,
            enable_embeddings=not args.no_embeddings
        )
        
        # Setup database if requested
        if args.setup_db:
            if pipeline.setup_database():
                logger.info("Database setup completed successfully")
                sys.exit(0)
            else:
                logger.error("Database setup failed")
                sys.exit(1)
        
        # Ensure database is set up
        if not pipeline.setup_database():
            logger.error("Failed to setup database")
            sys.exit(1)
        
        # Process data based on mode
        success = False
        
        if args.mode == "sample":
            success = pipeline.process_sample_data(args.sample)
            
        elif args.mode == "download":
            time_range = None
            if args.start_date and args.end_date:
                time_range = (args.start_date, args.end_date)
            
            success = pipeline.download_and_process_argo_data(
                time_range=time_range,
                region=args.region,
                source=args.source
            )
            
        elif args.mode == "file":
            if not args.file:
                logger.error("File path required for file mode")
                sys.exit(1)
            
            if not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}")
                sys.exit(1)
            
            success = pipeline.process_netcdf_file(args.file)
            
        elif args.mode == "data-folder":
            # Process all NetCDF files in the data folder
            data_folder = Path(__file__).parent / "data"
            logger.info(f"Processing all NetCDF files in: {data_folder}")
            success = pipeline.process_data_folder(str(data_folder))
        
        # Cleanup
        pipeline.cleanup()
        
        if success:
            logger.info("Pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()