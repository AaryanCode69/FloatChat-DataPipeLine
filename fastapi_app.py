"""
FastAPI wrapper for FloatChat Data Processing Pipeline
=====================================================

This FastAPI application provides REST endpoints for uploading and processing
Argo float data files (.nc, .ndrf) through the existing data processing pipeline.

Features:
- File upload endpoint for NetCDF and NDRF files
- Integration with existing data processing pipeline
- ChromaDB embedding with sentence transformers
- PostgreSQL storage via Supabase
- Async support and comprehensive error handling

Author: FloatChat Data Pipeline
"""

import os
import logging
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import traceback

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import aiofiles

# Import existing pipeline components
import sys
sys.path.append(str(Path(__file__).parent))

from ingest.db_handler import SupabaseHandler, ChromaDBHandler
from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API responses
class FileInfo(BaseModel):
    """Model for file information."""
    filename: str
    size: int
    processed_at: str

class DateRange(BaseModel):
    """Model for date range."""
    start: str
    end: str

class LocationRange(BaseModel):
    """Model for location range."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

class MeasurementData(BaseModel):
    """Model for measurement data."""
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    count: Optional[int] = None
    units: Optional[str] = None

class DataExtracted(BaseModel):
    """Model for extracted data."""
    float_id: str
    total_profiles: int
    date_range: DateRange
    measurements: Dict[str, MeasurementData]
    location_range: LocationRange

class ChromaDBStorageInfo(BaseModel):
    """Model for ChromaDB storage information."""
    status: str
    document_id: Optional[str] = None
    collection: Optional[str] = None
    has_measurements: bool = False
    measurement_types: List[str] = []
    embedding_dimension: Optional[int] = None

class SupabaseStorageInfo(BaseModel):
    """Model for Supabase storage information."""
    status: str
    float_id: Optional[str] = None
    table: Optional[str] = None
    platform_number: Optional[str] = None
    deploy_date: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

class StorageResults(BaseModel):
    """Model for storage results."""
    supabase: SupabaseStorageInfo
    chromadb: ChromaDBStorageInfo

class FileProcessingResponse(BaseModel):
    """Model for complete file processing response."""
    success: bool
    message: str
    file_info: FileInfo
    data_extracted: DataExtracted
    storage: StorageResults
    processing_time: float

class ProcessingStatus(BaseModel):
    """Model for processing status response."""
    task_id: str
    status: str
    message: str
    timestamp: datetime

class ProcessingResult(BaseModel):
    """Model for processing result response."""
    task_id: str
    status: str
    file_info: Dict[str, Any]
    extracted_data: Optional[Dict[str, Any]] = None
    storage_results: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None
    processing_time_seconds: Optional[float] = None

class HealthCheck(BaseModel):
    """Model for health check response."""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str = "1.0.0"

# Global storage for processing tasks
processing_tasks: Dict[str, Dict[str, Any]] = {}

# FastAPI app configuration
app = FastAPI(
    title="FloatChat Data Processing API",
    description="REST API for processing Argo float data files and storing in ChromaDB/PostgreSQL",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

class DataProcessingService:
    """Service class for handling data processing operations."""
    
    def __init__(self):
        """Initialize the data processing service."""
        self.supabase_handler: Optional[SupabaseHandler] = None
        self.chromadb_handler: Optional[ChromaDBHandler] = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections."""
        try:
            # Initialize Supabase connection
            self.supabase_handler = SupabaseHandler()
            logger.info("✓ Supabase connection initialized")
            
            # Initialize ChromaDB connection
            self.chromadb_handler = ChromaDBHandler()
            logger.info("✓ ChromaDB connection initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connections: {e}")
            raise
    
    async def process_file(self, file_path: Path, task_id: str) -> Dict[str, Any]:
        """
        Process an uploaded file through the data pipeline.
        
        Args:
            file_path: Path to the uploaded file
            task_id: Unique task identifier
            
        Returns:
            Dictionary containing processing results
        """
        start_time = datetime.now()
        
        try:
            # Update task status
            processing_tasks[task_id]["status"] = "processing"
            processing_tasks[task_id]["message"] = "Processing file through data pipeline"
            
            # Import and use existing pipeline components
            from main import ArgoDataPipeline
            
            # Create pipeline instance
            pipeline = ArgoDataPipeline()
            
            # Process the file
            logger.info(f"Processing file: {file_path}")
            
            # Extract data using existing pipeline
            extracted_data = await self._extract_data_from_file(file_path)
            
            if not extracted_data:
                raise ValueError("No data extracted from file")
            
            # Store in ChromaDB with embeddings
            chromadb_result = await self._store_in_chromadb(extracted_data, task_id)
            
            # Store in PostgreSQL via Supabase
            supabase_result = await self._store_in_supabase(extracted_data, task_id)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare results
            results = {
                "task_id": task_id,
                "status": "completed",
                "file_info": {
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "file_type": file_path.suffix
                },
                "extracted_data": extracted_data,
                "storage_results": {
                    "chromadb": chromadb_result,
                    "supabase": supabase_result
                },
                "processing_time_seconds": processing_time
            }
            
            # Update task status
            processing_tasks[task_id].update(results)
            processing_tasks[task_id]["message"] = "Processing completed successfully"
            
            logger.info(f"✓ File processing completed for task {task_id}")
            return results
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"❌ {error_msg} for task {task_id}")
            logger.error(traceback.format_exc())
            
            # Update task with error
            processing_tasks[task_id].update({
                "status": "failed",
                "error_details": error_msg,
                "message": error_msg
            })
            
            raise HTTPException(status_code=500, detail=error_msg)
    
    async def _extract_data_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract structured data from NetCDF/NDRF file."""
        try:
            import xarray as xr
            import numpy as np
            
            # Open the NetCDF file
            with xr.open_dataset(file_path) as ds:
                # Extract basic information
                data = {
                    "date_range": {},
                    "measurements": {},
                    "location_range": {},
                    "total_profiles": 0
                }
                
                # Extract time range
                if 'time' in ds.variables:
                    time_data = ds['time']
                    if len(time_data) > 0:
                        data["date_range"] = {
                            "start": str(time_data.min().values),
                            "end": str(time_data.max().values)
                        }
                
                # Extract location range
                if 'latitude' in ds.variables and 'longitude' in ds.variables:
                    lat_data = ds['latitude']
                    lon_data = ds['longitude']
                    
                    # Handle different latitude/longitude variable names
                    lat_values = lat_data.values if hasattr(lat_data, 'values') else [lat_data]
                    lon_values = lon_data.values if hasattr(lon_data, 'values') else [lon_data]
                    
                    # Remove NaN values
                    lat_values = lat_values[~np.isnan(lat_values)] if len(lat_values) > 0 else []
                    lon_values = lon_values[~np.isnan(lon_values)] if len(lon_values) > 0 else []
                    
                    if len(lat_values) > 0 and len(lon_values) > 0:
                        data["location_range"] = {
                            "lat_min": float(np.min(lat_values)),
                            "lat_max": float(np.max(lat_values)),
                            "lon_min": float(np.min(lon_values)),
                            "lon_max": float(np.max(lon_values))
                        }
                
                # Extract measurements (temperature, salinity, pressure)
                measurements = {}
                
                # Common variable names for each measurement type
                var_mappings = {
                    "temperature": ["temp", "temperature", "TEMP", "TEMPERATURE"],
                    "salinity": ["sal", "salinity", "PSAL", "SALINITY"],
                    "pressure": ["pres", "pressure", "PRES", "PRESSURE"]
                }
                
                for measurement_type, possible_names in var_mappings.items():
                    for var_name in possible_names:
                        if var_name in ds.variables:
                            var_data = ds[var_name]
                            values = var_data.values if hasattr(var_data, 'values') else [var_data]
                            
                            # Remove NaN values and convert to float
                            if len(values) > 0:
                                clean_values = []
                                for val in np.array(values).flatten():
                                    if not np.isnan(val) and np.isfinite(val):
                                        clean_values.append(float(val))
                                
                                if clean_values:
                                    measurements[measurement_type] = {
                                        "min": min(clean_values),
                                        "max": max(clean_values),
                                        "mean": sum(clean_values) / len(clean_values),
                                        "count": len(clean_values)
                                    }
                            break
                
                data["measurements"] = measurements
                
                # Calculate total profiles (could be based on time dimension or profile dimension)
                if 'profile' in ds.dims:
                    data["total_profiles"] = int(ds.dims['profile'])
                elif 'time' in ds.dims:
                    data["total_profiles"] = int(ds.dims['time'])
                else:
                    data["total_profiles"] = 1
                
                logger.info(f"✓ Extracted data from {file_path.name}: {len(measurements)} measurement types")
                return data
                
        except Exception as e:
            logger.error(f"❌ Failed to extract data from {file_path}: {e}")
            raise ValueError(f"Data extraction failed: {e}")
    
    async def _store_in_chromadb(self, data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Store data in ChromaDB with sentence transformer embeddings and measurements metadata."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Initialize sentence transformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create a natural language description of the data
            description = self._create_natural_language_description(data)
            
            # Generate embedding
            embedding = model.encode(description).tolist()
            
            # Use the main float_embeddings collection instead of separate API collection
            collection_name = "float_embeddings"
            try:
                collection = self.chromadb_handler.client.get_collection(collection_name)
            except:
                collection = self.chromadb_handler.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Float embeddings with measurements"}
                )
            
            # Extract float ID from data or use task-based ID
            float_id = f"api_upload_{task_id}"
            if 'float_id' in data and data['float_id'] != 'unknown':
                float_id = data['float_id']
            
            # Build metadata with measurements in the same format as update_chromadb_measurements.py
            metadata = {
                "task_id": task_id,
                "source": "fastapi_upload", 
                "timestamp": datetime.now().isoformat(),
                "data_type": "argo_float_data",
                "float_id": float_id,
                "has_measurements": False,
                "measurements_updated": True
            }
            
            # Add measurement metadata if available
            if data.get("measurements"):
                measurements = data["measurements"]
                metadata["has_measurements"] = True
                
                # Add flattened measurement metadata for each parameter
                for param in ['temperature', 'pressure', 'salinity']:
                    if param in measurements and isinstance(measurements[param], dict):
                        param_data = measurements[param]
                        metadata[f"{param}_min"] = param_data.get('min')
                        metadata[f"{param}_max"] = param_data.get('max') 
                        metadata[f"{param}_mean"] = param_data.get('mean')
                        metadata[f"{param}_count"] = param_data.get('count')
            
            # Add location and date metadata if available
            if data.get("location_range"):
                loc = data["location_range"]
                metadata.update({
                    "lat_min": loc.get('lat_min'),
                    "lat_max": loc.get('lat_max'),
                    "lon_min": loc.get('lon_min'),
                    "lon_max": loc.get('lon_max')
                })
            
            if data.get("date_range"):
                date_range = data["date_range"]
                metadata.update({
                    "date_start": date_range.get('start'),
                    "date_end": date_range.get('end')
                })
            
            if data.get("total_profiles"):
                metadata["total_profiles"] = data["total_profiles"]
            
            # Store in ChromaDB
            doc_id = f"api_upload_{task_id}"
            collection.add(
                ids=[doc_id],
                documents=[description],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            logger.info(f"✓ Stored data in ChromaDB with ID: {doc_id} and measurements metadata")
            
            return {
                "status": "success",
                "document_id": doc_id,
                "collection": collection_name,
                "embedding_dimension": len(embedding),
                "description_length": len(description),
                "has_measurements": metadata.get("has_measurements", False),
                "measurement_types": [param for param in ['temperature', 'pressure', 'salinity'] 
                                    if f"{param}_min" in metadata]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to store in ChromaDB: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _store_in_supabase(self, data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Store data in Supabase PostgreSQL with retry logic."""
        try:
            # Extract float_id from the extracted data or use task_id
            float_id = f"api_upload_{task_id}"
            
            # Try to get a more meaningful platform number from data
            platform_number = None
            if 'float_id' in data and data['float_id'] != 'unknown':
                platform_number = data['float_id']
            
            # Get deploy date from extracted data if available
            deploy_date = None
            if 'date_range' in data and 'start' in data['date_range']:
                deploy_date = data['date_range']['start']
            
            # Prepare data for insertion using the correct schema
            float_data = {
                "float_id": float_id,
                "platform_number": platform_number,
                "deploy_date": deploy_date,
                "properties": data  # Pass the dict directly, let the handler convert to JSON
            }
            
            # Insert into floats table using the existing method with retry
            success = self.supabase_handler.insert_float_data(float_data, retry_count=3)
            
            if success:
                logger.info(f"✓ Stored data in Supabase floats table")
                return {
                    "status": "success",
                    "float_id": float_id,
                    "table": "floats",
                    "platform_number": platform_number,
                    "deploy_date": deploy_date
                }
            else:
                raise Exception("Insert operation returned False after retries")
            
        except Exception as e:
            logger.error(f"❌ Failed to store in Supabase: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _create_natural_language_description(self, data: Dict[str, Any]) -> str:
        """Create a natural language description of the float data."""
        description_parts = []
        
        # Add date information
        if data.get("date_range"):
            date_range = data["date_range"]
            description_parts.append(f"Data collected from {date_range.get('start', 'unknown')} to {date_range.get('end', 'unknown')}")
        
        # Add location information
        if data.get("location_range"):
            loc = data["location_range"]
            description_parts.append(
                f"Location: latitude {loc.get('lat_min', 'unknown')} to {loc.get('lat_max', 'unknown')}, "
                f"longitude {loc.get('lon_min', 'unknown')} to {loc.get('lon_max', 'unknown')}"
            )
        
        # Add measurement information
        if data.get("measurements"):
            measurements = data["measurements"]
            for measurement_type, stats in measurements.items():
                if isinstance(stats, dict) and "min" in stats:
                    description_parts.append(
                        f"{measurement_type.capitalize()}: {stats['min']:.2f} to {stats['max']:.2f} "
                        f"(mean: {stats['mean']:.2f}, {stats['count']} measurements)"
                    )
        
        # Add profile information
        if data.get("total_profiles"):
            description_parts.append(f"Total profiles: {data['total_profiles']}")
        
        return ". ".join(description_parts)

# Initialize the data processing service
data_service = DataProcessingService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Starting FloatChat Data Processing API")
    logger.info("✓ Services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down FloatChat Data Processing API")

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FloatChat Data Processing API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/upload",
            "status": "/status/{task_id}",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    services = {}
    
    # Check Supabase connection
    try:
        with data_service.supabase_handler.engine.connect() as conn:
            conn.execute("SELECT 1")
        services["supabase"] = "healthy"
    except Exception:
        services["supabase"] = "unhealthy"
    
    # Check ChromaDB connection
    try:
        data_service.chromadb_handler.client.heartbeat()
        services["chromadb"] = "healthy"
    except Exception:
        services["chromadb"] = "unhealthy"
    
    overall_status = "healthy" if all(status == "healthy" for status in services.values()) else "unhealthy"
    
    return HealthCheck(
        status=overall_status,
        timestamp=datetime.now(),
        services=services
    )

@app.post("/upload", response_model=FileProcessingResponse)
async def upload_file(
    file: UploadFile = File(...)
):
    """
    Upload and process NetCDF (.nc) or NDRF files with immediate processing.
    
    This endpoint accepts file uploads, processes them synchronously through the data pipeline,
    extracts structured JSON data, stores it in both ChromaDB and PostgreSQL, and returns
    complete processing results including file info, extracted data, and storage status.
    """
    start_time = datetime.now()
    
    # Validate file type
    if not file.filename.lower().endswith(('.nc', '.ndrf')):
        raise HTTPException(
            status_code=400,
            detail="Only .nc and .ndrf files are supported"
        )
    
    try:
        # Save uploaded file to temporary location
        temp_dir = Path(tempfile.gettempdir()) / "floatchat_uploads"
        temp_dir.mkdir(exist_ok=True)
        
        task_id = str(uuid.uuid4())
        file_path = temp_dir / f"{task_id}_{file.filename}"
        
        # Write file asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Get file info
        file_stats = file_path.stat()
        file_info = FileInfo(
            filename=file.filename,
            size=file_stats.st_size,
            processed_at=datetime.now().isoformat()
        )
        
        logger.info(f"✓ File uploaded: {file.filename}, starting processing...")
        
        # Extract data from file
        extracted_data_dict = await data_service._extract_data_from_file(file_path)
        
        # Extract float ID from filename (common pattern in Argo files)
        float_id = "unknown"
        if "D" in file.filename:
            # Extract float ID from filename like "nodc_D1900975_339.nc"
            parts = file.filename.split("_")
            for part in parts:
                if part.startswith("D") and len(part) > 1:
                    float_id = part[1:]  # Remove the 'D' prefix
                    break
        elif "R" in file.filename:
            # Extract float ID from filename like "nodc_R7900647_003.nc"
            parts = file.filename.split("_")
            for part in parts:
                if part.startswith("R") and len(part) > 1:
                    float_id = part[1:]  # Remove the 'R' prefix
                    break
        
        # Create structured data_extracted response
        data_extracted = DataExtracted(
            float_id=float_id,
            total_profiles=extracted_data_dict.get("total_profiles", 1),
            date_range=DateRange(
                start=extracted_data_dict.get("date_range", {}).get("start", "unknown"),
                end=extracted_data_dict.get("date_range", {}).get("end", "unknown")
            ),
            measurements={
                key: MeasurementData(**value) 
                for key, value in extracted_data_dict.get("measurements", {}).items()
            },
            location_range=LocationRange(
                **extracted_data_dict.get("location_range", {
                    "lat_min": 0.0, "lat_max": 0.0, 
                    "lon_min": 0.0, "lon_max": 0.0
                })
            )
        )
        
        logger.info(f"✓ Data extracted, storing in databases...")
        
        # Store in databases
        try:
            chromadb_result = await data_service._store_in_chromadb(extracted_data_dict, task_id)
            chromadb_storage = ChromaDBStorageInfo(
                status="embedded",
                document_id=chromadb_result.get("document_id"),
                collection=chromadb_result.get("collection"),
                has_measurements=chromadb_result.get("has_measurements", False),
                measurement_types=chromadb_result.get("measurement_types", []),
                embedding_dimension=chromadb_result.get("embedding_dimension")
            )
            logger.info(f"✓ Data stored in ChromaDB with measurements: {chromadb_result.get('measurement_types', [])}")
        except Exception as e:
            logger.error(f"❌ ChromaDB storage failed: {e}")
            chromadb_storage = ChromaDBStorageInfo(status="failed")
        
        try:
            supabase_result = await data_service._store_in_supabase(extracted_data_dict, task_id)
            if supabase_result.get("status") == "success":
                supabase_storage = SupabaseStorageInfo(
                    status="stored",
                    float_id=supabase_result.get("float_id"),
                    table=supabase_result.get("table"),
                    platform_number=supabase_result.get("platform_number"),
                    deploy_date=supabase_result.get("deploy_date")
                )
                logger.info(f"✓ Data stored in Supabase")
            else:
                supabase_storage = SupabaseStorageInfo(
                    status="failed",
                    error=supabase_result.get("error"),
                    error_type=supabase_result.get("error_type")
                )
        except Exception as e:
            logger.error(f"❌ Supabase storage failed: {e}")
            supabase_storage = SupabaseStorageInfo(
                status="failed", 
                error=str(e),
                error_type=type(e).__name__
            )
        
        storage = StorageResults(
            supabase=supabase_storage,
            chromadb=chromadb_storage
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Cleanup temp file
        try:
            file_path.unlink()
            logger.info(f"✓ Cleaned up temporary file")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup temp file: {e}")
        
        logger.info(f"✅ File processing completed successfully in {processing_time:.2f}s")
        
        # Return complete response
        return FileProcessingResponse(
            success=True,
            message="File processed successfully",
            file_info=file_info,
            data_extracted=data_extracted,
            storage=storage,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        logger.error(traceback.format_exc())
        
        # Cleanup temp file in case of error
        try:
            if 'file_path' in locals():
                file_path.unlink()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/status/{task_id}", response_model=ProcessingResult)
async def get_processing_status(task_id: str):
    """Get the status of a processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = processing_tasks[task_id]
    
    return ProcessingResult(**task_data)

@app.get("/tasks", response_model=List[ProcessingStatus])
async def list_tasks():
    """List all processing tasks."""
    return [
        ProcessingStatus(
            task_id=task_data["task_id"],
            status=task_data["status"],
            message=task_data["message"],
            timestamp=task_data["timestamp"]
        )
        for task_data in processing_tasks.values()
    ]

async def cleanup_temp_file(file_path: Path):
    """Clean up temporary uploaded file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"✓ Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"⚠️  Failed to cleanup temp file {file_path}: {e}")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )