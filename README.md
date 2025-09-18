# FloatChat PreProcessing Pipeline

A comprehensive data processing pipeline for oceanographic data from ARGO floats, featuring a FastAPI wrapper, ChromaDB vector storage, and automated data ingestion from NetCDF files.

## Overview

This project processes oceanographic data from ARGO floats (autonomous robotic platforms that measure ocean temperature, salinity, and pressure), stores it in both PostgreSQL (Supabase) and ChromaDB vector database, and provides a REST API for data access and semantic search capabilities.

### Key Features

- **Data Processing**: Automated ingestion of NetCDF files from ARGO float data
- **Vector Database**: ChromaDB integration with semantic embeddings for oceanographic data
- **Dual Storage**: PostgreSQL (Supabase) for structured data, ChromaDB for vector search
- **REST API**: FastAPI wrapper with file upload and data retrieval endpoints
- **Measurements Extraction**: Automatic extraction of temperature, pressure, and salinity statistics
- **Comprehensive Testing**: Advanced query testing framework with 74 challenging test cases
- **Data Management**: Utilities for cleanup, export, and database maintenance

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NetCDF Files  │───▶│  FastAPI Server  │───▶│   ChromaDB      │
│  (ARGO Data)    │    │   (Port 8080)    │    │ (localhost:8000)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Supabase DB    │
                       │   (PostgreSQL)   │
                       └──────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL database (Supabase)
- ChromaDB server

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FloatChat-PreProcessing-Pipeline
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration**:
   Create a `.env` file with:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   CHROMADB_HOST=localhost
   CHROMADB_PORT=8000
   ```

5. **Start ChromaDB server**:
   ```bash
   chroma run --host localhost --port 8000
   ```

## Quick Start

### 1. Start the FastAPI Server

```bash
python fastapi_app.py
```

The API will be available at `http://localhost:8080`

### 2. Upload NetCDF Data

```bash
curl -X POST "http://localhost:8080/upload-file/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_file.nc"
```

### 3. Access API Documentation

Visit `http://localhost:8080/docs` for interactive API documentation.

## API Endpoints

### File Upload
- **POST** `/upload-file/`
  - Upload NetCDF files for processing
  - Extracts measurements (temperature, pressure, salinity)
  - Stores data in both Supabase and ChromaDB
  - Returns processing summary with statistics

### Data Retrieval
- **GET** `/floats/`
  - Retrieve all float data from Supabase
  - Returns structured data with metadata

### Health Check
- **GET** `/`
  - API health check endpoint
  - Returns service status

## Core Components

### 1. Data Processing Pipeline (`main.py`)

Main orchestration script that handles:
- NetCDF file processing
- Data extraction and transformation
- Database storage coordination
- Error handling and logging

```python
# Example usage
python main.py
```

### 2. Database Handler (`ingest/db_handler.py`)

Manages database connections with:
- Connection pooling and retry logic
- Supabase PostgreSQL integration
- ChromaDB client management
- Robust error recovery

### 3. FastAPI Application (`fastapi_app.py`)

REST API features:
- File upload with validation
- Automatic measurement extraction
- Dual database storage
- Enhanced error handling
- Comprehensive response formatting

### 4. Measurement Extraction (`update_chromadb_measurements.py`)

Processes measurement data:
- Temperature, pressure, salinity statistics
- JSON parsing and validation
- ChromaDB metadata enhancement
- Progress tracking

## Testing Framework

### Advanced ChromaDB Queries (`test_advanced_chromadb_queries.py`)

Comprehensive testing with 74 challenging queries across 8 categories:

1. **Geographic Queries** (9 tests)
   - Regional data searches
   - Coordinate-based filtering
   - Ocean basin analysis

2. **Temporal Queries** (10 tests)
   - Date range filtering
   - Seasonal analysis
   - Multi-year comparisons

3. **Measurement Range Queries** (8 tests)
   - Temperature thresholds
   - Pressure ranges
   - Salinity boundaries

4. **Statistical Queries** (9 tests)
   - Extremes identification
   - Variance analysis
   - Distribution patterns

5. **Deep Water Queries** (9 tests)
   - Depth-specific analysis
   - Vertical profiling
   - Deep ocean conditions

6. **Environmental Condition Queries** (9 tests)
   - Multi-parameter filtering
   - Environmental correlations
   - Condition-based searches

7. **Data Quality Queries** (10 tests)
   - Quality flag analysis
   - Data validation
   - Error detection

8. **Complex Multi-Dimensional Queries** (10 tests)
   - Cross-parameter analysis
   - Advanced filtering
   - Multi-constraint searches

### Running Tests

```bash
python test_advanced_chromadb_queries.py
```

## Utility Scripts

### Database Management

**Cleanup ChromaDB** (`cleanup_chromadb.py`):
```bash
python cleanup_chromadb.py
```

**Export ChromaDB to Text** (`export_chromadb_to_text.py`):
```bash
python export_chromadb_to_text.py
```

**Update Measurements** (`update_chromadb_measurements.py`):
```bash
python update_chromadb_measurements.py
```

### Data Processing

**Sync Supabase to ChromaDB** (`sync_supabase_to_chromadb.py`):
```bash
python sync_supabase_to_chromadb.py
```

**Schema Validation** (`check_floats_schema.py`):
```bash
python check_floats_schema.py
```

### Testing Scripts

**Connection Testing** (`test_connection.py`):
```bash
python test_connection.py
```

**Environment Testing** (`test_env.py`):
```bash
python test_env.py
```

**FastAPI Testing** (`test_fastapi.py`):
```bash
python test_fastapi.py
```

#### Process Sample Data (Recommended for Testing)
```powershell
python main.py --mode sample --sample small_test
```

#### Download and Process Data from ERDDAP
```powershell
python main.py --mode download --start-date 2023-01-01 --end-date 2023-01-31 --region indian_ocean --source ifremer
```

#### Process Single Local NetCDF File
```powershell
python main.py --mode file --file path/to/your/file.nc
```

## Pipeline Components

### Data Loader (`ingest/load_data.py`)
- Downloads Argo NetCDF datasets from ERDDAP servers
- Loads and validates NetCDF files using xarray
- Extracts essential oceanographic fields

### Preprocessor (`ingest/preprocess.py`)
- Cleans and validates oceanographic data
- Converts to database-ready format
- Handles missing values and outliers

### Database Handler (`ingest/db_handler.py`)
- Manages Supabase PostgreSQL connections
- Implements the schema defined in `schema.sql`
- Handles bulk data insertions

### Embeddings Generator (`embeddings/embed.py`)
- Generates natural language summaries of float metadata
- Creates vector embeddings using sentence-transformers
- Stores embeddings for semantic search

### Main Orchestrator (`main.py`)
- Coordinates the entire pipeline
- Provides CLI interface with multiple modes
- Handles error logging and recovery

## Database Schema

The pipeline uses the existing schema defined in `ingest/schema.sql`:

- **floats**: Float metadata and deployment information
- **profiles**: Individual profile measurements (temperature, salinity)
- **float_embeddings**: Vector embeddings for semantic search

## Usage Examples

### Basic Sample Processing
```powershell
# Process a small test dataset
python main.py --mode sample

# Process with custom cache directory
python main.py --mode sample --cache-dir "D:\argo_cache"

# Disable embeddings generation
python main.py --mode sample --no-embeddings
```

### ERDDAP Data Download
```powershell
# Download Indian Ocean data for specific time range
python main.py --mode download --start-date 2023-06-01 --end-date 2023-06-30 --region indian_ocean --source ifremer

# Download from different source
python main.py --mode download --source ncei --region indian_ocean
```

### Local File Processing
```powershell
# Process a local NetCDF file
python main.py --mode file --file "data/argo_sample.nc"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

2. **Database Connection Errors**: Verify your Supabase credentials in the `.env` file

3. **NetCDF Download Failures**: Check internet connection and ERDDAP server availability

4. **Memory Issues**: Use smaller time ranges or batch sizes for large datasets

### Logs
Pipeline logs are written to `pipeline.log` and also displayed in the console.

## Data Sources

The pipeline supports downloading from multiple ERDDAP servers:
- **IFREMER**: https://www.ifremer.fr/erddap
- **NCEI**: https://data.nodc.noaa.gov/erddap
- **INCOIS**: https://incois.gov.in/erddap
## Data Schema

### Supabase (PostgreSQL)

**floats table**:
```sql
CREATE TABLE floats (
    float_id SERIAL PRIMARY KEY,
    platform_number VARCHAR(20),
    deploy_date DATE,
    properties JSONB
);
```

### ChromaDB

**Metadata structure**:
```json
{
    "float_id": "string",
    "platform_number": "string", 
    "deploy_date": "string",
    "latitude": "float",
    "longitude": "float",
    "measurements": {
        "temperature": {
            "mean": "float",
            "std": "float", 
            "min": "float",
            "max": "float"
        },
        "pressure": {
            "mean": "float",
            "std": "float",
            "min": "float", 
            "max": "float"
        },
        "salinity": {
            "mean": "float",
            "std": "float",
            "min": "float",
            "max": "float"
        }
    }
}
```

## Configuration

### Environment Variables

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase API key
- `CHROMADB_HOST`: ChromaDB server host (default: localhost)
- `CHROMADB_PORT`: ChromaDB server port (default: 8000)

### ChromaDB Configuration

The system uses ChromaDB with:
- Collection name: `floats_collection`
- Embedding model: `all-MiniLM-L6-v2`
- Distance function: Cosine similarity
- Metadata indexing: Enabled for all fields

## Performance Considerations

### Database Optimization

- **Connection Pooling**: Configured for optimal Supabase connections
- **Retry Logic**: Exponential backoff for failed connections
- **Batch Processing**: Efficient bulk data operations
- **Index Strategy**: Optimized metadata indexing in ChromaDB

### Vector Search Performance

- **Embedding Caching**: Reuse embeddings where possible
- **Query Optimization**: Structured queries for better performance
- **Collection Management**: Proper collection organization
- **Memory Management**: Efficient resource utilization

## Troubleshooting

### Common Issues

1. **ChromaDB Connection Failed**:
   ```bash
   # Start ChromaDB server
   chroma run --host localhost --port 8000
   ```

2. **Supabase Connection Issues**:
   - Verify credentials in `.env` file
   - Check network connectivity
   - Validate API key permissions

3. **File Upload Errors**:
   - Ensure NetCDF file format
   - Check file size limits
   - Verify file path accessibility

4. **Vector Search Issues**:
   - Confirm ChromaDB collection exists
   - Validate embedding model availability
   - Check query syntax

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development

### Project Structure

```
FloatChat-PreProcessing-Pipeline/
├── fastapi_app.py              # Main FastAPI application
├── main.py                     # Core processing pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── ingest/                     # Data ingestion modules
│   ├── db_handler.py          # Database connection management
│   ├── ingest.py              # Data ingestion logic
│   ├── load_data.py           # Data loading utilities
│   ├── preprocess.py          # Data preprocessing
│   └── schema.sql             # Database schema
├── test_*.py                   # Testing scripts
├── cleanup_*.py               # Cleanup utilities
├── sync_*.py                  # Synchronization scripts
├── update_*.py                # Update utilities
├── export_*.py                # Export utilities
├── view_*.py                  # Data viewing utilities
├── 2019/                      # Sample NetCDF data (2019)
├── argo_data/                 # ARGO data directory
├── argo_data_2020_01/         # Monthly ARGO data
├── data/                      # Processed data
├── docker/                    # Docker configuration
└── embeddings/                # Embedding storage
```

### Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Use type hints where applicable
5. Maintain backward compatibility

### Code Quality

- **Type Checking**: Use mypy for static type checking
- **Code Formatting**: Use black for code formatting
- **Linting**: Use flake8 for code linting
- **Testing**: Maintain >90% test coverage

## Performance Notes

- Use bulk operations for large datasets
- Enable caching to avoid re-downloading files
- Consider disabling embeddings for faster processing during development
- Monitor memory usage with large NetCDF files

## Next Steps

After successful data ingestion:
1. Verify data in your Supabase dashboard
2. Test semantic search with embeddings
3. Proceed to Stage 2: Chatbot and Dashboard development

## License

[Add your license information here]

## Support

For questions or issues:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section

## Acknowledgments

- ARGO float data providers
- ChromaDB for vector database capabilities
- FastAPI for the web framework
- Supabase for database hosting