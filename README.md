# FloatChat Data Pipeline Configuration

## Overview
This is the data preprocessing pipeline for FloatChat - an AI-powered conversational interface for Argo ocean data discovery and visualization.

## Setup Instructions

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Environment Variables
1. Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

2. Edit `.env` file with your Supabase database credentials:
   - `SUPABASE_DB_USER`: Database username (typically 'postgres')
   - `SUPABASE_DB_PASSWORD`: Your database password
   - `SUPABASE_DB_HOST`: Database host (e.g., db.yourproject.supabase.co)
   - `SUPABASE_DB_PORT`: Database port (typically 5432)
   - `SUPABASE_DB_NAME`: Database name (typically 'postgres')

### 3. Initialize Database
```powershell
python main.py --setup-db
```

### 4. Run Pipeline

#### Process Local NetCDF Files (Your Use Case)
1. **Add your NetCDF files to the `data` folder:**
```powershell
# Copy your NetCDF files to the data directory
Copy-Item "path\to\your\files\*.nc" "data\"
```

2. **Process all files automatically:**
```powershell
# Using PowerShell script (recommended)
.\process_data.ps1

# Or using batch file
.\process_data.bat

# Or manually using Python
python main.py --mode data-folder
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