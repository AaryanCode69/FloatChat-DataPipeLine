"""
Data loader for Argo NetCDF datasets.
Handles downloading and loading NetCDF files from various sources.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import requests
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArgoDataLoader:
    """Handles loading and processing of Argo NetCDF datasets."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the Argo data loader.
        
        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "argo_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # ERDDAP endpoints for different regions
        self.erddap_endpoints = {
            "ifremer": "https://www.ifremer.fr/erddap",
            "ncei": "https://data.nodc.noaa.gov/erddap", 
            "incois": "https://incois.gov.in/erddap"
        }
        
        # Indian Ocean region bounds (for PoC)
        self.indian_ocean_bounds = {
            "lat_min": -60.0,
            "lat_max": 30.0, 
            "lon_min": 20.0,
            "lon_max": 120.0
        }
    
    def download_argo_data(self, 
                          dataset_id: str = "ArgoFloats",
                          source: str = "ifremer",
                          time_range: Tuple[str, str] = None,
                          region: str = "indian_ocean") -> Optional[str]:
        """
        Download Argo data from ERDDAP server.
        
        Args:
            dataset_id: ERDDAP dataset identifier
            source: Data source (ifremer, ncei, incois)
            time_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
            region: Region filter ('indian_ocean' or custom bounds)
            
        Returns:
            str: Path to downloaded NetCDF file, None if failed
        """
        try:
            base_url = self.erddap_endpoints.get(source)
            if not base_url:
                logger.error(f"Unknown data source: {source}")
                return None
            
            # Build ERDDAP query URL
            query_params = self._build_erddap_query(dataset_id, time_range, region)
            download_url = f"{base_url}/tabledap/{dataset_id}.nc?{query_params}"
            
            # Generate cache filename
            cache_filename = f"{dataset_id}_{source}_{region}_{datetime.now().strftime('%Y%m%d')}.nc"
            cache_path = self.cache_dir / cache_filename
            
            # Check if file already exists in cache
            if cache_path.exists():
                logger.info(f"Using cached file: {cache_path}")
                return str(cache_path)
            
            # Download file
            logger.info(f"Downloading Argo data from {source}...")
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded: {cache_path}")
            return str(cache_path)
            
        except requests.RequestException as e:
            logger.error(f"Failed to download Argo data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading data: {e}")
            return None
    
    def _build_erddap_query(self, 
                           dataset_id: str,
                           time_range: Tuple[str, str] = None,
                           region: str = "indian_ocean") -> str:
        """
        Build ERDDAP query parameters.
        
        Args:
            dataset_id: Dataset identifier
            time_range: Time range tuple
            region: Region filter
            
        Returns:
            str: Query parameter string
        """
        # Essential variables for Argo data
        variables = [
            "platform_number",
            "cycle_number", 
            "time",
            "latitude",
            "longitude",
            "pres",  # pressure
            "temp",  # temperature
            "psal"   # practical salinity
        ]
        
        # Build query
        query_parts = []
        
        # Select variables
        query_parts.append(",".join(variables))
        
        # Apply region filter
        if region == "indian_ocean":
            bounds = self.indian_ocean_bounds
            query_parts.extend([
                f"&latitude>={bounds['lat_min']}",
                f"&latitude<={bounds['lat_max']}",
                f"&longitude>={bounds['lon_min']}",
                f"&longitude<={bounds['lon_max']}"
            ])
        
        # Apply time filter
        if time_range:
            start_date, end_date = time_range
            query_parts.extend([
                f"&time>={start_date}T00:00:00Z",
                f"&time<={end_date}T23:59:59Z"
            ])
        
        return "".join(query_parts)
    
    def inspect_netcdf_file(self, file_path: str) -> Dict[str, Any]:
        """
        Inspect a NetCDF file to understand its structure.
        
        Args:
            file_path: Path to NetCDF file
            
        Returns:
            Dict: File inspection results
        """
        try:
            logger.info(f"Inspecting NetCDF file: {file_path}")
            
            with xr.open_dataset(file_path) as ds:
                inspection = {
                    'file_path': file_path,
                    'dimensions': dict(ds.dims),
                    'variables': list(ds.variables.keys()),
                    'data_vars': list(ds.data_vars.keys()),
                    'coords': list(ds.coords.keys()),
                    'attributes': dict(ds.attrs),
                    'variable_details': {}
                }
                
                # Get details for each variable
                for var_name in ds.variables:
                    var = ds[var_name]
                    inspection['variable_details'][var_name] = {
                        'shape': var.shape,
                        'dtype': str(var.dtype),
                        'dims': var.dims,
                        'attributes': dict(var.attrs) if hasattr(var, 'attrs') else {}
                    }
                
                logger.info(f"File dimensions: {inspection['dimensions']}")
                logger.info(f"Available variables: {inspection['variables']}")
                
                return inspection
                
        except Exception as e:
            logger.error(f"Error inspecting NetCDF file: {e}")
            return {'error': str(e)}
    
    def load_netcdf_file(self, file_path: str) -> Optional[xr.Dataset]:
        """
        Load NetCDF file using xarray.
        
        Args:
            file_path: Path to NetCDF file
            
        Returns:
            xr.Dataset: Loaded dataset, None if failed
        """
        try:
            logger.info(f"Loading NetCDF file: {file_path}")
            
            # Load with xarray
            ds = xr.open_dataset(file_path)
            
            logger.info(f"Dataset loaded successfully. Shape: {ds.dims}")
            logger.info(f"Variables: {list(ds.data_vars.keys())}")
            
            return ds
            
        except Exception as e:
            logger.error(f"Failed to load NetCDF file: {e}")
            return None
    
    def extract_argo_fields(self, dataset: xr.Dataset) -> Dict[str, np.ndarray]:
        """
        Extract essential fields from Argo dataset.
        
        Args:
            dataset: xarray Dataset
            
        Returns:
            Dict: Extracted fields as numpy arrays
        """
        try:
            extracted_data = {}
            
            # Map of standard field names to possible NetCDF variable names
            field_mapping = {
                "float_id": ["platform_number", "PLATFORM_NUMBER", "FLOAT_SERIAL_NO", "WMO_INST_TYPE"],
                "profile_id": ["cycle_number", "CYCLE_NUMBER", "PROFILE_NUMBER"],
                "time": ["time", "TIME", "JULD", "REFERENCE_DATE_TIME", "DATE_TIME"],
                "latitude": ["latitude", "LATITUDE", "lat", "LAT", "POSITION_LATITUDE"],
                "longitude": ["longitude", "LONGITUDE", "lon", "LON", "POSITION_LONGITUDE"],
                "pressure": ["pres", "PRES", "pressure", "PRESSURE", "PRES_ADJUSTED"],
                "depth": ["depth", "DEPTH", "DEPH"],
                "temperature": ["temp", "TEMP", "temperature", "TEMPERATURE", "TEMP_ADJUSTED"],
                "salinity": ["psal", "PSAL", "salinity", "SALINITY", "PSAL_ADJUSTED"]
            }
            
            # Log available variables for debugging
            logger.info(f"Available variables in dataset: {list(dataset.variables.keys())}")
            
            # Extract each field
            for field_name, possible_vars in field_mapping.items():
                extracted_data[field_name] = self._extract_variable(dataset, possible_vars)
                
                # If we didn't find the variable, try with case variations
                if extracted_data[field_name] is None:
                    case_variations = []
                    for var in possible_vars:
                        case_variations.extend([var.lower(), var.upper(), var.title()])
                    extracted_data[field_name] = self._extract_variable(dataset, case_variations)
            
            # Special handling for time variables
            if extracted_data["time"] is not None:
                extracted_data["time"] = self._process_time_variable(dataset, extracted_data["time"])
            
            # Special handling for float_id - ensure it's string format
            if extracted_data["float_id"] is not None:
                float_ids = extracted_data["float_id"]
                if float_ids.dtype.kind in ['U', 'S']:  # Unicode or byte string
                    # Already string, just ensure proper format
                    extracted_data["float_id"] = np.array([str(fid).strip() for fid in float_ids.flat]).reshape(float_ids.shape)
                else:
                    # Convert numbers to strings with safe NaN checking
                    def safe_convert_to_string(value):
                        try:
                            # For numeric types, check if it's NaN or a valid number
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                if np.isnan(value):
                                    return 'unknown'
                                else:
                                    return str(int(value))
                            else:
                                # For other types (strings, objects), convert directly
                                return str(value).strip()
                        except (ValueError, TypeError, OverflowError):
                            return 'unknown'
                    
                    extracted_data["float_id"] = np.array([safe_convert_to_string(fid) for fid in float_ids.flat]).reshape(float_ids.shape)
            
            # Log extraction summary
            for field, data in extracted_data.items():
                if data is not None:
                    logger.info(f"Extracted {field}: shape {data.shape}, dtype {data.dtype}")
                else:
                    logger.warning(f"Could not extract {field}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error extracting Argo fields: {e}")
            return {}
    
    def _extract_variable(self, dataset: xr.Dataset, possible_names: List[str]) -> Optional[np.ndarray]:
        """
        Extract a variable from dataset using possible variable names.
        
        Args:
            dataset: xarray Dataset
            possible_names: List of possible variable names
            
        Returns:
            np.ndarray: Variable data, None if not found
        """
        for var_name in possible_names:
            if var_name in dataset.variables:
                data = dataset[var_name].values
                # Handle different data types and missing values
                if hasattr(dataset[var_name], '_FillValue'):
                    fill_value = dataset[var_name]._FillValue
                    data = np.where(data == fill_value, np.nan, data)
                elif hasattr(dataset[var_name], 'missing_value'):
                    missing_value = dataset[var_name].missing_value
                    data = np.where(data == missing_value, np.nan, data)
                
                return data
        
        return None
    
    def _process_time_variable(self, dataset: xr.Dataset, time_data: np.ndarray) -> np.ndarray:
        """
        Process time variable to ensure proper datetime format.
        
        Args:
            dataset: xarray Dataset
            time_data: Raw time data
            
        Returns:
            np.ndarray: Processed time data
        """
        try:
            # Find the time variable in the dataset to get attributes
            time_var = None
            time_var_names = ["time", "TIME", "JULD", "REFERENCE_DATE_TIME", "DATE_TIME"]
            
            for var_name in time_var_names:
                if var_name in dataset.variables:
                    time_var = dataset[var_name]
                    break
            
            if time_var is None:
                return time_data
            
            # Handle different time formats
            if hasattr(time_var, 'units'):
                units = time_var.units.lower()
                
                # Common Argo time formats
                if 'days since' in units or 'hours since' in units:
                    # Use xarray's time conversion
                    time_decoded = time_var.values
                    if hasattr(time_decoded, 'astype'):
                        # Convert to datetime64 if not already
                        return time_decoded.astype('datetime64[ns]')
                    return time_decoded
                    
            # If no units or standard format, try to parse as is
            return time_data
            
        except Exception as e:
            logger.warning(f"Could not process time variable: {e}")
            return time_data
    
    def get_sample_dataset_urls(self) -> Dict[str, str]:
        """
        Get URLs for sample Argo datasets for testing.
        
        Returns:
            Dict: Sample dataset URLs
        """
        return {
            "ifremer_sample": "https://www.ifremer.fr/erddap/tabledap/ArgoFloats.nc?platform_number,cycle_number,time,latitude,longitude,pres,temp,psal&latitude>=0&latitude<=10&longitude>=70&longitude<=80&time>=2023-01-01T00:00:00Z&time<=2023-01-31T23:59:59Z",
            "small_test": "https://data.nodc.noaa.gov/erddap/tabledap/ArgoFloats-synthetic.nc?platform_number,time,latitude,longitude,temp&latitude>=-10&latitude<=10&longitude>=70&longitude<=90&time>=2023-06-01T00:00:00Z&time<=2023-06-02T23:59:59Z"
        }
    
    def download_sample_data(self, sample_name: str = "small_test") -> Optional[str]:
        """
        Download a small sample dataset for testing.
        
        Args:
            sample_name: Name of sample dataset
            
        Returns:
            str: Path to downloaded file
        """
        sample_urls = self.get_sample_dataset_urls()
        
        if sample_name not in sample_urls:
            logger.error(f"Unknown sample dataset: {sample_name}")
            return None
        
        try:
            url = sample_urls[sample_name]
            cache_filename = f"sample_{sample_name}.nc"
            cache_path = self.cache_dir / cache_filename
            
            if cache_path.exists():
                logger.info(f"Using cached sample: {cache_path}")
                return str(cache_path)
            
            logger.info(f"Downloading sample dataset: {sample_name}")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Sample dataset downloaded: {cache_path}")
            return str(cache_path)
            
        except Exception as e:
            logger.error(f"Failed to download sample data: {e}")
            return None


def create_data_loader(cache_dir: str = None) -> ArgoDataLoader:
    """
    Factory function to create an Argo data loader.
    
    Args:
        cache_dir: Directory for caching files
        
    Returns:
        ArgoDataLoader: Configured data loader
    """
    return ArgoDataLoader(cache_dir)


if __name__ == "__main__":
    # Test the data loader
    try:
        loader = create_data_loader()
        
        # Try downloading a sample dataset
        sample_file = loader.download_sample_data("small_test")
        
        if sample_file:
            # Load and examine the dataset
            dataset = loader.load_netcdf_file(sample_file)
            
            if dataset:
                # Extract fields
                fields = loader.extract_argo_fields(dataset)
                print(f"Successfully extracted {len(fields)} fields")
                
                # Close dataset
                dataset.close()
        
        print("Data loader test completed!")
        
    except Exception as e:
        print(f"Data loader test failed: {e}")