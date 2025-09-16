"""
Data preprocessing module for Argo oceanographic data.
Cleans, normalizes, and structures raw data into database-ready format.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArgoDataPreprocessor:
    """Handles preprocessing and cleaning of Argo float data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.temperature_bounds = (-5.0, 40.0)  # Reasonable ocean temperature range (°C)
        self.salinity_bounds = (0.0, 50.0)      # Reasonable salinity range (PSU)
        self.depth_bounds = (0.0, 6000.0)       # Maximum ocean depth (m)
        self.pressure_bounds = (0.0, 6000.0)    # Maximum pressure (dbar)
    
    def process_raw_data(self, raw_data: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process raw extracted Argo data into clean DataFrames.
        
        Args:
            raw_data: Dictionary of extracted NetCDF data arrays
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (floats_df, profiles_df)
        """
        try:
            logger.info("Starting data preprocessing...")
            
            # Convert to DataFrame for easier manipulation
            df = self._create_initial_dataframe(raw_data)
            
            if df.empty:
                logger.warning("No data to process")
                return pd.DataFrame(), pd.DataFrame()
            
            # Clean and validate data
            df = self._clean_data(df)
            
            # Create float metadata and profile data
            floats_df = self._create_floats_dataframe(df)
            profiles_df = self._create_profiles_dataframe(df)
            
            logger.info(f"Preprocessing complete. Floats: {len(floats_df)}, Profiles: {len(profiles_df)}")
            
            return floats_df, profiles_df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _create_initial_dataframe(self, raw_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Create initial DataFrame from raw data arrays.
        
        Args:
            raw_data: Dictionary of numpy arrays
            
        Returns:
            pd.DataFrame: Initial DataFrame
        """
        try:
            # Check if we have minimum required fields
            required_fields = ['float_id', 'time', 'latitude', 'longitude']
            missing_fields = [field for field in required_fields if raw_data.get(field) is None]
            
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                return pd.DataFrame()
            
            # Determine data length
            data_length = len(raw_data['time'])
            
            # Create DataFrame
            df_dict = {}
            
            for field, data in raw_data.items():
                if data is not None:
                    # Handle different array shapes
                    if data.ndim == 1 and len(data) == data_length:
                        df_dict[field] = data
                    elif data.ndim == 2:
                        # For 2D arrays (e.g., profiles with multiple levels)
                        # We'll handle this during profile expansion
                        df_dict[field] = data
                    else:
                        logger.warning(f"Skipping field {field} with incompatible shape: {data.shape}")
            
            # Handle case where we have 2D data (multiple depth levels per profile)
            if any(isinstance(v, np.ndarray) and v.ndim == 2 for v in df_dict.values()):
                df = self._expand_profile_data(df_dict)
            else:
                df = pd.DataFrame(df_dict)
            
            logger.info(f"Initial DataFrame created with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error creating initial DataFrame: {e}")
            return pd.DataFrame()
    
    def _expand_profile_data(self, data_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Expand 2D profile data into individual records for each depth level.
        
        Args:
            data_dict: Dictionary of data arrays
            
        Returns:
            pd.DataFrame: Expanded DataFrame
        """
        try:
            rows = []
            
            # Get 1D arrays (profile-level data)
            profile_data = {}
            level_data = {}
            
            for field, data in data_dict.items():
                if isinstance(data, np.ndarray):
                    if data.ndim == 1:
                        profile_data[field] = data
                    elif data.ndim == 2:
                        level_data[field] = data
            
            # Get number of profiles and levels
            if level_data:
                first_2d_field = next(iter(level_data.values()))
                n_profiles, n_levels = first_2d_field.shape
            else:
                n_profiles = len(next(iter(profile_data.values())))
                n_levels = 1
            
            # Expand data
            for profile_idx in range(n_profiles):
                for level_idx in range(n_levels):
                    row = {}
                    
                    # Add profile-level data
                    for field, data in profile_data.items():
                        row[field] = data[profile_idx]
                    
                    # Add level-specific data
                    for field, data in level_data.items():
                        value = data[profile_idx, level_idx]
                        # Skip NaN values
                        if not np.isnan(value):
                            row[field] = value
                            row['level'] = level_idx
                        else:
                            continue  # Skip this level if key measurement is NaN
                    
                    # Only add row if we have valid level data
                    if 'level' in row:
                        rows.append(row)
            
            df = pd.DataFrame(rows)
            logger.info(f"Expanded {n_profiles} profiles into {len(df)} level records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error expanding profile data: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            initial_count = len(df)
            
            # Convert float_id to string and handle missing values
            if 'float_id' in df.columns:
                df['float_id'] = df['float_id'].astype(str)
                df = df[df['float_id'] != 'nan']
            
            # Clean time data
            if 'time' in df.columns:
                df = self._clean_time_data(df)
            
            # Clean geographic data
            df = self._clean_geographic_data(df)
            
            # Clean oceanographic measurements
            df = self._clean_measurements(df)
            
            # Remove rows with critical missing data
            critical_columns = ['float_id', 'time', 'latitude', 'longitude']
            existing_critical = [col for col in critical_columns if col in df.columns]
            df = df.dropna(subset=existing_critical)
            
            final_count = len(df)
            removed_count = initial_count - final_count
            
            logger.info(f"Data cleaning complete. Removed {removed_count} invalid records ({removed_count/initial_count*100:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df
    
    def _clean_time_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize time data."""
        try:
            if 'time' in df.columns:
                # Convert to datetime
                df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
                
                # Remove invalid dates
                df = df.dropna(subset=['time'])
                
                # Filter reasonable date range (Argo program started in 1999)
                min_date = pd.Timestamp('1999-01-01', tz='utc')
                max_date = pd.Timestamp.now(tz='utc')
                
                df = df[(df['time'] >= min_date) & (df['time'] <= max_date)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning time data: {e}")
            return df
    
    def _clean_geographic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean geographic coordinates."""
        try:
            # Clean latitude
            if 'latitude' in df.columns:
                df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
                df = df[(df['latitude'] >= -90.0) & (df['latitude'] <= 90.0)]
            
            # Clean longitude
            if 'longitude' in df.columns:
                df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
                df = df[(df['longitude'] >= -180.0) & (df['longitude'] <= 180.0)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning geographic data: {e}")
            return df
    
    def _clean_measurements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean oceanographic measurements."""
        try:
            # Clean temperature
            if 'temperature' in df.columns:
                df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
                temp_min, temp_max = self.temperature_bounds
                df.loc[(df['temperature'] < temp_min) | (df['temperature'] > temp_max), 'temperature'] = np.nan
            
            # Clean salinity
            if 'salinity' in df.columns:
                df['salinity'] = pd.to_numeric(df['salinity'], errors='coerce')
                sal_min, sal_max = self.salinity_bounds
                df.loc[(df['salinity'] < sal_min) | (df['salinity'] > sal_max), 'salinity'] = np.nan
            
            # Clean pressure
            if 'pressure' in df.columns:
                df['pressure'] = pd.to_numeric(df['pressure'], errors='coerce')
                pres_min, pres_max = self.pressure_bounds
                df.loc[(df['pressure'] < pres_min) | (df['pressure'] > pres_max), 'pressure'] = np.nan
            
            # Clean depth
            if 'depth' in df.columns:
                df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
                depth_min, depth_max = self.depth_bounds
                df.loc[(df['depth'] < depth_min) | (df['depth'] > depth_max), 'depth'] = np.nan
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning measurements: {e}")
            return df
    
    def _create_floats_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create floats metadata DataFrame.
        
        Args:
            df: Processed data DataFrame
            
        Returns:
            pd.DataFrame: Floats metadata
        """
        try:
            # Group by float_id to get metadata
            float_groups = df.groupby('float_id')
            
            floats_data = []
            
            for float_id, group in float_groups:
                # Calculate deployment date (earliest time)
                deploy_date = group['time'].min()
                
                # Create properties JSON with summary statistics
                properties = {
                    'total_profiles': len(group['profile_id'].unique()) if 'profile_id' in group.columns else len(group),
                    'date_range': {
                        'start': deploy_date.isoformat(),
                        'end': group['time'].max().isoformat()
                    },
                    'location_range': {
                        'lat_min': float(group['latitude'].min()),
                        'lat_max': float(group['latitude'].max()),
                        'lon_min': float(group['longitude'].min()),
                        'lon_max': float(group['longitude'].max())
                    },
                    'measurements': {}
                }
                
                # Add measurement statistics if available
                for var in ['temperature', 'salinity', 'pressure', 'depth']:
                    if var in group.columns and not group[var].isna().all():
                        properties['measurements'][var] = {
                            'count': int(group[var].count()),
                            'min': float(group[var].min()),
                            'max': float(group[var].max()),
                            'mean': float(group[var].mean())
                        }
                
                float_data = {
                    'float_id': str(float_id),
                    'platform_number': str(float_id),  # Often the same as float_id
                    'deploy_date': deploy_date,
                    'properties': properties
                }
                
                floats_data.append(float_data)
            
            floats_df = pd.DataFrame(floats_data)
            logger.info(f"Created floats DataFrame with {len(floats_df)} floats")
            
            return floats_df
            
        except Exception as e:
            logger.error(f"Error creating floats DataFrame: {e}")
            return pd.DataFrame()
    
    def _create_profiles_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create profiles DataFrame in the format expected by the database schema.
        
        Args:
            df: Processed data DataFrame
            
        Returns:
            pd.DataFrame: Profiles data
        """
        try:
            profiles_data = []
            
            # Group by float and time to create profiles
            if 'profile_id' in df.columns:
                profile_groups = df.groupby(['float_id', 'profile_id'])
            else:
                # Create profile groups based on float and time
                df['profile_group'] = df.groupby(['float_id', 'time']).ngroup()
                profile_groups = df.groupby(['float_id', 'profile_group'])
            
            for (float_id, profile_key), group in profile_groups:
                
                # Create records for temperature and salinity separately (following schema design)
                profile_time = group['time'].iloc[0]
                lat = group['latitude'].iloc[0]
                lon = group['longitude'].iloc[0]
                
                # Generate unique profile ID
                profile_id = f"{float_id}_{profile_time.strftime('%Y%m%d_%H%M%S')}"
                
                # Create temperature profiles
                temp_data = group.dropna(subset=['temperature']) if 'temperature' in group.columns else pd.DataFrame()
                for _, row in temp_data.iterrows():
                    profiles_data.append({
                        'profile_id': f"{profile_id}_TEMP_{row.get('level', 0)}",
                        'float_id': str(float_id),
                        'profile_time': profile_time,
                        'lat': float(lat),
                        'lon': float(lon),
                        'pressure': float(row.get('pressure', np.nan)) if pd.notna(row.get('pressure')) else None,
                        'depth': float(row.get('depth', np.nan)) if pd.notna(row.get('depth')) else None,
                        'variable_name': 'TEMP',
                        'variable_value': float(row['temperature']),
                        'level': int(row.get('level', 0)),
                        'raw_profile': None  # Could store additional metadata here
                    })
                
                # Create salinity profiles
                sal_data = group.dropna(subset=['salinity']) if 'salinity' in group.columns else pd.DataFrame()
                for _, row in sal_data.iterrows():
                    profiles_data.append({
                        'profile_id': f"{profile_id}_PSAL_{row.get('level', 0)}",
                        'float_id': str(float_id),
                        'profile_time': profile_time,
                        'lat': float(lat),
                        'lon': float(lon),
                        'pressure': float(row.get('pressure', np.nan)) if pd.notna(row.get('pressure')) else None,
                        'depth': float(row.get('depth', np.nan)) if pd.notna(row.get('depth')) else None,
                        'variable_name': 'PSAL',
                        'variable_value': float(row['salinity']),
                        'level': int(row.get('level', 0)),
                        'raw_profile': None
                    })
            
            profiles_df = pd.DataFrame(profiles_data)
            logger.info(f"Created profiles DataFrame with {len(profiles_df)} records")
            
            return profiles_df
            
        except Exception as e:
            logger.error(f"Error creating profiles DataFrame: {e}")
            return pd.DataFrame()
    
    def validate_dataframes(self, floats_df: pd.DataFrame, profiles_df: pd.DataFrame) -> bool:
        """
        Validate the processed DataFrames.
        
        Args:
            floats_df: Floats DataFrame
            profiles_df: Profiles DataFrame
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Check floats DataFrame
            if not floats_df.empty:
                required_float_cols = ['float_id', 'platform_number', 'deploy_date', 'properties']
                missing_cols = [col for col in required_float_cols if col not in floats_df.columns]
                if missing_cols:
                    logger.error(f"Missing required float columns: {missing_cols}")
                    return False
            
            # Check profiles DataFrame
            if not profiles_df.empty:
                required_profile_cols = ['profile_id', 'float_id', 'profile_time', 'lat', 'lon', 'variable_name', 'variable_value']
                missing_cols = [col for col in required_profile_cols if col not in profiles_df.columns]
                if missing_cols:
                    logger.error(f"Missing required profile columns: {missing_cols}")
                    return False
            
            logger.info("DataFrame validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating DataFrames: {e}")
            return False


def create_preprocessor() -> ArgoDataPreprocessor:
    """
    Factory function to create a data preprocessor.
    
    Returns:
        ArgoDataPreprocessor: Configured preprocessor
    """
    return ArgoDataPreprocessor()


if __name__ == "__main__":
    # Test the preprocessor
    try:
        preprocessor = create_preprocessor()
        
        # Create sample data for testing
        sample_data = {
            'float_id': np.array(['1234', '1234', '1234', '5678', '5678']),
            'time': np.array(['2023-01-01T12:00:00Z', '2023-01-01T12:00:00Z', '2023-01-01T12:00:00Z', 
                             '2023-01-02T06:00:00Z', '2023-01-02T06:00:00Z']),
            'latitude': np.array([10.5, 10.5, 10.5, -5.2, -5.2]),
            'longitude': np.array([75.3, 75.3, 75.3, 82.1, 82.1]),
            'pressure': np.array([10.0, 50.0, 100.0, 5.0, 25.0]),
            'temperature': np.array([28.5, 25.2, 22.1, 29.1, 27.8]),
            'salinity': np.array([35.1, 35.3, 35.5, 34.8, 35.0]),
            'level': np.array([0, 1, 2, 0, 1])
        }
        
        floats_df, profiles_df = preprocessor.process_raw_data(sample_data)
        
        if preprocessor.validate_dataframes(floats_df, profiles_df):
            print("Preprocessor test successful!")
            print(f"Generated {len(floats_df)} float records and {len(profiles_df)} profile records")
        else:
            print("Preprocessor validation failed!")
        
    except Exception as e:
        print(f"Preprocessor test failed: {e}")