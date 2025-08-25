"""
Data Loader for collecting data from various sources
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
# import aiofiles  # Not used - removed
import httpx
import json

from core.config import Config
from data.schemas import DataCollectionConfig, DataSourceType, AttritionDataSchema


class DataLoader:
    """
    Handles data collection from various sources
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # HTTP client for API calls
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # Collection statistics
        self.collection_stats: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize the data loader"""
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                timeout=self.config.groq_timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            
            self.logger.info("Data loader initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data loader: {e}")
            raise
    
    async def collect_data(self, config: DataCollectionConfig) -> pd.DataFrame:
        """Collect data from specified sources"""
        try:
            self.logger.info(f"Starting data collection from {len(config.sources)} sources")
            
            collected_data = []
            
            for source in config.sources:
                try:
                    if source == DataSourceType.CSV:
                        data = await self._collect_csv_data(config.csv_paths or [])
                    elif source == DataSourceType.DATABASE:
                        data = await self._collect_database_data(config.database_query)
                    elif source == DataSourceType.API:
                        data = await self._collect_api_data(config.api_endpoints or [])
                    elif source == DataSourceType.EXCEL:
                        data = await self._collect_excel_data(config.csv_paths or [])
                    elif source == DataSourceType.JSON:
                        data = await self._collect_json_data(config.csv_paths or [])
                    else:
                        self.logger.warning(f"Unknown data source: {source}")
                        continue
                    
                    if data is not None and not data.empty:
                        collected_data.append(data)
                        self.logger.info(f"Collected {len(data)} records from {source}")
                        
                        # Store collection statistics
                        self.collection_stats[source] = {
                            "records_collected": len(data),
                            "columns": list(data.columns),
                            "data_types": data.dtypes.to_dict(),
                            "collection_time": datetime.utcnow().isoformat()
                        }
                    
                except Exception as e:
                    self.logger.error(f"Failed to collect data from {source}: {e}")
                    continue
            
            if not collected_data:
                raise ValueError("No data collected from any source")
            
            # Combine collected data
            combined_data = pd.concat(collected_data, ignore_index=True)
            
            # Cache the combined data
            cache_key = f"combined_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.data_cache[cache_key] = combined_data
            
            self.logger.info(f"Data collection completed: {len(combined_data)} total records")
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            raise
    
    async def _collect_csv_data(self, file_paths: List[str]) -> Optional[pd.DataFrame]:
        """Collect data from CSV files"""
        try:
            if not file_paths:
                return None
            
            data_frames = []
            
            for file_path in file_paths:
                try:
                    if not Path(file_path).exists():
                        self.logger.warning(f"CSV file not found: {file_path}")
                        continue
                    
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    if not df.empty:
                        # Add source information
                        df['_source_file'] = file_path
                        df['_collection_timestamp'] = datetime.utcnow()
                        
                        data_frames.append(df)
                        self.logger.info(f"Loaded CSV data from {file_path}: {df.shape}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load CSV file {file_path}: {e}")
                    continue
            
            if not data_frames:
                return None
            
            # Combine all CSV data
            combined_df = pd.concat(data_frames, ignore_index=True)
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"CSV data collection failed: {e}")
            return None
    
    async def _collect_excel_data(self, file_paths: List[str]) -> Optional[pd.DataFrame]:
        """Collect data from Excel files"""
        try:
            if not file_paths:
                return None
            
            data_frames = []
            
            for file_path in file_paths:
                try:
                    if not Path(file_path).exists():
                        self.logger.warning(f"Excel file not found: {file_path}")
                        continue
                    
                    # Read Excel file
                    excel_file = pd.ExcelFile(file_path)
                    
                    for sheet_name in excel_file.sheet_names:
                        try:
                            df = pd.read_excel(file_path, sheet_name=sheet_name)
                            
                            if not df.empty:
                                # Add source information
                                df['_source_file'] = file_path
                                df['_sheet_name'] = sheet_name
                                df['_collection_timestamp'] = datetime.utcnow()
                                
                                data_frames.append(df)
                                self.logger.info(f"Loaded Excel data from {file_path} sheet {sheet_name}: {df.shape}")
                        
                        except Exception as e:
                            self.logger.warning(f"Failed to load sheet {sheet_name} from {file_path}: {e}")
                            continue
                    
                except Exception as e:
                    self.logger.error(f"Failed to load Excel file {file_path}: {e}")
                    continue
            
            if not data_frames:
                return None
            
            # Combine all Excel data
            combined_df = pd.concat(data_frames, ignore_index=True)
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Excel data collection failed: {e}")
            return None
    
    async def _collect_json_data(self, file_paths: List[str]) -> Optional[pd.DataFrame]:
        """Collect data from JSON files"""
        try:
            if not file_paths:
                return None
            
            data_frames = []
            
            for file_path in file_paths:
                try:
                    if not Path(file_path).exists():
                        self.logger.warning(f"JSON file not found: {file_path}")
                        continue
                    
                    # Read JSON file
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Convert JSON to DataFrame
                    if isinstance(json_data, list):
                        df = pd.DataFrame(json_data)
                    elif isinstance(json_data, dict):
                        # Handle nested JSON structures
                        df = pd.json_normalize(json_data)
                    else:
                        self.logger.warning(f"Unsupported JSON structure in {file_path}")
                        continue
                    
                    if not df.empty:
                        # Add source information
                        df['_source_file'] = file_path
                        df['_collection_timestamp'] = datetime.utcnow()
                        
                        data_frames.append(df)
                        self.logger.info(f"Loaded JSON data from {file_path}: {df.shape}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load JSON file {file_path}: {e}")
                    continue
            
            if not data_frames:
                return None
            
            # Combine all JSON data
            combined_df = pd.concat(data_frames, ignore_index=True)
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"JSON data collection failed: {e}")
            return None
    
    async def _collect_database_data(self, query: str) -> Optional[pd.DataFrame]:
        """Collect data from database"""
        try:
            if not query:
                return None
            
            # This would implement database connection and query execution
            # For now, return None as placeholder
            self.logger.info("Database data collection not implemented yet")
            return None
            
        except Exception as e:
            self.logger.error(f"Database data collection failed: {e}")
            return None
    
    async def _collect_api_data(self, endpoints: List[str]) -> Optional[pd.DataFrame]:
        """Collect data from APIs"""
        try:
            if not endpoints or not self.http_client:
                return None
            
            data_frames = []
            
            for endpoint in endpoints:
                try:
                    # Make API request
                    response = await self.http_client.get(endpoint)
                    response.raise_for_status()
                    
                    # Parse response
                    data = response.json()
                    
                    # Convert to DataFrame
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        df = pd.json_normalize(data)
                    else:
                        self.logger.warning(f"Unsupported API response structure from {endpoint}")
                        continue
                    
                    if not df.empty:
                        # Add source information
                        df['_source_endpoint'] = endpoint
                        df['_collection_timestamp'] = datetime.utcnow()
                        
                        data_frames.append(df)
                        self.logger.info(f"Loaded API data from {endpoint}: {df.shape}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to collect data from API {endpoint}: {e}")
                    continue
            
            if not data_frames:
                return None
            
            # Combine all API data
            combined_df = pd.concat(data_frames, ignore_index=True)
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"API data collection failed: {e}")
            return None
    
    async def validate_data_schema(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data against expected schema"""
        try:
            validation_errors = []
            
            # Check for required columns (basic attrition analysis columns)
            required_columns = ['attrition']
            optional_columns = [
                'age', 'gender', 'education', 'marital_status', 'job_role', 'department',
                'business_travel', 'daily_rate', 'hourly_rate', 'monthly_rate',
                'monthly_income', 'overtime', 'percent_salary_hike', 'performance_rating',
                'relationship_satisfaction', 'stock_option_level', 'total_working_years',
                'training_times_last_year', 'years_at_company', 'years_in_current_role',
                'years_since_last_promotion', 'years_with_curr_manager'
            ]
            
            # Check required columns
            missing_required = [col for col in required_columns if col not in data.columns]
            if missing_required:
                validation_errors.append(f"Missing required columns: {missing_required}")
            
            # Check data types for numerical columns
            numerical_columns = ['age', 'daily_rate', 'hourly_rate', 'monthly_rate', 'monthly_income']
            for col in numerical_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        validation_errors.append(f"Column {col} is not numeric")
            
            # Check attrition column values
            if 'attrition' in data.columns:
                attrition_values = data['attrition'].unique()
                expected_values = ['Yes', 'No', 'True', 'False', 'Left', 'Stayed']
                invalid_values = [val for val in attrition_values if val not in expected_values]
                if invalid_values:
                    validation_errors.append(f"Invalid attrition values: {invalid_values}")
            
            # Check for reasonable data ranges
            if 'age' in data.columns:
                age_range = data['age'].describe()
                if age_range['min'] < 18 or age_range['max'] > 100:
                    validation_errors.append("Age values outside reasonable range (18-100)")
            
            is_valid = len(validation_errors) == 0
            
            return is_valid, validation_errors
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    async def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for the collected data"""
        try:
            if data is None or data.empty:
                return {"error": "No data available"}
            
            summary = {
                "total_records": len(data),
                "total_columns": len(data.columns),
                "column_info": {},
                "missing_values": {},
                "data_types": data.dtypes.to_dict(),
                "collection_timestamp": datetime.utcnow().isoformat()
            }
            
            # Analyze each column
            for column in data.columns:
                col_data = data[column]
                
                # Skip metadata columns
                if column.startswith('_'):
                    continue
                
                col_info = {
                    "data_type": str(col_data.dtype),
                    "missing_count": col_data.isnull().sum(),
                    "missing_percentage": (col_data.isnull().sum() / len(col_data)) * 100
                }
                
                # Add statistics based on data type
                if pd.api.types.is_numeric_dtype(col_data):
                    col_info.update({
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std())
                    })
                elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                    unique_values = col_data.nunique()
                    col_info.update({
                        "unique_values": int(unique_values),
                        "most_common": col_data.mode().iloc[0] if not col_data.mode().empty else None
                    })
                
                summary["column_info"][column] = col_info
                summary["missing_values"][column] = col_info["missing_count"]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate data summary: {e}")
            return {"error": f"Summary generation failed: {str(e)}"}
    
    async def export_data(self, data: pd.DataFrame, format: str = "csv", path: str = None) -> str:
        """Export collected data to file"""
        try:
            if data is None or data.empty:
                raise ValueError("No data to export")
            
            if path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = f"./exports/collected_data_{timestamp}"
            
            # Create export directory
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                export_path = f"{path}.csv"
                data.to_csv(export_path, index=False)
            elif format.lower() == "excel":
                export_path = f"{path}.xlsx"
                data.to_excel(export_path, index=False)
            elif format.lower() == "parquet":
                export_path = f"{path}.parquet"
                data.to_parquet(export_path, index=False)
            elif format.lower() == "json":
                export_path = f"{path}.json"
                data.to_json(export_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Data exported to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            raise
    
    async def get_collection_statistics(self) -> Dict[str, Any]:
        """Get statistics about data collection operations"""
        try:
            return {
                "total_sources": len(self.collection_stats),
                "total_records_collected": sum(
                    stats.get("records_collected", 0) for stats in self.collection_stats.values()
                ),
                "source_details": self.collection_stats,
                "cache_size": len(self.data_cache),
                "last_collection": max(
                    (stats.get("collection_time", "1970-01-01") for stats in self.collection_stats.values()),
                    default="1970-01-01"
                )
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection statistics: {e}")
            return {"error": f"Statistics retrieval failed: {str(e)}"}
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear the data cache"""
        try:
            cache_size = len(self.data_cache)
            self.data_cache.clear()
            
            self.logger.info(f"Cleared data cache ({cache_size} entries)")
            
            return {
                "status": "success",
                "message": f"Cache cleared successfully",
                "entries_removed": cache_size
            }
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return {"error": f"Cache clearing failed: {str(e)}"}
    
    async def shutdown(self):
        """Shutdown the data loader"""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            # Clear cache
            self.data_cache.clear()
            
            self.logger.info("Data loader shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Data loader shutdown failed: {e}")
