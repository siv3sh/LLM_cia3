"""
Data Processor for cleaning, preprocessing, and validating data
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from core.config import Config
from data.schemas import DataPreprocessingConfig, DataQualityReport


class DataProcessor:
    """
    Handles data cleaning, preprocessing, and validation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Preprocessing components
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        
        # Processing statistics
        self.processing_stats: Dict[str, Any] = {}
        
        # Data quality metrics
        self.quality_metrics: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize the data processor"""
        try:
            # Initialize preprocessing components
            self.scaler = StandardScaler()
            
            self.logger.info("Data processor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data processor: {e}")
            raise
    
    async def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the input data"""
        try:
            self.logger.info("Starting data cleaning process")
            
            if data is None or data.empty:
                raise ValueError("No data provided for cleaning")
            
            # Create a copy to avoid modifying original data
            cleaned_data = data.copy()
            
            # Remove metadata columns
            metadata_columns = [col for col in cleaned_data.columns if col.startswith('_')]
            if metadata_columns:
                cleaned_data = cleaned_data.drop(columns=metadata_columns)
                self.logger.info(f"Removed {len(metadata_columns)} metadata columns")
            
            # Clean column names
            cleaned_data = self._clean_column_names(cleaned_data)
            
            # Remove completely empty rows and columns
            cleaned_data = self._remove_empty_rows_columns(cleaned_data)
            
            # Standardize data types
            cleaned_data = await self._standardize_data_types(cleaned_data)
            
            # Store cleaning statistics
            self.processing_stats["cleaning"] = {
                "original_shape": data.shape,
                "cleaned_shape": cleaned_data.shape,
                "columns_removed": len(data.columns) - len(cleaned_data.columns),
                "rows_removed": len(data) - len(cleaned_data),
                "cleaning_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Data cleaning completed: {cleaned_data.shape}")
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            raise
    
    async def handle_missing_values(self, data: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
        """Handle missing values in the data"""
        try:
            self.logger.info(f"Handling missing values using strategy: {strategy}")
            
            if data is None or data.empty:
                raise ValueError("No data provided for missing value handling")
            
            # Create a copy
            processed_data = data.copy()
            
            # Analyze missing values
            missing_analysis = self._analyze_missing_values(processed_data)
            
            # Apply missing value strategy
            if strategy == "auto":
                processed_data = await self._apply_auto_missing_strategy(processed_data, missing_analysis)
            else:
                processed_data = await self._apply_missing_strategy(processed_data, strategy)
            
            # Store processing statistics
            self.processing_stats["missing_values"] = {
                "original_missing": missing_analysis["total_missing"],
                "missing_percentage": missing_analysis["missing_percentage"],
                "strategy_used": strategy,
                "columns_processed": list(missing_analysis["column_missing"].keys()),
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Missing value handling completed")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Missing value handling failed: {e}")
            raise
    
    async def handle_outliers(self, data: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers in numerical columns"""
        try:
            self.logger.info(f"Handling outliers using method: {method}")
            
            if data is None or data.empty:
                raise ValueError("No data provided for outlier handling")
            
            # Create a copy
            processed_data = data.copy()
            
            # Get numerical columns
            numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
            
            if len(numerical_columns) == 0:
                self.logger.info("No numerical columns found for outlier handling")
                return processed_data
            
            # Detect and handle outliers
            outlier_stats = {}
            for column in numerical_columns:
                try:
                    original_data = processed_data[column].copy()
                    
                    if method == "iqr":
                        processed_data[column] = self._handle_outliers_iqr(processed_data[column], threshold)
                    elif method == "zscore":
                        processed_data[column] = self._handle_outliers_zscore(processed_data[column], threshold)
                    elif method == "isolation_forest":
                        processed_data[column] = await self._handle_outliers_isolation_forest(processed_data[column])
                    else:
                        self.logger.warning(f"Unknown outlier method: {method}")
                        continue
                    
                    # Calculate outlier statistics
                    outliers_removed = (original_data != processed_data[column]).sum()
                    outlier_stats[column] = {
                        "outliers_removed": int(outliers_removed),
                        "outlier_percentage": float(outliers_removed / len(original_data) * 100)
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to handle outliers in column {column}: {e}")
                    continue
            
            # Store processing statistics
            self.processing_stats["outliers"] = {
                "method_used": method,
                "threshold": threshold,
                "columns_processed": list(outlier_stats.keys()),
                "outlier_statistics": outlier_stats,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Outlier handling completed")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Outlier handling failed: {e}")
            raise
    
    async def validate_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data against expected schema"""
        try:
            self.logger.info("Validating data schema")
            
            if data is None or data.empty:
                raise ValueError("No data provided for schema validation")
            
            # Create a copy
            validated_data = data.copy()
            
            # Validate required columns
            required_columns = ['attrition']
            missing_required = [col for col in required_columns if col not in validated_data.columns]
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")
            
            # Validate data types
            validated_data = await self._validate_data_types(validated_data)
            
            # Validate data ranges
            validated_data = await self._validate_data_ranges(validated_data)
            
            # Validate categorical values
            validated_data = await self._validate_categorical_values(validated_data)
            
            # Store validation statistics
            self.processing_stats["validation"] = {
                "validation_timestamp": datetime.utcnow().isoformat(),
                "required_columns_present": True,
                "data_types_validated": True,
                "data_ranges_validated": True,
                "categorical_values_validated": True
            }
            
            self.logger.info("Data schema validation completed")
            
            return validated_data
            
        except Exception as e:
            self.logger.error(f"Schema validation failed: {e}")
            raise
    
    async def create_data_quality_report(self, data: pd.DataFrame) -> DataQualityReport:
        """Create a comprehensive data quality report"""
        try:
            self.logger.info("Creating data quality report")
            
            if data is None or data.empty:
                raise ValueError("No data provided for quality report")
            
            # Calculate quality metrics
            total_records = len(data)
            total_features = len(data.columns)
            
            # Missing values analysis
            missing_values = data.isnull().sum().sum()
            missing_percentage = (missing_values / (total_records * total_features)) * 100
            
            # Duplicate records
            duplicate_records = len(data[data.duplicated()])
            
            # Data types
            data_types = data.dtypes.to_dict()
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(data, missing_percentage, duplicate_records)
            
            # Create quality report
            quality_report = DataQualityReport(
                timestamp=datetime.utcnow(),
                total_records=total_records,
                total_features=total_features,
                missing_values_percentage=missing_percentage,
                duplicate_records=duplicate_records,
                data_types=data_types,
                quality_score=quality_score
            )
            
            # Store quality metrics
            self.quality_metrics = {
                "total_records": total_records,
                "total_features": total_features,
                "missing_values": missing_values,
                "missing_percentage": missing_percentage,
                "duplicate_records": duplicate_records,
                "quality_score": quality_score,
                "data_types": data_types
            }
            
            self.logger.info(f"Data quality report created: Score = {quality_score:.1f}")
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Data quality report creation failed: {e}")
            raise
    
    def _clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean column names"""
        try:
            # Remove leading/trailing whitespace
            data.columns = data.columns.str.strip()
            
            # Replace spaces with underscores
            data.columns = data.columns.str.replace(' ', '_')
            
            # Convert to lowercase
            data.columns = data.columns.str.lower()
            
            # Remove special characters
            data.columns = data.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Column name cleaning failed: {e}")
            return data
    
    def _remove_empty_rows_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty rows and columns"""
        try:
            # Remove completely empty columns
            data = data.dropna(axis=1, how='all')
            
            # Remove completely empty rows
            data = data.dropna(axis=0, how='all')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Empty row/column removal failed: {e}")
            return data
    
    async def _standardize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types"""
        try:
            # Convert attrition column to standard format
            if 'attrition' in data.columns:
                data['attrition'] = data['attrition'].map({
                    'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 'Left': 1, 'Stayed': 0
                })
            
            # Convert numerical columns
            numerical_columns = ['age', 'daily_rate', 'hourly_rate', 'monthly_rate', 'monthly_income']
            for col in numerical_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Convert categorical columns
            categorical_columns = ['gender', 'education', 'marital_status', 'job_role', 'department']
            for col in categorical_columns:
                if col in data.columns:
                    data[col] = data[col].astype('category')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data type standardization failed: {e}")
            return data
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in the data"""
        try:
            missing_counts = data.isnull().sum()
            total_cells = data.size
            
            return {
                "total_missing": int(missing_counts.sum()),
                "missing_percentage": float(missing_counts.sum() / total_cells * 100),
                "column_missing": missing_counts.to_dict(),
                "columns_with_missing": list(missing_counts[missing_counts > 0].index)
            }
            
        except Exception as e:
            self.logger.error(f"Missing value analysis failed: {e}")
            return {}
    
    async def _apply_auto_missing_strategy(self, data: pd.DataFrame, missing_analysis: Dict[str, Any]) -> pd.DataFrame:
        """Apply automatic missing value strategy based on data analysis"""
        try:
            processed_data = data.copy()
            
            for column in missing_analysis.get("columns_with_missing", []):
                missing_percentage = missing_analysis["column_missing"][column] / len(data) * 100
                
                if missing_percentage > 50:
                    # High missing values - drop column
                    processed_data = processed_data.drop(columns=[column])
                    self.logger.info(f"Dropped column {column} due to high missing values ({missing_percentage:.1f}%)")
                
                elif missing_percentage > 20:
                    # Medium missing values - use median for numerical, mode for categorical
                    if pd.api.types.is_numeric_dtype(data[column]):
                        median_value = data[column].median()
                        processed_data[column] = data[column].fillna(median_value)
                        self.logger.info(f"Filled missing values in {column} with median: {median_value}")
                    else:
                        mode_value = data[column].mode().iloc[0] if not data[column].mode().empty else "Unknown"
                        processed_data[column] = data[column].fillna(mode_value)
                        self.logger.info(f"Filled missing values in {column} with mode: {mode_value}")
                
                else:
                    # Low missing values - use forward fill then backward fill
                    processed_data[column] = data[column].fillna(method='ffill').fillna(method='bfill')
                    self.logger.info(f"Filled missing values in {column} using forward/backward fill")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Auto missing value strategy failed: {e}")
            return data
    
    async def _apply_missing_strategy(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply specific missing value strategy"""
        try:
            processed_data = data.copy()
            
            if strategy == "drop":
                # Drop rows with any missing values
                processed_data = processed_data.dropna()
                self.logger.info("Dropped all rows with missing values")
            
            elif strategy == "mean":
                # Fill numerical columns with mean
                numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
                for col in numerical_columns:
                    mean_value = processed_data[col].mean()
                    processed_data[col] = processed_data[col].fillna(mean_value)
                self.logger.info("Filled numerical missing values with mean")
            
            elif strategy == "median":
                # Fill numerical columns with median
                numerical_columns = processed_data.select_dtypes(include=[np.number]).columns
                for col in numerical_columns:
                    median_value = processed_data[col].median()
                    processed_data[col] = processed_data[col].fillna(median_value)
                self.logger.info("Filled numerical missing values with median")
            
            elif strategy == "mode":
                # Fill categorical columns with mode
                categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
                for col in categorical_columns:
                    mode_value = processed_data[col].mode().iloc[0] if not processed_data[col].mode().empty else "Unknown"
                    processed_data[col] = processed_data[col].fillna(mode_value)
                self.logger.info("Filled categorical missing values with mode")
            
            elif strategy == "interpolate":
                # Interpolate missing values
                processed_data = processed_data.interpolate(method='linear')
                self.logger.info("Interpolated missing values")
            
            else:
                self.logger.warning(f"Unknown missing value strategy: {strategy}")
                return data
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Missing value strategy application failed: {e}")
            return data
    
    def _handle_outliers_iqr(self, series: pd.Series, threshold: float = 1.5) -> pd.Series:
        """Handle outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap outliers to bounds
            series_cleaned = series.copy()
            series_cleaned[series_cleaned < lower_bound] = lower_bound
            series_cleaned[series_cleaned > upper_bound] = upper_bound
            
            return series_cleaned
            
        except Exception as e:
            self.logger.error(f"IQR outlier handling failed: {e}")
            return series
    
    def _handle_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Handle outliers using Z-score method"""
        try:
            z_scores = np.abs(stats.zscore(series.dropna()))
            
            # Cap outliers to threshold
            series_cleaned = series.copy()
            outlier_mask = z_scores > threshold
            
            if outlier_mask.any():
                # Replace outliers with median
                median_value = series.median()
                series_cleaned.iloc[outlier_mask] = median_value
            
            return series_cleaned
            
        except Exception as e:
            self.logger.error(f"Z-score outlier handling failed: {e}")
            return series
    
    async def _handle_outliers_isolation_forest(self, series: pd.Series) -> pd.Series:
        """Handle outliers using Isolation Forest (placeholder)"""
        try:
            # This would implement Isolation Forest outlier detection
            # For now, return the original series
            self.logger.info("Isolation Forest outlier handling not implemented yet")
            return series
            
        except Exception as e:
            self.logger.error(f"Isolation Forest outlier handling failed: {e}")
            return series
    
    async def _validate_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct data types"""
        try:
            # Validate numerical columns
            numerical_columns = ['age', 'daily_rate', 'hourly_rate', 'monthly_rate', 'monthly_income']
            for col in numerical_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Validate categorical columns
            categorical_columns = ['gender', 'education', 'marital_status', 'job_role', 'department']
            for col in categorical_columns:
                if col in data.columns:
                    data[col] = data[col].astype('category')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data type validation failed: {e}")
            return data
    
    async def _validate_data_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges for numerical columns"""
        try:
            # Validate age range
            if 'age' in data.columns:
                data = data[(data['age'] >= 18) & (data['age'] <= 100)]
            
            # Validate performance rating
            if 'performance_rating' in data.columns:
                data = data[(data['performance_rating'] >= 1) & (data['performance_rating'] <= 5)]
            
            # Validate satisfaction scores
            satisfaction_columns = ['relationship_satisfaction']
            for col in satisfaction_columns:
                if col in data.columns:
                    data = data[(data[col] >= 1) & (data[col] <= 5)]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Data range validation failed: {e}")
            return data
    
    async def _validate_categorical_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate categorical values"""
        try:
            # Validate attrition values
            if 'attrition' in data.columns:
                valid_attrition = [0, 1]
                data = data[data['attrition'].isin(valid_attrition)]
            
            # Validate gender values
            if 'gender' in data.columns:
                valid_genders = ['Male', 'Female', 'M', 'F']
                data = data[data['gender'].isin(valid_genders)]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Categorical value validation failed: {e}")
            return data
    
    def _calculate_quality_score(self, data: pd.DataFrame, missing_percentage: float, duplicate_records: int) -> float:
        """Calculate overall data quality score"""
        try:
            score = 100.0
            
            # Deduct points for missing values
            if missing_percentage > 0:
                score -= min(missing_percentage * 0.5, 30)  # Max 30 points deduction
            
            # Deduct points for duplicates
            duplicate_percentage = duplicate_records / len(data) * 100 if len(data) > 0 else 0
            if duplicate_percentage > 0:
                score -= min(duplicate_percentage * 0.3, 20)  # Max 20 points deduction
            
            # Deduct points for data type inconsistencies
            score -= 5.0  # Placeholder deduction
            
            # Deduct points for range violations
            score -= 5.0  # Placeholder deduction
            
            return max(0.0, score)
            
        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics about data processing operations"""
        try:
            return {
                "processing_stats": self.processing_stats,
                "quality_metrics": self.quality_metrics,
                "total_operations": len(self.processing_stats),
                "last_processing": max(
                    (stats.get("processing_timestamp", "1970-01-01") for stats in self.processing_stats.values()),
                    default="1970-01-01"
                )
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get processing statistics: {e}")
            return {"error": f"Statistics retrieval failed: {str(e)}"}
    
    async def export_processed_data(self, data: pd.DataFrame, format: str = "csv", path: str = None) -> str:
        """Export processed data to file"""
        try:
            if data is None or data.empty:
                raise ValueError("No data to export")
            
            if path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = f"./exports/processed_data_{timestamp}"
            
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
            
            self.logger.info(f"Processed data exported to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the data processor"""
        try:
            # Clear processing components
            self.scaler = None
            self.label_encoders.clear()
            self.imputers.clear()
            
            # Clear statistics
            self.processing_stats.clear()
            self.quality_metrics.clear()
            
            self.logger.info("Data processor shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Data processor shutdown failed: {e}")
