"""
Data Agent for handling data collection, preprocessing, and feature engineering
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from .base_agent import BaseAgent, AgentMessage
from core.config import Config
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from data.schemas import AttritionDataSchema, DataQualityReport


class DataAgent(BaseAgent):
    """
    Agent responsible for data collection, preprocessing, and feature engineering
    """
    
    def __init__(self, config: Config, agent_id: Optional[str] = None):
        super().__init__(config, agent_id)
        
        # Data processing components
        self.data_loader = DataLoader(config)
        self.data_processor = DataProcessor(config)
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.feature_metadata: Dict[str, Any] = {}
        
        # Data quality tracking
        self.quality_reports: List[DataQualityReport] = []
        self.data_sources: List[str] = []
        
        # Setup data-specific message handlers
        self._setup_data_handlers()
    
    def _setup_data_handlers(self):
        """Setup data-specific message handlers"""
        self.message_handlers.update({
            "data_collection": self._handle_data_collection,
            "data_preprocessing": self._handle_data_preprocessing,
            "feature_engineering": self._handle_feature_engineering,
            "data_quality_check": self._handle_data_quality_check,
            "data_export": self._handle_data_export,
        })
    
    async def _initialize_agent(self):
        """Initialize data agent specific components"""
        try:
            # Initialize data loader and processor
            await self.data_loader.initialize()
            await self.data_processor.initialize()
            
            # Create data directories if they don't exist
            Path(self.config.vector_store_path).mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Data agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data agent components: {e}")
            raise
    
    async def _execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data-related tasks"""
        task_type = task_data.get("task_type")
        
        if task_type == "collect_data":
            return await self._collect_data(task_data)
        elif task_type == "preprocess_data":
            return await self._preprocess_data(task_data)
        elif task_type == "engineer_features":
            return await self._engineer_features(task_data)
        elif task_type == "validate_data":
            return await self._validate_data(task_data)
        elif task_type == "export_data":
            return await self._export_data(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _get_requested_data(self, request_data: Dict[str, Any]) -> Any:
        """Get requested data"""
        data_type = request_data.get("data_type")
        
        if data_type == "raw_data":
            return self.raw_data.to_dict() if self.raw_data is not None else None
        elif data_type == "processed_data":
            return self.processed_data.to_dict() if self.processed_data is not None else None
        elif data_type == "feature_metadata":
            return self.feature_metadata
        elif data_type == "quality_report":
            return self.quality_reports[-1] if self.quality_reports else None
        elif data_type == "data_summary":
            return self._get_data_summary()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    async def _handle_data_collection(self, message: AgentMessage):
        """Handle data collection requests"""
        try:
            collection_config = message.content
            result = await self._collect_data(collection_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="data_collection_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="data_collection_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_data_preprocessing(self, message: AgentMessage):
        """Handle data preprocessing requests"""
        try:
            preprocessing_config = message.content
            result = await self._preprocess_data(preprocessing_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="data_preprocessing_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="data_preprocessing_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_feature_engineering(self, message: AgentMessage):
        """Handle feature engineering requests"""
        try:
            feature_config = message.content
            result = await self._engineer_features(feature_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="feature_engineering_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="feature_engineering_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_data_quality_check(self, message: AgentMessage):
        """Handle data quality check requests"""
        try:
            quality_result = await self._validate_data({})
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=quality_result,
                message_type="data_quality_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Data quality check failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="data_quality_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _handle_data_export(self, message: AgentMessage):
        """Handle data export requests"""
        try:
            export_config = message.content
            result = await self._export_data(export_config)
            
            response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content=result,
                message_type="data_export_response",
                response_to=message.id
            )
            await self.send_message(response)
            
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            error_response = AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                content={"error": str(e)},
                message_type="data_export_error",
                response_to=message.id
            )
            await self.send_message(error_response)
    
    async def _collect_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from various sources"""
        try:
            self.state.current_task = "data_collection"
            self.update_task_progress(0.1)
            
            # Determine data sources
            sources = config.get("sources", ["csv", "database", "api"])
            self.data_sources = sources
            
            # Collect data from each source
            collected_data = []
            for i, source in enumerate(sources):
                self.update_task_progress(0.1 + (i * 0.3 / len(sources)))
                
                if source == "csv":
                    data = await self._collect_csv_data(config)
                elif source == "database":
                    data = await self._collect_database_data(config)
                elif source == "api":
                    data = await self._collect_api_data(config)
                else:
                    self.logger.warning(f"Unknown data source: {source}")
                    continue
                
                if data is not None:
                    collected_data.append(data)
            
            # Combine collected data
            if collected_data:
                self.raw_data = pd.concat(collected_data, ignore_index=True)
                self.logger.info(f"Collected {len(self.raw_data)} records from {len(sources)} sources")
            else:
                raise ValueError("No data collected from any source")
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "records_collected": len(self.raw_data),
                "sources": sources,
                "data_shape": self.raw_data.shape,
                "columns": list(self.raw_data.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _collect_csv_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Collect data from CSV files"""
        try:
            file_paths = config.get("csv_paths", [])
            if not file_paths:
                return None
            
            data_frames = []
            for file_path in file_paths:
                if Path(file_path).exists():
                    df = pd.read_csv(file_path)
                    data_frames.append(df)
                    self.logger.info(f"Loaded CSV data from {file_path}: {df.shape}")
                else:
                    self.logger.warning(f"CSV file not found: {file_path}")
            
            return pd.concat(data_frames, ignore_index=True) if data_frames else None
            
        except Exception as e:
            self.logger.error(f"CSV data collection failed: {e}")
            return None
    
    async def _collect_database_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Collect data from database"""
        try:
            query = config.get("database_query")
            if not query:
                return None
            
            # This would use the database connection from data_loader
            # For now, return None as placeholder
            self.logger.info("Database data collection not implemented yet")
            return None
            
        except Exception as e:
            self.logger.error(f"Database data collection failed: {e}")
            return None
    
    async def _collect_api_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Collect data from APIs"""
        try:
            api_endpoints = config.get("api_endpoints", [])
            if not api_endpoints:
                return None
            
            # This would use HTTP client to fetch data from APIs
            # For now, return None as placeholder
            self.logger.info("API data collection not implemented yet")
            return None
            
        except Exception as e:
            self.logger.error(f"API data collection failed: {e}")
            return None
    
    async def _preprocess_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the collected data"""
        try:
            if self.raw_data is None:
                raise ValueError("No raw data available for preprocessing")
            
            self.state.current_task = "data_preprocessing"
            self.update_task_progress(0.1)
            
            # Clean data
            self.update_task_progress(0.3)
            cleaned_data = await self.data_processor.clean_data(self.raw_data)
            
            # Handle missing values
            self.update_task_progress(0.5)
            cleaned_data = await self.data_processor.handle_missing_values(cleaned_data)
            
            # Handle outliers
            self.update_task_progress(0.7)
            cleaned_data = await self.data_processor.handle_outliers(cleaned_data)
            
            # Validate data schema
            self.update_task_progress(0.9)
            validated_data = await self.data_processor.validate_schema(cleaned_data)
            
            self.processed_data = validated_data
            self.update_task_progress(1.0)
            
            self.logger.info(f"Data preprocessing completed: {self.processed_data.shape}")
            
            return {
                "status": "success",
                "original_shape": self.raw_data.shape,
                "processed_shape": self.processed_data.shape,
                "cleaning_summary": "Data cleaned and validated successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _engineer_features(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features for attrition analysis"""
        try:
            if self.processed_data is None:
                raise ValueError("No processed data available for feature engineering")
            
            self.state.current_task = "feature_engineering"
            self.update_task_progress(0.1)
            
            # Create temporal features
            self.update_task_progress(0.2)
            temporal_features = await self._create_temporal_features()
            
            # Create categorical features
            self.update_task_progress(0.4)
            categorical_features = await self._create_categorical_features()
            
            # Create numerical features
            self.update_task_progress(0.6)
            numerical_features = await self._create_numerical_features()
            
            # Create interaction features
            self.update_task_progress(0.8)
            interaction_features = await self._create_interaction_features()
            
            # Combine all features
            self.update_task_progress(0.9)
            final_data = pd.concat([
                self.processed_data,
                temporal_features,
                categorical_features,
                numerical_features,
                interaction_features
            ], axis=1)
            
            # Update processed data with engineered features
            self.processed_data = final_data
            self.update_task_progress(1.0)
            
            # Store feature metadata
            self.feature_metadata = {
                "temporal_features": list(temporal_features.columns),
                "categorical_features": list(categorical_features.columns),
                "numerical_features": list(numerical_features.columns),
                "interaction_features": list(interaction_features.columns),
                "total_features": len(final_data.columns)
            }
            
            self.logger.info(f"Feature engineering completed: {len(final_data.columns)} features created")
            
            return {
                "status": "success",
                "features_created": len(final_data.columns),
                "feature_metadata": self.feature_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _create_temporal_features(self) -> pd.DataFrame:
        """Create temporal features from date columns"""
        try:
            temporal_features = pd.DataFrame()
            
            # Find date columns
            date_columns = self.processed_data.select_dtypes(include=['datetime64']).columns
            
            for col in date_columns:
                # Extract year, month, day, day of week
                temporal_features[f"{col}_year"] = self.processed_data[col].dt.year
                temporal_features[f"{col}_month"] = self.processed_data[col].dt.month
                temporal_features[f"{col}_day"] = self.processed_data[col].dt.day
                temporal_features[f"{col}_dayofweek"] = self.processed_data[col].dt.dayofweek
                temporal_features[f"{col}_quarter"] = self.processed_data[col].dt.quarter
                
                # Calculate tenure if it's a hire date
                if "hire" in col.lower() or "start" in col.lower():
                    current_date = pd.Timestamp.now()
                    temporal_features[f"{col}_tenure_days"] = (current_date - self.processed_data[col]).dt.days
                    temporal_features[f"{col}_tenure_years"] = temporal_features[f"{col}_tenure_days"] / 365.25
            
            return temporal_features
            
        except Exception as e:
            self.logger.error(f"Temporal feature creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_categorical_features(self) -> pd.DataFrame:
        """Create categorical features"""
        try:
            categorical_features = pd.DataFrame()
            
            # Find categorical columns
            categorical_columns = self.processed_data.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_columns:
                # Create dummy variables for categorical columns
                dummies = pd.get_dummies(self.processed_data[col], prefix=col, drop_first=True)
                categorical_features = pd.concat([categorical_features, dummies], axis=1)
            
            return categorical_features
            
        except Exception as e:
            self.logger.error(f"Categorical feature creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_numerical_features(self) -> pd.DataFrame:
        """Create numerical features"""
        try:
            numerical_features = pd.DataFrame()
            
            # Find numerical columns
            numerical_columns = self.processed_data.select_dtypes(include=['int64', 'float64']).columns
            
            for col in numerical_columns:
                # Create log transformation for positive values
                if (self.processed_data[col] > 0).all():
                    numerical_features[f"{col}_log"] = np.log(self.processed_data[col])
                
                # Create square transformation
                numerical_features[f"{col}_squared"] = self.processed_data[col] ** 2
                
                # Create square root transformation for positive values
                if (self.processed_data[col] >= 0).all():
                    numerical_features[f"{col}_sqrt"] = np.sqrt(self.processed_data[col])
            
            return numerical_features
            
        except Exception as e:
            self.logger.error(f"Numerical feature creation failed: {e}")
            return pd.DataFrame()
    
    async def _create_interaction_features(self) -> pd.DataFrame:
        """Create interaction features between numerical columns"""
        try:
            interaction_features = pd.DataFrame()
            
            # Find numerical columns
            numerical_columns = self.processed_data.select_dtypes(include=['int64', 'float64']).columns
            
            # Create pairwise interactions for top features
            top_features = numerical_columns[:5]  # Limit to prevent explosion
            
            for i, col1 in enumerate(top_features):
                for col2 in top_features[i+1:]:
                    interaction_features[f"{col1}_x_{col2}"] = (
                        self.processed_data[col1] * self.processed_data[col2]
                    )
            
            return interaction_features
            
        except Exception as e:
            self.logger.error(f"Interaction feature creation failed: {e}")
            return pd.DataFrame()
    
    async def _validate_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and create quality report"""
        try:
            if self.processed_data is None:
                raise ValueError("No processed data available for validation")
            
            self.state.current_task = "data_validation"
            self.update_task_progress(0.1)
            
            # Create quality report
            quality_report = DataQualityReport(
                timestamp=datetime.utcnow(),
                total_records=len(self.processed_data),
                total_features=len(self.processed_data.columns),
                missing_values_percentage=self.processed_data.isnull().sum().sum() / self.processed_data.size * 100,
                duplicate_records=len(self.processed_data[self.processed_data.duplicated()]),
                data_types=self.processed_data.dtypes.to_dict(),
                quality_score=0.0  # Will be calculated
            )
            
            # Calculate quality score
            self.update_task_progress(0.5)
            quality_score = await self._calculate_quality_score()
            quality_report.quality_score = quality_score
            
            # Store quality report
            self.quality_reports.append(quality_report)
            
            self.update_task_progress(1.0)
            
            return {
                "status": "success",
                "quality_report": quality_report.dict(),
                "recommendations": await self._generate_quality_recommendations(quality_report)
            }
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    async def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score"""
        try:
            score = 100.0
            
            # Deduct points for missing values
            missing_pct = self.processed_data.isnull().sum().sum() / self.processed_data.size * 100
            score -= missing_pct * 0.5
            
            # Deduct points for duplicates
            duplicate_pct = len(self.processed_data[self.processed_data.duplicated()]) / len(self.processed_data) * 100
            score -= duplicate_pct * 0.3
            
            # Deduct points for data type inconsistencies
            # This is a simplified version
            score -= 5.0  # Placeholder
            
            return max(0.0, score)
            
        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    async def _generate_quality_recommendations(self, quality_report: DataQualityReport) -> List[str]:
        """Generate recommendations based on quality report"""
        recommendations = []
        
        if quality_report.missing_values_percentage > 10:
            recommendations.append("High percentage of missing values detected. Consider imputation strategies.")
        
        if quality_report.duplicate_records > 0:
            recommendations.append("Duplicate records found. Consider deduplication.")
        
        if quality_report.quality_score < 70:
            recommendations.append("Overall data quality is below recommended threshold. Review data sources and preprocessing.")
        
        if not recommendations:
            recommendations.append("Data quality is good. No immediate actions required.")
        
        return recommendations
    
    async def _export_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Export processed data"""
        try:
            if self.processed_data is None:
                raise ValueError("No processed data available for export")
            
            self.state.current_task = "data_export"
            self.update_task_progress(0.1)
            
            export_format = config.get("format", "csv")
            export_path = config.get("path", f"./exports/attrition_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Create export directory
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            if export_format == "csv":
                export_file = f"{export_path}.csv"
                self.processed_data.to_csv(export_file, index=False)
            elif export_format == "parquet":
                export_file = f"{export_path}.parquet"
                self.processed_data.to_parquet(export_file, index=False)
            elif export_format == "excel":
                export_file = f"{export_path}.xlsx"
                self.processed_data.to_excel(export_file, index=False)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            self.update_task_progress(1.0)
            
            self.logger.info(f"Data exported to {export_file}")
            
            return {
                "status": "success",
                "export_file": export_file,
                "export_format": export_format,
                "records_exported": len(self.processed_data)
            }
            
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            raise
        finally:
            self.state.current_task = None
    
    def _get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data"""
        summary = {
            "raw_data_available": self.raw_data is not None,
            "processed_data_available": self.processed_data is not None,
            "data_sources": self.data_sources,
            "quality_reports_count": len(self.quality_reports)
        }
        
        if self.raw_data is not None:
            summary["raw_data_shape"] = self.raw_data.shape
            summary["raw_data_columns"] = list(self.raw_data.columns)
        
        if self.processed_data is not None:
            summary["processed_data_shape"] = self.processed_data.shape
            summary["processed_data_columns"] = list(self.processed_data.columns)
        
        if self.feature_metadata:
            summary["feature_metadata"] = self.feature_metadata
        
        return summary
