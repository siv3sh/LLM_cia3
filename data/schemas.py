"""
Data schemas for the Multi-Agent Attrition Analysis System
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class DataSourceType(str, Enum):
    """Data source types"""
    CSV = "csv"
    DATABASE = "database"
    API = "api"
    EXCEL = "excel"
    JSON = "json"


class DataQualityLevel(str, Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class AttritionStatus(str, Enum):
    """Attrition status values"""
    YES = "Yes"
    NO = "No"
    TRUE = "True"
    FALSE = "False"
    LEFT = "Left"
    STAYED = "Stayed"


class DataQualityReport(BaseModel):
    """Data quality report schema"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_records: int = Field(..., description="Total number of records")
    total_features: int = Field(..., description="Total number of features")
    missing_values_percentage: float = Field(..., description="Percentage of missing values")
    duplicate_records: int = Field(..., description="Number of duplicate records")
    data_types: Dict[str, str] = Field(..., description="Data types of each column")
    quality_score: float = Field(..., description="Overall quality score (0-100)")
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Quality score must be between 0 and 100')
        return v
    
    @validator('missing_values_percentage')
    def validate_missing_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Missing values percentage must be between 0 and 100')
        return v


class AttritionDataSchema(BaseModel):
    """Schema for attrition data"""
    employee_id: Optional[str] = Field(None, description="Employee identifier")
    age: Optional[int] = Field(None, ge=18, le=100, description="Employee age")
    gender: Optional[str] = Field(None, description="Employee gender")
    education: Optional[str] = Field(None, description="Education level")
    marital_status: Optional[str] = Field(None, description="Marital status")
    job_role: Optional[str] = Field(None, description="Job role/title")
    department: Optional[str] = Field(None, description="Department")
    business_travel: Optional[str] = Field(None, description="Business travel frequency")
    daily_rate: Optional[float] = Field(None, description="Daily rate")
    hourly_rate: Optional[float] = Field(None, description="Hourly rate")
    monthly_rate: Optional[float] = Field(None, description="Monthly rate")
    monthly_income: Optional[float] = Field(None, description="Monthly income")
    overtime: Optional[str] = Field(None, description="Overtime status")
    percent_salary_hike: Optional[float] = Field(None, description="Percentage salary hike")
    performance_rating: Optional[int] = Field(None, ge=1, le=5, description="Performance rating")
    relationship_satisfaction: Optional[int] = Field(None, ge=1, le=5, description="Relationship satisfaction")
    stock_option_level: Optional[int] = Field(None, ge=0, description="Stock option level")
    total_working_years: Optional[float] = Field(None, description="Total working years")
    training_times_last_year: Optional[int] = Field(None, description="Training times last year")
    years_at_company: Optional[float] = Field(None, description="Years at company")
    years_in_current_role: Optional[float] = Field(None, description="Years in current role")
    years_since_last_promotion: Optional[float] = Field(None, description="Years since last promotion")
    years_with_curr_manager: Optional[float] = Field(None, description="Years with current manager")
    attrition: Optional[str] = Field(None, description="Attrition status")
    
    class Config:
        extra = "allow"  # Allow additional fields


class DataCollectionConfig(BaseModel):
    """Configuration for data collection"""
    sources: List[DataSourceType] = Field(default=[DataSourceType.CSV], description="Data sources to use")
    csv_paths: Optional[List[str]] = Field(None, description="Paths to CSV files")
    database_query: Optional[str] = Field(None, description="Database query for data extraction")
    api_endpoints: Optional[List[str]] = Field(None, description="API endpoints for data collection")
    batch_size: Optional[int] = Field(1000, description="Batch size for data processing")
    timeout: Optional[int] = Field(300, description="Timeout for data collection operations")
    
    @validator('csv_paths')
    def validate_csv_paths(cls, v, values):
        if DataSourceType.CSV in values.get('sources', []) and not v:
            raise ValueError('CSV paths must be provided when CSV source is selected')
        return v


class DataPreprocessingConfig(BaseModel):
    """Configuration for data preprocessing"""
    handle_missing_values: bool = Field(True, description="Whether to handle missing values")
    missing_value_strategy: str = Field("mean", description="Strategy for handling missing values")
    remove_duplicates: bool = Field(True, description="Whether to remove duplicate records")
    handle_outliers: bool = Field(True, description="Whether to handle outliers")
    outlier_method: str = Field("iqr", description="Method for outlier detection")
    outlier_threshold: float = Field(1.5, description="Threshold for outlier detection")
    normalize_numerical: bool = Field(False, description="Whether to normalize numerical features")
    encoding_method: str = Field("onehot", description="Method for categorical encoding")
    
    @validator('missing_value_strategy')
    def validate_missing_strategy(cls, v):
        valid_strategies = ["mean", "median", "mode", "drop", "interpolate"]
        if v not in valid_strategies:
            raise ValueError(f'Missing value strategy must be one of: {valid_strategies}')
        return v
    
    @validator('outlier_method')
    def validate_outlier_method(cls, v):
        valid_methods = ["iqr", "zscore", "isolation_forest"]
        if v not in valid_methods:
            raise ValueError(f'Outlier method must be one of: {valid_methods}')
        return v


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering"""
    create_temporal_features: bool = Field(True, description="Whether to create temporal features")
    create_interaction_features: bool = Field(True, description="Whether to create interaction features")
    create_polynomial_features: bool = Field(False, description="Whether to create polynomial features")
    polynomial_degree: int = Field(2, ge=1, le=5, description="Degree of polynomial features")
    feature_selection_method: str = Field("correlation", description="Method for feature selection")
    max_features: Optional[int] = Field(None, description="Maximum number of features to keep")
    correlation_threshold: float = Field(0.8, description="Correlation threshold for feature selection")
    
    @validator('polynomial_degree')
    def validate_polynomial_degree(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Polynomial degree must be between 1 and 5')
        return v
    
    @validator('correlation_threshold')
    def validate_correlation_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Correlation threshold must be between 0 and 1')
        return v


class AnalysisConfig(BaseModel):
    """Configuration for statistical analysis"""
    analysis_type: str = Field("comprehensive", description="Type of analysis to perform")
    descriptive_statistics: bool = Field(True, description="Whether to compute descriptive statistics")
    correlation_analysis: bool = Field(True, description="Whether to perform correlation analysis")
    hypothesis_testing: bool = Field(True, description="Whether to perform hypothesis testing")
    feature_importance: bool = Field(True, description="Whether to compute feature importance")
    pca_analysis: bool = Field(False, description="Whether to perform PCA analysis")
    clustering_analysis: bool = Field(False, description="Whether to perform clustering analysis")
    confidence_level: float = Field(0.95, description="Confidence level for statistical tests")
    
    @validator('confidence_level')
    def validate_confidence_level(cls, v):
        if not 0.5 <= v <= 0.99:
            raise ValueError('Confidence level must be between 0.5 and 0.99')
        return v


class ModelConfig(BaseModel):
    """Configuration for machine learning models"""
    model_type: str = Field("random_forest", description="Type of model to use")
    test_size: float = Field(0.2, description="Proportion of data for testing")
    random_state: int = Field(42, description="Random state for reproducibility")
    cross_validation_folds: int = Field(5, description="Number of cross-validation folds")
    hyperparameter_tuning: bool = Field(True, description="Whether to perform hyperparameter tuning")
    ensemble_methods: bool = Field(False, description="Whether to use ensemble methods")
    feature_scaling: bool = Field(True, description="Whether to scale features")
    
    @validator('test_size')
    def validate_test_size(cls, v):
        if not 0.1 <= v <= 0.5:
            raise ValueError('Test size must be between 0.1 and 0.5')
        return v
    
    @validator('cross_validation_folds')
    def validate_cv_folds(cls, v):
        if not 3 <= v <= 10:
            raise ValueError('Cross-validation folds must be between 3 and 10')
        return v


class PredictionConfig(BaseModel):
    """Configuration for predictions"""
    model_id: str = Field(..., description="ID of the trained model to use")
    prediction_data: Optional[Dict[str, Any]] = Field(None, description="Data for making predictions")
    return_probabilities: bool = Field(True, description="Whether to return prediction probabilities")
    threshold: float = Field(0.5, description="Threshold for binary classification")
    batch_size: Optional[int] = Field(None, description="Batch size for predictions")
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Threshold must be between 0 and 1')
        return v


class InsightConfig(BaseModel):
    """Configuration for insight generation"""
    insight_type: str = Field("comprehensive", description="Type of insights to generate")
    risk_assessment: bool = Field(True, description="Whether to perform risk assessment")
    trend_analysis: bool = Field(True, description="Whether to perform trend analysis")
    cost_benefit_analysis: bool = Field(False, description="Whether to perform cost-benefit analysis")
    actionable_recommendations: bool = Field(True, description="Whether to generate actionable recommendations")
    visualization: bool = Field(True, description="Whether to create visualizations")
    report_format: str = Field("json", description="Format for insight reports")
    
    @validator('report_format')
    def validate_report_format(cls, v):
        valid_formats = ["json", "html", "pdf", "markdown"]
        if v not in valid_formats:
            raise ValueError(f'Report format must be one of: {valid_formats}')
        return v


class WorkflowConfig(BaseModel):
    """Configuration for workflow execution"""
    workflow_type: str = Field("comprehensive", description="Type of workflow to execute")
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    data_preprocessing: DataPreprocessingConfig = Field(default_factory=DataPreprocessingConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    modeling: ModelConfig = Field(default_factory=ModelConfig)
    prediction: Optional[PredictionConfig] = Field(None)
    insights: InsightConfig = Field(default_factory=InsightConfig)
    parallel_execution: bool = Field(False, description="Whether to execute steps in parallel")
    max_retries: int = Field(3, description="Maximum number of retries for failed steps")
    
    @validator('workflow_type')
    def validate_workflow_type(cls, v):
        valid_types = ["basic", "comprehensive", "predictive"]
        if v not in valid_types:
            raise ValueError(f'Workflow type must be one of: {valid_types}')
        return v


class AnalysisResult(BaseModel):
    """Schema for analysis results"""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    workflow_config: WorkflowConfig = Field(..., description="Configuration used for the analysis")
    data_quality_report: Optional[DataQualityReport] = Field(None, description="Data quality assessment")
    descriptive_statistics: Optional[Dict[str, Any]] = Field(None, description="Descriptive statistics")
    correlation_analysis: Optional[Dict[str, Any]] = Field(None, description="Correlation analysis results")
    hypothesis_testing: Optional[Dict[str, Any]] = Field(None, description="Hypothesis testing results")
    feature_importance: Optional[Dict[str, Any]] = Field(None, description="Feature importance analysis")
    model_performance: Optional[Dict[str, Any]] = Field(None, description="Model performance metrics")
    predictions: Optional[Dict[str, Any]] = Field(None, description="Prediction results")
    business_insights: Optional[Dict[str, Any]] = Field(None, description="Business insights generated")
    recommendations: Optional[List[str]] = Field(None, description="Actionable recommendations")
    execution_time: Optional[float] = Field(None, description="Total execution time in seconds")
    status: str = Field("completed", description="Status of the analysis")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")


class JobRequest(BaseModel):
    """Schema for job requests"""
    job_id: str = Field(..., description="Unique identifier for the job")
    company_name: str = Field(..., description="Name of the company")
    analysis_type: str = Field("comprehensive", description="Type of analysis requested")
    workflow_config: WorkflowConfig = Field(..., description="Workflow configuration")
    priority: int = Field(1, ge=1, le=5, description="Job priority (1=low, 5=high)")
    requested_by: str = Field(..., description="User who requested the analysis")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class JobStatus(BaseModel):
    """Schema for job status"""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Job progress (0-1)")
    current_step: Optional[str] = Field(None, description="Current step being executed")
    start_time: Optional[datetime] = Field(None, description="Job start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class SystemHealth(BaseModel):
    """Schema for system health status"""
    overall_health: str = Field(..., description="Overall system health status")
    health_score: float = Field(..., ge=0.0, le=100.0, description="System health score")
    agent_status: Dict[str, Dict[str, Any]] = Field(..., description="Status of individual agents")
    system_metrics: Dict[str, Any] = Field(..., description="System performance metrics")
    active_jobs: int = Field(..., description="Number of active jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    last_health_check: datetime = Field(default_factory=datetime.utcnow)
    recommendations: Optional[List[str]] = Field(None, description="Health improvement recommendations")


class APIResponse(BaseModel):
    """Generic API response schema"""
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")


class ErrorResponse(BaseModel):
    """Error response schema"""
    status: str = Field("error", description="Response status")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

# Chat and Document Interaction Schemas
class ChatMessage(BaseModel):
    """Schema for chat messages"""
    session_id: str
    user_id: str
    message_id: str
    content: str
    timestamp: datetime
    message_type: str = "user"  # user, system, assistant
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Schema for chat responses"""
    session_id: str
    user_id: str
    message_id: str
    content: str
    source_documents: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class DocumentQuery(BaseModel):
    """Schema for document queries"""
    query: str
    max_results: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class ChatSession(BaseModel):
    """Schema for chat sessions"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    status: str = "active"  # active, inactive, closed
    metadata: Optional[Dict[str, Any]] = None
