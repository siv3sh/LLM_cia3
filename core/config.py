"""
Configuration management for the Multi-Agent Attrition Analysis System
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration using environment variables"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Groq API Configuration
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-70b-8192", env="GROQ_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", env="GROQ_BASE_URL")
    groq_timeout: int = Field(default=300, env="GROQ_TIMEOUT")
    groq_max_tokens: int = Field(default=4096, env="GROQ_MAX_TOKENS")
    groq_temperature: float = Field(default=0.1, env="GROQ_TEMPERATURE")
    
    # Database Configuration
    database_url: str = Field(default="postgresql://user:password@localhost:5432/attrition_db", env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Vector Store Configuration
    vector_store_path: str = Field(default="./vector_store", env="VECTOR_STORE_PATH")
    vector_store_type: str = Field(default="chroma", env="VECTOR_STORE_TYPE")
    vector_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="VECTOR_EMBEDDING_MODEL")
    vector_chunk_size: int = Field(default=1000, env="VECTOR_CHUNK_SIZE")
    vector_chunk_overlap: int = Field(default=200, env="VECTOR_CHUNK_OVERLAP")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_max_size: str = Field(default="100MB", env="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    api_debug: bool = Field(default=False, env="API_DEBUG")
    
    # Security Configuration
    secret_key: str = Field(default="your_secret_key_here_change_in_production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Agent Configuration
    agent_timeout: int = Field(default=300, env="AGENT_TIMEOUT")
    agent_max_retries: int = Field(default=3, env="AGENT_MAX_RETRIES")
    agent_concurrent_limit: int = Field(default=5, env="AGENT_CONCURRENT_LIMIT")
    agent_memory_size: int = Field(default=1000, env="AGENT_MEMORY_SIZE")
    
    # Model Configuration
    model_update_interval: int = Field(default=3600, env="MODEL_UPDATE_INTERVAL")
    model_backup_enabled: bool = Field(default=True, env="MODEL_BACKUP_ENABLED")
    model_backup_path: str = Field(default="./models/backup", env="MODEL_BACKUP_PATH")
    model_version: str = Field(default="1.0.0", env="MODEL_VERSION")
    
    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    performance_monitoring: bool = Field(default=True, env="PERFORMANCE_MONITORING")
    
    # Cache Configuration
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # External Services (Optional)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: Optional[str] = Field(default="gpt-4", env="OPENAI_MODEL")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    wandb_api_key: Optional[str] = Field(default=None, env="WANDB_API_KEY")
    wandb_project: Optional[str] = Field(default="attrition-analysis", env="WANDB_PROJECT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        if not self.groq_api_key or self.groq_api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY must be set to a valid API key")
        
        if self.groq_temperature < 0 or self.groq_temperature > 2:
            raise ValueError("GROQ_TEMPERATURE must be between 0 and 2")
        
        if self.api_port < 1 or self.api_port > 65535:
            raise ValueError("API_PORT must be between 1 and 65535")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment.lower() == "testing"
