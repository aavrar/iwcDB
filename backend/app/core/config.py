from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "IWC Sentiment Tracker"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./iwc_sentiment.db"
    
    # Production Database Options
    NEON_DATABASE_URL: Optional[str] = None  # Neon PostgreSQL (recommended)
    TURSO_DATABASE_URL: Optional[str] = None  # Turso SQLite (alternative)
    TURSO_AUTH_TOKEN: Optional[str] = None
    
    # Database Pool Settings (for PostgreSQL)
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_PASSWORD: Optional[str] = None
    
    # Scraping Configuration
    SCRAPING_DELAY_MIN: int = 1
    SCRAPING_DELAY_MAX: int = 3
    MAX_TWEETS_PER_QUERY: int = 100
    PLAYWRIGHT_TIMEOUT: int = 30000
    
    # NLP Configuration
    MODEL_NAME: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    MODEL_CACHE_DIR: str = "./models"
    BATCH_SIZE: int = 16
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Security
    SECRET_KEY: str
    ALLOWED_HOSTS: list = ["*"]
    CORS_ORIGINS: list = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    
    # Background Tasks
    CELERY_BROKER_URL: str = "redis://localhost:6379"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379"
    
    # Proxy Configuration (for scraping)
    USE_PROXY: bool = False
    PROXY_LIST: list = []
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()