from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import redis
from typing import Generator
import os


# SQLAlchemy setup
def get_database_url():
    """Get database URL based on environment."""
    # Priority: Neon > Turso > SQLite
    if settings.NEON_DATABASE_URL:
        # Use Neon PostgreSQL in production (recommended)
        return settings.NEON_DATABASE_URL
    elif settings.TURSO_DATABASE_URL and settings.TURSO_AUTH_TOKEN:
        # Use Turso SQLite as alternative
        return f"{settings.TURSO_DATABASE_URL}?authToken={settings.TURSO_AUTH_TOKEN}"
    else:
        # Use SQLite locally
        return settings.DATABASE_URL


def get_engine_config(database_url: str):
    """Get engine configuration based on database type."""
    config = {
        "pool_pre_ping": True,
        "echo": settings.DEBUG,
    }
    
    if "postgresql" in database_url or "postgres" in database_url:
        # PostgreSQL configuration (Neon)
        config.update({
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": settings.DB_POOL_TIMEOUT,
        })
    elif "sqlite" in database_url:
        # SQLite configuration
        config["connect_args"] = {"check_same_thread": False}
    
    return config

database_url = get_database_url()
engine = create_engine(database_url, **get_engine_config(database_url))

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis client
try:
    redis_client = redis.from_url(
        settings.REDIS_URL,
        password=settings.REDIS_PASSWORD,
        decode_responses=True
    )
    # Test Redis connection
    redis_client.ping()
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None


def get_db() -> Generator:
    """Database session generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    return redis_client


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)