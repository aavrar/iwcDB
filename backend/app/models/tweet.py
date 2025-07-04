from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class PostModel(Base):
    __tablename__ = "posts"
    
    id = Column(String, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    username = Column(String, nullable=False, index=True)
    datetime = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False, index=True)
    query = Column(String, nullable=False, index=True)
    extra_data = Column(Text, nullable=True)  # JSON string for additional data
    created_at = Column(DateTime, default=func.now())
    
    # Remove problematic relationship for now
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_query_datetime', 'query', 'datetime'),
        Index('idx_query_sentiment', 'query', 'sentiment_score'),
    )


class QueryModel(Base):
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=func.now())
    last_updated = Column(DateTime, default=func.now())
    post_count = Column(Integer, default=0)
    avg_sentiment = Column(Float, default=0.0)
    
    # Image caching fields
    image_url = Column(String, nullable=True)
    image_cached_at = Column(DateTime, nullable=True)
    image_source = Column(String, nullable=True)  # 'wikipedia', 'cagematch', 'placeholder'
    
    # Remove problematic relationships for now


class TimelineModel(Base):
    __tablename__ = "timeline"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    avg_sentiment = Column(Float, nullable=False)
    post_count = Column(Integer, nullable=False)
    
    # Remove problematic relationship for now
    
    # Composite index for timeline queries
    __table_args__ = (
        Index('idx_query_timeline', 'query_id', 'timestamp'),
    )


class AutocompleteModel(Base):
    __tablename__ = "autocomplete"
    
    id = Column(Integer, primary_key=True, index=True)
    term = Column(String, nullable=False, unique=True, index=True)
    category = Column(String, nullable=False, index=True)  # wrestler, event, promotion
    popularity_score = Column(Float, default=0.0, index=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now())


# Pydantic models for API responses
class PostResponse(BaseModel):
    id: str
    content: str
    username: str
    datetime: datetime
    sentiment_score: float
    query: str
    
    class Config:
        from_attributes = True


class SentimentScoreResponse(BaseModel):
    query: str
    score: float
    positive_count: int
    negative_count: int
    neutral_count: int
    total_posts: int


class TimelinePoint(BaseModel):
    timestamp: datetime
    avg_sentiment: float
    post_count: int
    
    class Config:
        from_attributes = True


class TopPostsResponse(BaseModel):
    top_positive: list[PostResponse]
    top_negative: list[PostResponse]


class QueryResponse(BaseModel):
    id: int
    query_text: str
    created_at: datetime
    last_updated: datetime
    post_count: int
    avg_sentiment: float
    
    class Config:
        from_attributes = True


class AutocompleteResponse(BaseModel):
    term: str
    category: str
    popularity_score: float
    
    class Config:
        from_attributes = True