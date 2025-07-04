from fastapi import APIRouter, Query, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.cache import cache_manager
from app.core.logging import logger
from app.models.tweet import (
    PostResponse, PostModel, QueryModel, TimelineModel,
    SentimentScoreResponse, TimelinePoint, TopPostsResponse
)

router = APIRouter()


def calculate_sentiment_distribution(tweets: List[PostModel]) -> dict:
    """Calculate sentiment distribution from tweets."""
    if not tweets:
        return {"positive": 0, "negative": 0, "neutral": 0}
    
    distribution = {"positive": 0, "negative": 0, "neutral": 0}
    
    for tweet in tweets:
        if tweet.sentiment_score > 0.1:
            distribution["positive"] += 1
        elif tweet.sentiment_score < -0.1:
            distribution["negative"] += 1
        else:
            distribution["neutral"] += 1
    
    return distribution


@router.get("/sentiment/score", response_model=SentimentScoreResponse)
async def get_sentiment_score(
    background_tasks: BackgroundTasks,
    query: str = Query(..., min_length=1, description="Query to get sentiment score for"),
    use_cache: bool = Query(True, description="Whether to use cached results"),
    db: Session = Depends(get_db)
):
    """
    Get aggregate sentiment score for a query.
    
    - **query**: Search term to analyze sentiment for
    - **use_cache**: Whether to use cached sentiment data
    """
    try:
        # Check cache first
        if use_cache:
            cached_sentiment = await cache_manager.get_sentiment_for_query(query)
            if cached_sentiment:
                logger.info(f"Returning cached sentiment for query: {query}")
                return SentimentScoreResponse(**cached_sentiment)
        
        # Get tweets from database
        tweets = db.query(PostModel).filter(PostModel.query == query).all()
        
        if not tweets:
            raise HTTPException(
                status_code=404,
                detail=f"No sentiment data found for query: {query}. Please search for tweets first."
            )
        
        # Calculate sentiment statistics
        sentiment_scores = [tweet.sentiment_score for tweet in tweets]
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        
        distribution = calculate_sentiment_distribution(tweets)
        
        sentiment_response = SentimentScoreResponse(
            query=query,
            score=avg_score,
            positive_count=distribution["positive"],
            negative_count=distribution["negative"],
            neutral_count=distribution["neutral"],
            total_tweets=len(tweets)
        )
        
        # Cache the result
        background_tasks.add_task(
            cache_manager.cache_sentiment_for_query,
            query,
            sentiment_response.dict()
        )
        
        logger.info(f"Calculated sentiment score for query: {query} = {avg_score:.3f}")
        return sentiment_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment score endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while calculating sentiment"
        )


@router.get("/sentiment/timeline", response_model=List[TimelinePoint])
async def get_sentiment_timeline(
    background_tasks: BackgroundTasks,
    query: str = Query(..., min_length=1, description="Query to get timeline for"),
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    use_cache: bool = Query(True, description="Whether to use cached timeline data"),
    db: Session = Depends(get_db)
):
    """
    Get sentiment timeline for a query over time.
    
    - **query**: Search term to get timeline for
    - **hours**: Number of hours to look back (1-168)
    - **use_cache**: Whether to use cached timeline data
    """
    try:
        # Check cache first
        if use_cache:
            cached_timeline = await cache_manager.get_timeline_for_query(query)
            if cached_timeline:
                logger.info(f"Returning cached timeline for query: {query}")
                return [TimelinePoint(**point) for point in cached_timeline['timeline']]
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get tweets from database within time range
        tweets = db.query(PostModel).filter(
            PostModel.query == query,
            PostModel.datetime >= start_time,
            PostModel.datetime <= end_time
        ).order_by(PostModel.datetime).all()
        
        if not tweets:
            raise HTTPException(
                status_code=404,
                detail=f"No timeline data found for query: {query} in the last {hours} hours"
            )
        
        # Group tweets by hour and calculate sentiment
        timeline_points = []
        current_hour = start_time.replace(minute=0, second=0, microsecond=0)
        
        while current_hour <= end_time:
            next_hour = current_hour + timedelta(hours=1)
            
            # Get tweets for this hour
            hour_tweets = [
                tweet for tweet in tweets
                if current_hour <= tweet.datetime < next_hour
            ]
            
            if hour_tweets:
                avg_sentiment = sum(tweet.sentiment_score for tweet in hour_tweets) / len(hour_tweets)
                timeline_points.append(TimelinePoint(
                    timestamp=current_hour,
                    avg_sentiment=avg_sentiment,
                    tweet_count=len(hour_tweets)
                ))
            
            current_hour = next_hour
        
        # Cache the timeline data
        if timeline_points:
            background_tasks.add_task(
                cache_manager.cache_timeline_for_query,
                query,
                [point.dict() for point in timeline_points]
            )
        
        logger.info(f"Generated timeline for query: {query} with {len(timeline_points)} points")
        return timeline_points
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Timeline endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while generating timeline"
        )


@router.get("/sentiment/top-tweets", response_model=TopPostsResponse)
async def get_top_tweets(
    query: str = Query(..., min_length=1, description="Query to get top tweets for"),
    count: int = Query(5, ge=1, le=10, description="Number of top tweets per category"),
    db: Session = Depends(get_db)
):
    """
    Get top positive and negative tweets for a query.
    
    - **query**: Search term to get top tweets for
    - **count**: Number of tweets per category (1-10)
    """
    try:
        # Get tweets from database
        tweets = db.query(PostModel).filter(PostModel.query == query).all()
        
        if not tweets:
            raise HTTPException(
                status_code=404,
                detail=f"No tweets found for query: {query}. Please search for tweets first."
            )
        
        # Sort tweets by sentiment score
        sorted_tweets = sorted(tweets, key=lambda x: x.sentiment_score)
        
        # Get top negative tweets (lowest sentiment scores)
        top_negative = sorted_tweets[:count]
        top_negative_responses = [
            PostResponse(
                id=tweet.id,
                content=tweet.content,
                username=tweet.username,
                datetime=tweet.datetime,
                sentiment_score=tweet.sentiment_score,
                query=tweet.query
            ) for tweet in top_negative
        ]
        
        # Get top positive tweets (highest sentiment scores)
        top_positive = sorted_tweets[-count:][::-1]  # Reverse to get highest first
        top_positive_responses = [
            PostResponse(
                id=tweet.id,
                content=tweet.content,
                username=tweet.username,
                datetime=tweet.datetime,
                sentiment_score=tweet.sentiment_score,
                query=tweet.query
            ) for tweet in top_positive
        ]
        
        response = TopPostsResponse(
            top_positive=top_positive_responses,
            top_negative=top_negative_responses
        )
        
        logger.info(f"Retrieved top tweets for query: {query}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Top tweets endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while retrieving top tweets"
        )


@router.get("/sentiment/stats", response_model=dict)
async def get_sentiment_stats(
    query: Optional[str] = Query(None, description="Specific query to get stats for"),
    db: Session = Depends(get_db)
):
    """
    Get general sentiment statistics.
    
    - **query**: Optional specific query to get stats for
    """
    try:
        if query:
            # Stats for specific query
            tweets = db.query(PostModel).filter(PostModel.query == query).all()
            if not tweets:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for query: {query}"
                )
            
            sentiment_scores = [tweet.sentiment_score for tweet in tweets]
            distribution = calculate_sentiment_distribution(tweets)
            
            return {
                "query": query,
                "total_tweets": len(tweets),
                "avg_sentiment": sum(sentiment_scores) / len(sentiment_scores),
                "min_sentiment": min(sentiment_scores),
                "max_sentiment": max(sentiment_scores),
                "distribution": distribution,
                "unique_users": len(set(tweet.username for tweet in tweets))
            }
        else:
            # General stats across all queries
            total_tweets = db.query(func.count(PostModel.id)).scalar()
            total_queries = db.query(func.count(QueryModel.id)).scalar()
            avg_sentiment = db.query(func.avg(PostModel.sentiment_score)).scalar()
            
            # Most active queries
            top_queries = db.query(
                QueryModel.query_text,
                QueryModel.tweet_count,
                QueryModel.avg_sentiment
            ).order_by(desc(QueryModel.tweet_count)).limit(5).all()
            
            return {
                "total_tweets": total_tweets or 0,
                "total_queries": total_queries or 0,
                "avg_sentiment": float(avg_sentiment) if avg_sentiment else 0.0,
                "top_queries": [
                    {
                        "query": q.query_text,
                        "tweet_count": q.tweet_count,
                        "avg_sentiment": q.avg_sentiment
                    } for q in top_queries
                ]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment stats endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while retrieving stats"
        )