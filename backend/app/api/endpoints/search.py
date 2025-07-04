from fastapi import APIRouter, Query, HTTPException, Depends, BackgroundTasks, Request
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.database import get_db
from app.core.scraper import RedditScraper
from app.core.nlp import analyze_texts_sentiment
from app.core.cache import cache_manager
from app.core.logging import logger
from app.core.security import sanitize_user_input, check_rate_limit
from app.models.tweet import (
    PostResponse, PostModel, QueryModel,
    SentimentScoreResponse, TimelinePoint, TopPostsResponse
)

limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


async def process_and_store_tweets(
    query: str, 
    raw_tweets: List[dict], 
    db: Session
) -> List[PostResponse]:
    """Process raw tweets, analyze sentiment, and store in database."""
    if not raw_tweets:
        return []
    
    # Extract text content for sentiment analysis
    texts = [tweet.get('content', '') for tweet in raw_tweets]
    
    # Analyze sentiment in batch
    sentiment_results = await analyze_texts_sentiment(texts)
    
    processed_tweets = []
    tweet_models = []
    
    for tweet_data, (sentiment_score, confidence) in zip(raw_tweets, sentiment_results):
        try:
            # Create tweet model
            tweet_model = PostModel(
                id=tweet_data.get('id', ''),
                content=tweet_data.get('content', ''),
                username=tweet_data.get('username', 'unknown'),
                datetime=datetime.fromisoformat(tweet_data.get('datetime', datetime.utcnow().isoformat())),
                sentiment_score=sentiment_score,
                query=query,
                extra_data=str({"confidence": confidence, "url": tweet_data.get('url', '')})
            )
            
            tweet_models.append(tweet_model)
            
            # Create response model
            tweet_response = PostResponse(
                id=tweet_model.id,
                content=tweet_model.content,
                username=tweet_model.username,
                datetime=tweet_model.datetime,
                sentiment_score=tweet_model.sentiment_score,
                query=tweet_model.query
            )
            
            processed_tweets.append(tweet_response)
            
        except Exception as e:
            logger.error(f"Error processing tweet {tweet_data.get('id', 'unknown')}: {e}")
            continue
    
    # Store in database
    try:
        # Check if query exists
        query_model = db.query(QueryModel).filter(QueryModel.query_text == query).first()
        if not query_model:
            query_model = QueryModel(
                query_text=query,
                tweet_count=len(tweet_models),
                avg_sentiment=sum(t.sentiment_score for t in tweet_models) / len(tweet_models) if tweet_models else 0.0
            )
            db.add(query_model)
        else:
            query_model.last_updated = datetime.utcnow()
            query_model.tweet_count += len(tweet_models)
            if tweet_models:
                query_model.avg_sentiment = (
                    query_model.avg_sentiment + 
                    sum(t.sentiment_score for t in tweet_models) / len(tweet_models)
                ) / 2
        
        # Add tweets to database
        for tweet_model in tweet_models:
            # Check if tweet already exists
            existing = db.query(PostModel).filter(PostModel.id == tweet_model.id).first()
            if not existing:
                db.add(tweet_model)
        
        db.commit()
        logger.info(f"Stored {len(tweet_models)} tweets for query: {query}")
        
    except Exception as e:
        logger.error(f"Database error storing tweets: {e}")
        db.rollback()
    
    return processed_tweets


@router.get("/posts", response_model=List[PostResponse])
@limiter.limit("10/minute")
async def search_tweets(
    request: Request,
    background_tasks: BackgroundTasks,
    q: str = Query(..., min_length=1, max_length=200, description="Search query for tweets"),
    max_results: int = Query(50, ge=1, le=200, description="Maximum number of tweets to return"),
    use_cache: bool = Query(True, description="Whether to use cached results"),
    db: Session = Depends(get_db)
):
    """
    Search and analyze tweets for a given query.
    
    - **query**: Search term (wrestler name, event, etc.)
    - **max_results**: Number of tweets to return (1-200)
    - **use_cache**: Whether to use cached results if available
    """
    try:
        # Sanitize query input
        query = sanitize_user_input(q)
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty after sanitization")
        
        # Increment query count for popularity tracking
        background_tasks.add_task(cache_manager.increment_query_count, query)
        
        # Check cache first if enabled
        if use_cache:
            cached_tweets = await cache_manager.get_tweets_for_query(query, max_results)
            if cached_tweets:
                logger.info(f"Returning cached results for query: {query}")
                return [PostResponse(**tweet) for tweet in cached_tweets['tweets']]
        
        # Scrape new tweets
        logger.info(f"Scraping tweets for query: {query}")
        scraper = RedditScraper()
        raw_tweets = await scraper.scrape_reddit_posts(query, max_results)
        
        if not raw_tweets:
            raise HTTPException(
                status_code=404, 
                detail=f"No tweets found for query: {query}"
            )
        
        # Process tweets and analyze sentiment
        processed_tweets = await process_and_store_tweets(query, raw_tweets, db)
        
        # Cache results
        if processed_tweets:
            background_tasks.add_task(
                cache_manager.cache_tweets_for_query,
                query,
                [tweet.dict() for tweet in processed_tweets],
                max_results
            )
        
        logger.info(f"Successfully processed {len(processed_tweets)} tweets for query: {query}")
        return processed_tweets
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error occurred while searching tweets"
        )


@router.get("/search/history", response_model=List[str])
async def get_search_history(
    limit: int = Query(10, ge=1, le=50, description="Number of recent queries to return"),
    db: Session = Depends(get_db)
):
    """
    Get recent search queries ordered by last updated time.
    """
    try:
        queries = db.query(QueryModel).order_by(QueryModel.last_updated.desc()).limit(limit).all()
        return [query.query_text for query in queries]
    
    except Exception as e:
        logger.error(f"Search history endpoint error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Error retrieving search history"
        )


@router.get("/search/popular", response_model=List[dict])
async def get_popular_queries(
    limit: int = Query(10, ge=1, le=20, description="Number of popular queries to return")
):
    """
    Get popular queries based on usage frequency.
    """
    try:
        popular_queries = await cache_manager.get_popular_queries(limit)
        return popular_queries
    
    except Exception as e:
        logger.error(f"Popular queries endpoint error: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Error retrieving popular queries"
        )