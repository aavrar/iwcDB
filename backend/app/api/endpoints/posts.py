from fastapi import APIRouter, Query, HTTPException, Depends, Request
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc
from datetime import datetime, timedelta
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.database import get_db
from app.core.logging import logger
from app.models.tweet import PostModel

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.get("/posts/top")
@limiter.limit("60/minute")
async def get_top_posts(
    request: Request,
    type: str = Query(..., regex="^(positive|negative)$", description="Type of posts to retrieve"),
    limit: int = Query(3, ge=1, le=10, description="Number of posts to return"),
    period: str = Query("30d", regex="^(30d|90d|6m|1y)$", description="Time period"),
    query: str = Query(None, description="Optional query filter"),
    db: Session = Depends(get_db)
):
    """Get top positive or negative posts."""
    try:
        # Parse period
        period_days = {
            "30d": 30,
            "90d": 90,
            "6m": 180,
            "1y": 365
        }
        days = period_days.get(period, 30)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Build query
        posts_query = db.query(PostModel).filter(
            PostModel.datetime >= start_date
        )
        
        # Apply query filter if provided
        if query:
            posts_query = posts_query.filter(
                PostModel.query.ilike(f"%{query}%")
            )
        
        # Filter by sentiment type and sort
        if type == "positive":
            posts_query = posts_query.filter(
                PostModel.sentiment_score > 0.1
            ).order_by(desc(PostModel.sentiment_score))
        else:  # negative
            posts_query = posts_query.filter(
                PostModel.sentiment_score < -0.1
            ).order_by(asc(PostModel.sentiment_score))  # Most negative first
        
        posts = posts_query.limit(limit).all()
        
        # Format response
        formatted_posts = []
        for post in posts:
            # Try to extract extra_data for additional info
            try:
                import json
                extra_data = json.loads(post.extra_data) if post.extra_data else {}
            except:
                extra_data = {}
            
            formatted_posts.append({
                "id": post.id,
                "content": post.content,
                "title": "",  # Reddit posts typically don't have separate titles
                "score": extra_data.get("score", 0),
                "url": extra_data.get("url", ""),
                "source": "reddit",
                "subreddit": extra_data.get("subreddit", ""),
                "created_at": post.datetime.isoformat(),
                "author": post.username,
                "sentiment_score": round(post.sentiment_score, 3),
                "image_url": extra_data.get("image_url"),
                "video_url": extra_data.get("video_url")
            })
        
        return formatted_posts
        
    except Exception as e:
        logger.error(f"Top posts endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving top posts"
        )


@router.get("/posts/recent")
@limiter.limit("60/minute") 
async def get_recent_posts(
    request: Request,
    limit: int = Query(20, ge=1, le=50, description="Number of recent posts to return"),
    query: str = Query(None, description="Optional query filter"),
    db: Session = Depends(get_db)
):
    """Get recent posts, optionally filtered by query."""
    try:
        # Build query
        posts_query = db.query(PostModel)
        
        # Apply query filter if provided
        if query:
            posts_query = posts_query.filter(
                PostModel.query.ilike(f"%{query}%")
            )
        
        # Order by most recent and limit
        posts = posts_query.order_by(desc(PostModel.datetime)).limit(limit).all()
        
        # Format response
        formatted_posts = []
        for post in posts:
            try:
                import json
                extra_data = json.loads(post.extra_data) if post.extra_data else {}
            except:
                extra_data = {}
            
            formatted_posts.append({
                "id": post.id,
                "content": post.content,
                "title": "",
                "score": extra_data.get("score", 0),
                "url": extra_data.get("url", ""),
                "source": "reddit",
                "subreddit": extra_data.get("subreddit", ""),
                "created_at": post.datetime.isoformat(),
                "author": post.username,
                "sentiment_score": round(post.sentiment_score, 3),
                "query": post.query,
                "image_url": extra_data.get("image_url"),
                "video_url": extra_data.get("video_url")
            })
        
        return formatted_posts
        
    except Exception as e:
        logger.error(f"Recent posts endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving recent posts"
        )


@router.get("/posts/stats")
@limiter.limit("30/minute")
async def get_posts_stats(
    request: Request,
    period: str = Query("30d", regex="^(30d|90d|6m|1y)$", description="Time period"),
    query: str = Query(None, description="Optional query filter"),
    db: Session = Depends(get_db)
):
    """Get statistical breakdown of posts for a period."""
    try:
        # Parse period
        period_days = {
            "30d": 30,
            "90d": 90, 
            "6m": 180,
            "1y": 365
        }
        days = period_days.get(period, 30)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Build query
        posts_query = db.query(PostModel).filter(
            PostModel.datetime >= start_date
        )
        
        # Apply query filter if provided
        if query:
            posts_query = posts_query.filter(
                PostModel.query.ilike(f"%{query}%")
            )
        
        posts = posts_query.all()
        
        if not posts:
            return {
                "total_posts": 0,
                "positive_posts": 0,
                "negative_posts": 0,
                "neutral_posts": 0,
                "average_sentiment": 0.0,
                "period": period,
                "query": query
            }
        
        # Calculate stats
        total_posts = len(posts)
        positive_posts = len([p for p in posts if p.sentiment_score > 0.1])
        negative_posts = len([p for p in posts if p.sentiment_score < -0.1])
        neutral_posts = total_posts - positive_posts - negative_posts
        
        average_sentiment = sum(p.sentiment_score for p in posts) / total_posts
        
        return {
            "total_posts": total_posts,
            "positive_posts": positive_posts,
            "negative_posts": negative_posts,
            "neutral_posts": neutral_posts,
            "positive_percentage": round((positive_posts / total_posts) * 100, 1),
            "negative_percentage": round((negative_posts / total_posts) * 100, 1),
            "neutral_percentage": round((neutral_posts / total_posts) * 100, 1),
            "average_sentiment": round(average_sentiment, 3),
            "period": period,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Posts stats endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving posts statistics"
        )