from fastapi import APIRouter, Query, HTTPException, Depends, Request
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.database import get_db
from app.core.logging import logger
from app.models.tweet import PostModel

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.get("/timeline")
@limiter.limit("30/minute")
async def get_sentiment_timeline(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    period: str = Query("30d", regex="^(30d|90d|6m|1y)$", description="Time period"),
    db: Session = Depends(get_db)
):
    """Get sentiment timeline for a query over time."""
    try:
        # Parse period to determine grouping
        period_config = {
            "30d": {"days": 30, "group_by": "day"},
            "90d": {"days": 90, "group_by": "day"},
            "6m": {"days": 180, "group_by": "week"},
            "1y": {"days": 365, "group_by": "week"}
        }
        
        config = period_config.get(period, period_config["30d"])
        start_date = datetime.utcnow() - timedelta(days=config["days"])
        
        # Query posts for the time period
        posts = db.query(PostModel).filter(
            PostModel.query.ilike(f"%{q}%"),
            PostModel.datetime >= start_date
        ).order_by(PostModel.datetime).all()
        
        if not posts:
            return []
        
        # Group by time period
        timeline_data = {}
        
        for post in posts:
            if config["group_by"] == "day":
                # Group by date (YYYY-MM-DD)
                date_key = post.datetime.strftime("%Y-%m-%d")
            else:
                # Group by week (start of week)
                start_of_week = post.datetime - timedelta(days=post.datetime.weekday())
                date_key = start_of_week.strftime("%Y-%m-%d")
            
            if date_key not in timeline_data:
                timeline_data[date_key] = {
                    "date": date_key,
                    "sentiments": [],
                    "posts_count": 0
                }
            
            timeline_data[date_key]["sentiments"].append(post.sentiment_score)
            timeline_data[date_key]["posts_count"] += 1
        
        # Calculate average sentiment for each time period
        timeline_points = []
        for date_key, data in sorted(timeline_data.items()):
            avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
            timeline_points.append({
                "date": date_key,
                "sentiment": round(avg_sentiment, 3),
                "posts_count": data["posts_count"]
            })
        
        return timeline_points
        
    except Exception as e:
        logger.error(f"Timeline endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving timeline data"
        )


@router.get("/compare")
@limiter.limit("20/minute")
async def compare_wrestlers(
    request: Request,
    w1: str = Query(..., min_length=1, max_length=100, description="First wrestler name"),
    w2: str = Query(..., min_length=1, max_length=100, description="Second wrestler name"),
    period: str = Query("30d", regex="^(30d|90d|6m|1y)$", description="Time period"),
    db: Session = Depends(get_db)
):
    """Compare sentiment between two wrestlers."""
    try:
        period_days = {
            "30d": 30,
            "90d": 90,
            "6m": 180,
            "1y": 365
        }
        days = period_days.get(period, 30)
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get data for both wrestlers
        wrestler1_posts = db.query(PostModel).filter(
            PostModel.query.ilike(f"%{w1}%"),
            PostModel.datetime >= start_date
        ).all()
        
        wrestler2_posts = db.query(PostModel).filter(
            PostModel.query.ilike(f"%{w2}%"),
            PostModel.datetime >= start_date
        ).all()
        
        def calculate_wrestler_stats(posts, name):
            if not posts:
                return {
                    "name": name,
                    "total_posts": 0,
                    "positive_posts": 0,
                    "negative_posts": 0,
                    "neutral_posts": 0,
                    "average_sentiment": 0.0,
                    "popularity_score": 0.0,
                    "love_score": 0.0,
                    "hate_score": 0.0,
                    "recent_posts": []
                }
            
            total_posts = len(posts)
            positive_posts = len([p for p in posts if p.sentiment_score > 0.1])
            negative_posts = len([p for p in posts if p.sentiment_score < -0.1])
            neutral_posts = total_posts - positive_posts - negative_posts
            
            avg_sentiment = sum(p.sentiment_score for p in posts) / total_posts
            
            # Calculate scores
            positive_ratio = positive_posts / total_posts if total_posts > 0 else 0
            negative_ratio = negative_posts / total_posts if total_posts > 0 else 0
            
            love_score = avg_sentiment * positive_ratio if avg_sentiment > 0 else 0
            hate_score = abs(avg_sentiment) * negative_ratio if avg_sentiment < 0 else 0
            popularity_score = total_posts * (1 + abs(avg_sentiment))  # More posts + strong sentiment = popular
            
            # Get recent posts
            recent_posts = sorted(posts, key=lambda x: x.datetime, reverse=True)[:5]
            
            return {
                "name": name,
                "total_posts": total_posts,
                "positive_posts": positive_posts,
                "negative_posts": negative_posts,
                "neutral_posts": neutral_posts,
                "average_sentiment": round(avg_sentiment, 3),
                "popularity_score": round(popularity_score, 2),
                "love_score": round(love_score, 3),
                "hate_score": round(hate_score, 3),
                "recent_posts": [
                    {
                        "id": post.id,
                        "content": post.content[:200] + "..." if len(post.content) > 200 else post.content,
                        "sentiment_score": round(post.sentiment_score, 3),
                        "created_at": post.datetime.isoformat()
                    }
                    for post in recent_posts
                ]
            }
        
        wrestler1_stats = calculate_wrestler_stats(wrestler1_posts, w1)
        wrestler2_stats = calculate_wrestler_stats(wrestler2_posts, w2)
        
        # Calculate comparison differences
        comparison = {
            "sentiment_difference": wrestler1_stats["average_sentiment"] - wrestler2_stats["average_sentiment"],
            "popularity_difference": wrestler1_stats["popularity_score"] - wrestler2_stats["popularity_score"],
            "love_difference": wrestler1_stats["love_score"] - wrestler2_stats["love_score"],
            "hate_difference": wrestler1_stats["hate_score"] - wrestler2_stats["hate_score"]
        }
        
        return {
            "wrestler1": wrestler1_stats,
            "wrestler2": wrestler2_stats,
            "comparison": {
                "sentiment_difference": round(comparison["sentiment_difference"], 3),
                "popularity_difference": round(comparison["popularity_difference"], 2),
                "love_difference": round(comparison["love_difference"], 3),
                "hate_difference": round(comparison["hate_difference"], 3)
            }
        }
        
    except Exception as e:
        logger.error(f"Compare endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error comparing wrestlers"
        )