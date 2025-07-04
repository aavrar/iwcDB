from fastapi import APIRouter, Query, HTTPException, Depends, Request
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, case
from datetime import datetime, timedelta
from slowapi import Limiter
from slowapi.util import get_remote_address
import json

from app.core.database import get_db
from app.core.enhanced_scraper import EnhancedWrestlingScraper
from app.core.nlp import analyze_texts_sentiment
from app.core.cache import cache_manager
from app.core.logging import logger
from app.core.image_cache import image_cache_manager
from app.models.tweet import PostModel, QueryModel

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# Scraper instance
scraper = EnhancedWrestlingScraper()


async def _should_scrape_new_data(query: str, existing_posts: list, db: Session) -> bool:
    """Determine if we should scrape new data based on intelligent criteria."""
    post_count = len(existing_posts)
    
    # Always scrape if we have very little data
    if post_count < 5:
        logger.info(f"Low post count ({post_count}) for {query}, scraping needed")
        return True
    
    # For moderate data, check if data is stale
    if post_count < 20:
        # Check when we last updated this query
        query_model = db.query(QueryModel).filter(QueryModel.query_text == query).first()
        if query_model and query_model.last_updated:
            hours_since_update = (datetime.utcnow() - query_model.last_updated).total_seconds() / 3600
            if hours_since_update > 6:  # Data older than 6 hours
                logger.info(f"Stale data ({hours_since_update:.1f}h old) for {query}, scraping needed")
                return True
        else:
            # No query record, definitely need to scrape
            return True
    
    # For popular queries with good data, check if data is very stale
    if post_count >= 20:
        query_model = db.query(QueryModel).filter(QueryModel.query_text == query).first()
        if query_model and query_model.last_updated:
            hours_since_update = (datetime.utcnow() - query_model.last_updated).total_seconds() / 3600
            if hours_since_update > 24:  # Data older than 24 hours for popular wrestlers
                logger.info(f"Very stale data ({hours_since_update:.1f}h old) for popular wrestler {query}, scraping needed")
                return True
    
    logger.info(f"Sufficient fresh data for {query} ({post_count} posts), no scraping needed")
    return False


def calculate_normalized_score(post_count: int, avg_sentiment: float, min_posts: int = 10) -> float:
    """Calculate normalized score to prevent low-post-count wrestlers from dominating."""
    if post_count < min_posts:
        # Heavily penalize low post counts
        return avg_sentiment * (post_count / min_posts) * 0.5
    else:
        # Weight by log of post count to prevent extreme outliers
        import math
        weight = min(1.0, math.log(post_count) / math.log(100))
        return avg_sentiment * weight


@router.get("/popular")
@limiter.limit("30/minute")
async def get_popular_wrestlers(
    request: Request,
    limit: int = Query(10, ge=1, le=20, description="Number of wrestlers to return"),
    db: Session = Depends(get_db)
):
    """Get most discussed wrestlers based on post count."""
    try:
        # Get wrestlers with most posts in last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        results = db.query(
            PostModel.query,
            func.count(PostModel.id).label('post_count'),
            func.avg(PostModel.sentiment_score).label('avg_sentiment')
        ).filter(
            PostModel.datetime >= thirty_days_ago
        ).group_by(
            PostModel.query
        ).having(
            func.count(PostModel.id) >= 5  # Minimum 5 posts
        ).order_by(
            desc('post_count')
        ).limit(limit).all()
        
        popular_wrestlers = []
        for i, (name, post_count, avg_sentiment) in enumerate(results):
            image_url = await image_cache_manager.get_wrestler_image_with_cache(name, db)
            popular_wrestlers.append({
                "name": name,
                "post_count": post_count,
                "sentiment_score": round(avg_sentiment or 0.0, 3),
                "rank": i + 1,
                "image_url": image_url
            })
        
        return popular_wrestlers
        
    except Exception as e:
        logger.error(f"Popular wrestlers endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving popular wrestlers"
        )


@router.get("/loved")
@limiter.limit("30/minute")
async def get_most_loved_wrestlers(
    request: Request,
    limit: int = Query(10, ge=1, le=20, description="Number of wrestlers to return"),
    db: Session = Depends(get_db)
):
    """Get most loved wrestlers based on normalized positive sentiment."""
    try:
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        # Get all wrestler sentiment data
        all_results = db.query(
            PostModel.query,
            func.count(PostModel.id).label('post_count'),
            func.avg(PostModel.sentiment_score).label('avg_sentiment'),
            func.sum(case((PostModel.sentiment_score > 0, 1), else_=0)).label('positive_count'),
            func.sum(case((PostModel.sentiment_score < 0, 1), else_=0)).label('negative_count')
        ).filter(
            PostModel.datetime >= thirty_days_ago
        ).group_by(
            PostModel.query
        ).having(
            func.count(PostModel.id) >= 5  # Minimum 5 posts
        ).all()
        
        # Filter for loved wrestlers - must have net positive sentiment and more positive than negative posts
        loved_wrestlers = []
        for name, post_count, avg_sentiment, positive_count, negative_count in all_results:
            avg_sentiment = avg_sentiment or 0.0
            
            # Must have positive average sentiment AND more positive posts than negative
            if avg_sentiment > 0.1 and positive_count > negative_count:
                normalized_score = calculate_normalized_score(post_count, avg_sentiment)
                image_url = await image_cache_manager.get_wrestler_image_with_cache(name, db)
                loved_wrestlers.append({
                    "name": name,
                    "post_count": post_count,
                    "sentiment_score": round(avg_sentiment, 3),
                    "normalized_score": round(normalized_score, 3),
                    "image_url": image_url
                })
        
        # Sort by normalized score and add ranks
        loved_wrestlers.sort(key=lambda x: x["normalized_score"], reverse=True)
        for i, wrestler in enumerate(loved_wrestlers[:limit]):
            wrestler["rank"] = i + 1
            wrestler.pop("normalized_score")  # Remove internal scoring
        
        return loved_wrestlers[:limit]
        
    except Exception as e:
        logger.error(f"Loved wrestlers endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving loved wrestlers"
        )


@router.get("/hated")
@limiter.limit("30/minute")
async def get_most_hated_wrestlers(
    request: Request,
    limit: int = Query(10, ge=1, le=20, description="Number of wrestlers to return"),
    db: Session = Depends(get_db)
):
    """Get most hated wrestlers based on normalized negative sentiment."""
    try:
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        # Get all wrestler sentiment data
        all_results = db.query(
            PostModel.query,
            func.count(PostModel.id).label('post_count'),
            func.avg(PostModel.sentiment_score).label('avg_sentiment'),
            func.sum(case((PostModel.sentiment_score > 0, 1), else_=0)).label('positive_count'),
            func.sum(case((PostModel.sentiment_score < 0, 1), else_=0)).label('negative_count')
        ).filter(
            PostModel.datetime >= thirty_days_ago
        ).group_by(
            PostModel.query
        ).having(
            func.count(PostModel.id) >= 5  # Minimum 5 posts
        ).all()
        
        # Filter for hated wrestlers - must have net negative sentiment and more negative than positive posts
        hated_wrestlers = []
        for name, post_count, avg_sentiment, positive_count, negative_count in all_results:
            avg_sentiment = avg_sentiment or 0.0
            
            # Must have negative average sentiment AND more negative posts than positive
            if avg_sentiment < -0.1 and negative_count > positive_count:
                # For hated, we want the most negative normalized score
                normalized_score = calculate_normalized_score(post_count, abs(avg_sentiment))
                image_url = await image_cache_manager.get_wrestler_image_with_cache(name, db)
                hated_wrestlers.append({
                    "name": name,
                    "post_count": post_count,
                    "sentiment_score": round(avg_sentiment, 3),
                    "normalized_score": round(normalized_score, 3),
                    "image_url": image_url
                })
        
        # Sort by most negative normalized score
        hated_wrestlers.sort(key=lambda x: x["normalized_score"], reverse=True)
        for i, wrestler in enumerate(hated_wrestlers[:limit]):
            wrestler["rank"] = i + 1
            wrestler.pop("normalized_score")
        
        return hated_wrestlers[:limit]
        
    except Exception as e:
        logger.error(f"Hated wrestlers endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving hated wrestlers"
        )


@router.get("/news")
@limiter.limit("30/minute")
async def get_recent_news(
    request: Request,
    limit: int = Query(15, ge=5, le=25, description="Number of news items to return"),
    db: Session = Depends(get_db)
):
    """Get recent wrestling news and rumors from all subreddits."""
    try:
        hours_ago = datetime.utcnow() - timedelta(days=7)  # Last 7 days
        
        # Get recent posts sorted by recency (score not available in model)
        recent_posts = db.query(PostModel).filter(
            PostModel.datetime >= hours_ago
        ).order_by(
            desc(PostModel.datetime)  # Recency
        ).limit(limit * 3).all()  # Get more to filter since we'll filter for news
        
        # Process and format news items
        news_items = []
        seen_titles = set()
        
        for post in recent_posts:
            # Parse extra_data to get Reddit-specific fields
            extra_data = {}
            if post.extra_data:
                try:
                    extra_data = json.loads(post.extra_data)
                except:
                    pass
            
            # Get title and other Reddit fields from extra_data
            title = extra_data.get('title', '')
            subreddit = extra_data.get('subreddit', 'wrestling')
            author = extra_data.get('author', post.username)
            score = extra_data.get('score', 0)
            url = extra_data.get('url', '#')
            
            # Skip duplicates based on similar titles
            title_key = title[:50].lower() if title else post.content[:50].lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            
            # Enhanced news detection using multiple signals
            is_news = _classify_as_news(title, post.content, subreddit, author)
            
            if is_news and len(news_items) < limit:
                # Get sentiment color
                sentiment_color = 'positive' if post.sentiment_score > 0.1 else \
                                'negative' if post.sentiment_score < -0.1 else 'neutral'
                
                news_items.append({
                    "id": post.id,
                    "title": title or post.content[:100] + "...",
                    "content": post.content[:200] + "..." if len(post.content) > 200 else post.content,
                    "subreddit": subreddit,
                    "score": score,
                    "url": url,
                    "created_at": post.datetime.isoformat(),
                    "author": author,
                    "sentiment_score": round(post.sentiment_score or 0.0, 2),
                    "sentiment_color": sentiment_color,
                    "time_ago": _format_time_ago(post.datetime)
                })
        
        logger.info(f"Retrieved {len(news_items)} recent news items")
        return news_items
        
    except Exception as e:
        logger.error(f"News endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving recent news"
        )


def _classify_as_news(title: str, content: str, subreddit: str, author: str) -> bool:
    """
    Enhanced news classification using multiple signals.
    Combines keyword detection, linguistic patterns, and source credibility.
    """
    if not title and not content:
        return False
    
    text = (title + " " + content).lower()
    
    # 1. Strong news indicators (high confidence)
    strong_news_keywords = [
        'breaking:', 'report:', 'exclusive:', 'confirmed:', 'announced:',
        'according to sources', 'meltzer reports', 'fightful select',
        'pwinsider', 'wrestling observer', 'backstage news', 'spoiler alert',
        'official statement', 'press release', 'wwe announces', 'aew announces',
        'signing', 'contract', 'injury update', 'suspended', 'released',
        'debuts', 'returns', 'retirement', 'heel turn', 'face turn'
    ]
    
    # 2. News source indicators
    news_sources = [
        'dave meltzer', 'bryan alvarez', 'sean ross sapp', 'mike johnson',
        'fightful', 'pwinsider', 'wrestling observer', 'wrestlinginc',
        'cageside seats', 'bleacher report', 'sports illustrated'
    ]
    
    # 3. Reporting verbs and phrases
    reporting_phrases = [
        'sources say', 'sources tell', 'reports indicate', 'it is being reported',
        'according to', 'sources close to', 'backstage word', 'word is',
        'hearing that', 'told that', 'sources confirm', 'multiple sources'
    ]
    
    # 4. Opinion indicators (negative signals)
    opinion_indicators = [
        'i think', 'my opinion', 'in my view', 'personally', 'i believe',
        'hot take', 'unpopular opinion', 'am i the only one', 'does anyone else',
        'change my mind', 'convince me', 'thoughts?', 'what do you think',
        'i feel like', 'i wish', 'i hope', 'i want', 'i love', 'i hate',
        'imo', 'imho', 'just me', 'anyone agree', 'shower thoughts'
    ]
    
    # 5. Question indicators (usually not news)
    question_indicators = [
        'what if', 'who would', 'who should', 'why did', 'why doesn\'t',
        'when will', 'how would', 'what would happen', 'who wins',
        'predictions', 'fantasy booking', 'should we expect'
    ]
    
    # Calculate confidence scores
    news_score = 0
    
    # Strong news keywords (highest weight)
    for keyword in strong_news_keywords:
        if keyword in text:
            news_score += 3
    
    # News sources mentioned
    for source in news_sources:
        if source in text:
            news_score += 2
    
    # Reporting language
    for phrase in reporting_phrases:
        if phrase in text:
            news_score += 2
    
    # Factual indicators
    factual_keywords = ['rumor', 'rumour', 'leaked', 'update', 'news', 'report']
    for keyword in factual_keywords:
        if keyword in text:
            news_score += 1
    
    # Credible authors (wrestling journalists known to break news)
    credible_authors = [
        'daprice82', 'skeach101', 'fellonmyhead', 'FuzzyWuzzyMooMoo',
        'Strike_Gently', 'pwinsider', 'fightful'
    ]
    if author.lower() in [a.lower() for a in credible_authors]:
        news_score += 1
    
    # Penalty for opinion indicators
    for indicator in opinion_indicators:
        if indicator in text:
            news_score -= 2
    
    # Penalty for questions
    for indicator in question_indicators:
        if indicator in text:
            news_score -= 1
    
    # Title structure analysis
    if title:
        title_lower = title.lower()
        # News titles often start with reporting verbs or wrestler names
        if any(title_lower.startswith(verb) for verb in ['reports:', 'breaking:', 'exclusive:', 'update:']):
            news_score += 2
        
        # Questions are usually not news
        if title.strip().endswith('?'):
            news_score -= 1
    
    # Post length analysis (very short posts less likely to be substantial news)
    total_length = len(title or '') + len(content or '')
    if total_length < 50:
        news_score -= 1
    elif total_length > 200:
        news_score += 1
    
    # Threshold for classification (more selective for better accuracy)
    return news_score >= 2


def _extract_url_from_extra_data(extra_data: str) -> str:
    """Extract URL from extra_data JSON string."""
    try:
        import json
        data = json.loads(extra_data)
        return data.get('url', '#')
    except:
        # If extra_data is not JSON or doesn't have URL, return fallback
        return '#'


def _format_time_ago(created_at: datetime) -> str:
    """Format datetime as 'X hours ago' or 'X days ago'."""
    now = datetime.utcnow()
    diff = now - created_at
    
    if diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours}h ago"
    else:
        minutes = diff.seconds // 60
        return f"{minutes}m ago"


@router.get("/suggestions")
@limiter.limit("60/minute")
async def get_wrestler_suggestions(
    request: Request,
    q: str = Query(..., min_length=1, max_length=50, description="Search query for autocomplete"),
    limit: int = Query(8, ge=1, le=15, description="Number of suggestions to return"),
    db: Session = Depends(get_db)
):
    """Get wrestler name suggestions for autocomplete."""
    try:
        # Get unique wrestler names (queries) that match the search term
        suggestions = db.query(
            PostModel.query.label('name'),
            func.count(PostModel.id).label('post_count'),
            func.avg(PostModel.sentiment_score).label('avg_sentiment')
        ).filter(
            PostModel.query.ilike(f"%{q}%")  # Case-insensitive partial match
        ).group_by(
            PostModel.query
        ).having(
            func.count(PostModel.id) >= 3  # Only suggest wrestlers with some data
        ).order_by(
            desc(func.count(PostModel.id))  # Order by popularity (post count)
        ).limit(limit).all()
        
        # Format suggestions with additional data
        suggestion_list = []
        for name, post_count, avg_sentiment in suggestions:
            # Get cached image
            image_url = await image_cache_manager.get_wrestler_image_with_cache(name, db)
            
            suggestion_list.append({
                "name": name,
                "post_count": post_count,
                "sentiment_score": round(avg_sentiment or 0.0, 2),
                "image_url": image_url
            })
        
        return suggestion_list
        
    except Exception as e:
        logger.error(f"Suggestions endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving wrestler suggestions"
        )


@router.get("/trending")
@limiter.limit("30/minute")
async def get_trending_wrestlers(
    request: Request,
    limit: int = Query(10, ge=1, le=20, description="Number of trending wrestlers to return"),
    db: Session = Depends(get_db)
):
    """Get rising and falling stars based on sentiment trends."""
    try:
        # Get data from last 7 days and 7-14 days ago for comparison
        current_week = datetime.utcnow() - timedelta(days=7)
        previous_week = datetime.utcnow() - timedelta(days=14)
        
        # Current week sentiment
        current_data = db.query(
            PostModel.query.label('name'),
            func.count(PostModel.id).label('current_posts'),
            func.avg(PostModel.sentiment_score).label('current_sentiment')
        ).filter(
            PostModel.datetime >= current_week
        ).group_by(
            PostModel.query
        ).having(
            func.count(PostModel.id) >= 5  # Minimum posts for reliability
        ).filter(
            PostModel.query != 'test'  # Exclude test data
        ).filter(
            PostModel.query.notlike('%test%')  # Exclude any test-related queries
        ).all()
        
        # Previous week sentiment  
        previous_data = db.query(
            PostModel.query.label('name'),
            func.count(PostModel.id).label('previous_posts'),
            func.avg(PostModel.sentiment_score).label('previous_sentiment')
        ).filter(
            PostModel.datetime >= previous_week,
            PostModel.datetime < current_week
        ).group_by(
            PostModel.query
        ).having(
            func.count(PostModel.id) >= 5
        ).filter(
            PostModel.query != 'test'  # Exclude test data
        ).filter(
            PostModel.query.notlike('%test%')  # Exclude any test-related queries
        ).all()
        
        # Convert to dictionaries for easy lookup
        current_dict = {name: {'posts': posts, 'sentiment': sentiment} 
                       for name, posts, sentiment in current_data}
        previous_dict = {name: {'posts': posts, 'sentiment': sentiment} 
                        for name, posts, sentiment in previous_data}
        
        rising_stars = []
        falling_stars = []
        
        # Calculate trends for wrestlers present in both periods
        for name in current_dict:
            if name in previous_dict:
                current_sentiment = current_dict[name]['sentiment'] or 0.0
                previous_sentiment = previous_dict[name]['sentiment'] or 0.0
                current_posts = current_dict[name]['posts']
                
                # Calculate sentiment change
                sentiment_change = current_sentiment - previous_sentiment
                
                # Get wrestler image
                image_url = await image_cache_manager.get_wrestler_image_with_cache(name, db)
                
                wrestler_data = {
                    "name": name,
                    "current_sentiment": round(current_sentiment, 3),
                    "previous_sentiment": round(previous_sentiment, 3),
                    "sentiment_change": round(sentiment_change, 3),
                    "current_posts": current_posts,
                    "image_url": image_url,
                    "trend_strength": abs(sentiment_change) * (current_posts ** 0.5)  # Weight by activity
                }
                
                # Rising stars: significant positive sentiment increase
                if sentiment_change > 0.1:
                    rising_stars.append(wrestler_data)
                
                # Falling stars: significant negative sentiment decrease  
                elif sentiment_change < -0.1:
                    falling_stars.append(wrestler_data)
        
        # Sort by trend strength (combination of change magnitude and activity)
        rising_stars.sort(key=lambda x: x['trend_strength'], reverse=True)
        falling_stars.sort(key=lambda x: x['trend_strength'], reverse=True)
        
        return {
            "rising_stars": rising_stars[:limit],
            "falling_stars": falling_stars[:limit],
            "period": "Last 7 days vs previous 7 days"
        }
        
    except Exception as e:
        logger.error(f"Trending endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving trending wrestlers"
        )


@router.get("/search")
@limiter.limit("20/minute")
async def search_analysis(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    period: str = Query("30d", regex="^(30d|90d|6m|1y)$", description="Time period"),
    db: Session = Depends(get_db)
):
    """Analyze sentiment for a search query with comprehensive results."""
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
        
        # Get existing data from database
        posts = db.query(PostModel).filter(
            PostModel.query.ilike(f"%{q}%"),
            PostModel.datetime >= start_date
        ).all()
        
        # Smart auto-population logic
        should_scrape = await _should_scrape_new_data(q, posts, db)
        if should_scrape:
            logger.info(f"Auto-populating data for {q}...")
            raw_posts = await scraper.scrape_reddit_posts(q, 150)
            
            # Process and analyze new posts
            if raw_posts:
                texts = [post.get('content', '') for post in raw_posts]
                sentiment_results = await analyze_texts_sentiment(texts)
                
                for post_data, (sentiment_score, confidence) in zip(raw_posts, sentiment_results):
                    post_model = PostModel(
                        id=post_data.get('id', ''),
                        content=post_data.get('content', ''),
                        username=post_data.get('author', 'unknown'),
                        datetime=datetime.fromisoformat(post_data.get('created_at', datetime.utcnow().isoformat())),
                        sentiment_score=sentiment_score,
                        query=q,
                        extra_data=str({"confidence": confidence, "url": post_data.get('url', '')})
                    )
                    
                    # Check if already exists
                    existing = db.query(PostModel).filter(PostModel.id == post_model.id).first()
                    if not existing:
                        db.add(post_model)
                
                db.commit()
                
                # Re-query to get updated data
                posts = db.query(PostModel).filter(
                    PostModel.query.ilike(f"%{q}%"),
                    PostModel.datetime >= start_date
                ).all()
        
        if not posts:
            return {
                "query": q,
                "type": "wrestler",
                "sentiment_summary": {
                    "overall_sentiment": 0.0,
                    "total_posts": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "period": period
                },
                "top_positive_posts": [],
                "top_negative_posts": [],
                "timeline": []
            }
        
        # Calculate sentiment summary
        total_posts = len(posts)
        positive_posts = [p for p in posts if p.sentiment_score > 0.1]
        negative_posts = [p for p in posts if p.sentiment_score < -0.1]
        neutral_posts = [p for p in posts if -0.1 <= p.sentiment_score <= 0.1]
        
        overall_sentiment = sum(p.sentiment_score for p in posts) / total_posts
        
        # Get top posts
        top_positive = sorted(positive_posts, key=lambda x: x.sentiment_score, reverse=True)[:3]
        top_negative = sorted(negative_posts, key=lambda x: x.sentiment_score)[:3]
        
        # Get wrestler image
        wrestler_image = await image_cache_manager.get_wrestler_image_with_cache(q, db)
        
        return {
            "query": q,
            "type": "wrestler",  # TODO: Detect type (wrestler/event/brand)
            "wrestler_image": wrestler_image,
            "sentiment_summary": {
                "overall_sentiment": round(overall_sentiment, 3),
                "total_posts": total_posts,
                "positive_count": len(positive_posts),
                "negative_count": len(negative_posts),
                "neutral_count": len(neutral_posts),
                "period": period
            },
            "top_positive_posts": [
                {
                    "id": post.id,
                    "content": post.content,
                    "title": "",  # Reddit posts don't have separate titles
                    "score": 0,  # TODO: Get Reddit scores
                    "url": _extract_url_from_extra_data(post.extra_data),
                    "source": "reddit",
                    "created_at": post.datetime.isoformat(),
                    "author": post.username,
                    "sentiment_score": round(post.sentiment_score, 3)
                }
                for post in top_positive
            ],
            "top_negative_posts": [
                {
                    "id": post.id,
                    "content": post.content,
                    "title": "",
                    "score": 0,
                    "url": _extract_url_from_extra_data(post.extra_data),
                    "source": "reddit",
                    "created_at": post.datetime.isoformat(),
                    "author": post.username,
                    "sentiment_score": round(post.sentiment_score, 3)
                }
                for post in top_negative
            ],
            "timeline": []  # TODO: Implement timeline data
        }
        
    except Exception as e:
        logger.error(f"Search analysis endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error performing search analysis"
        )