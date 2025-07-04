from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.database import init_db, get_db
from app.core.logging import logger
from app.core.security import security_middleware, limiter, get_cors_origins
from app.api.endpoints import search, sentiment, wrestlers, timeline, posts, training, phase2_training, labeling
from app.core.enhanced_scraper import EnhancedWrestlingScraper, ScrapedPost
from app.core.nlp import analyze_texts_sentiment
from app.models.tweet import PostModel, QueryModel
from sqlalchemy.orm import Session
from sqlalchemy import func

# Popular wrestlers to pre-cache for landing page
LANDING_CACHE_WRESTLERS = [
    'CM Punk', 'Roman Reigns', 'Cody Rhodes', 'Seth Rollins',
    'Drew McIntyre', 'Jon Moxley', 'Kenny Omega', 'MJF',
    'Hangman Page', 'Orange Cassidy', 'Brock Lesnar', 'John Cena',
    'The Rock', 'Triple H', 'Undertaker', 'Rhea Ripley',
    'Bianca Belair', 'Sasha Banks', 'Becky Lynch', 'Charlotte Flair'
]

async def populate_landing_cache():
    """Pre-populate cache with popular wrestler data for fast landing page loads."""
    try:
        # Get database session
        db_gen = get_db()
        db = next(db_gen)
        
        try:
            # Check if we already have cached data
            existing_posts = db.query(func.count(PostModel.id)).scalar()
            if existing_posts and existing_posts > 50:
                logger.info(f"Landing cache already populated with {existing_posts} posts")
                return
            
            logger.info("Populating landing page cache with popular wrestler data...")
            scraper = EnhancedWrestlingScraper()
            
            total_cached = 0
            for wrestler in LANDING_CACHE_WRESTLERS[:10]:  # Limit to first 10 for startup speed
                try:
                    logger.info(f"Caching data for {wrestler}...")
                    
                    # Check if we already have recent data for this wrestler
                    existing_query = db.query(QueryModel).filter(QueryModel.query_text == wrestler).first()
                    if existing_query and existing_query.post_count > 5:
                        logger.info(f"Skipping {wrestler} - already has {existing_query.post_count} posts")
                        continue
                    
                    # Scrape posts for this wrestler
                    posts = await scraper.scrape_subreddit_posts('SquaredCircle', limit=50, search_query=wrestler)
                    
                    if not posts:
                        logger.warning(f"No posts found for {wrestler}")
                        continue
                    
                    # Analyze sentiment for all posts
                    post_contents = [post.content for post in posts]
                    sentiment_results = await analyze_texts_sentiment(post_contents)
                    
                    # Store in database
                    cached_posts = []
                    for post, (sentiment_score, confidence) in zip(posts, sentiment_results):
                        # Check if post already exists
                        existing_post = db.query(PostModel).filter(PostModel.id == post.id).first()
                        if existing_post:
                            continue
                            
                        post_model = PostModel(
                            id=post.id,
                            content=post.content,
                            username=post.username,
                            datetime=post.datetime,
                            sentiment_score=sentiment_score,
                            query=wrestler,
                            extra_data=f'{{"confidence": {confidence}, "cache_type": "landing"}}'
                        )
                        cached_posts.append(post_model)
                    
                    if cached_posts:
                        # Add posts to database
                        for post_model in cached_posts:
                            db.add(post_model)
                        
                        # Update or create query record
                        if not existing_query:
                            query_model = QueryModel(
                                query_text=wrestler,
                                post_count=len(cached_posts),
                                avg_sentiment=sum(p.sentiment_score for p in cached_posts) / len(cached_posts)
                            )
                            db.add(query_model)
                        else:
                            existing_query.post_count += len(cached_posts)
                            existing_query.avg_sentiment = sum(p.sentiment_score for p in cached_posts) / len(cached_posts)
                            existing_query.last_updated = func.now()
                        
                        db.commit()
                        total_cached += len(cached_posts)
                        logger.info(f"Cached {len(cached_posts)} posts for {wrestler}")
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed to cache data for {wrestler}: {e}")
                    continue
            
            logger.info(f"Landing cache population complete! Cached {total_cached} posts for {len(LANDING_CACHE_WRESTLERS[:10])} wrestlers")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Landing cache population failed: {e}")
        # Don't fail startup if caching fails


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
        
        # Skip landing page cache population for deployment
        # await populate_landing_cache()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Backend API for IWC Sentiment Tracker - Real-time sentiment analysis of wrestling community tweets",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Security middleware
app.middleware("http")(security_middleware)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error"
        }
    )


# Health check endpoints
@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": f"{settings.APP_NAME} is running",
        "version": settings.VERSION,
        "status": "healthy",
        "docs_url": "/docs" if settings.DEBUG else "disabled"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    from app.core.database import get_redis
    from app.core.cache import cache_manager
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG else "production"
    }
    
    # Check Redis connection
    try:
        redis_client = get_redis()
        if redis_client:
            redis_client.ping()
            health_status["redis"] = "connected"
        else:
            health_status["redis"] = "disconnected"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"
    
    # Check cache stats
    try:
        cache_stats = await cache_manager.get_cache_stats()
        health_status["cache"] = cache_stats
    except Exception as e:
        health_status["cache"] = {"status": "error", "error": str(e)}
    
    return health_status


@app.get("/ping")
async def ping():
    """Simple ping endpoint to keep the server awake."""
    return {
        "message": "pong",
        "timestamp": time.time(),
        "status": "alive"
    }


@app.get("/info")
async def get_api_info():
    """Get API information and configuration."""
    from app.core.nlp import sentiment_analyzer
    
    info = {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "api_version": settings.API_V1_STR,
        "debug_mode": settings.DEBUG,
        "model_info": sentiment_analyzer.get_model_info() if sentiment_analyzer.model_loaded else "Model not loaded"
    }
    
    return info


# Include API routers
app.include_router(
    search.router,
    prefix=settings.API_V1_STR,
    tags=["search"]
)

app.include_router(
    sentiment.router,
    prefix=settings.API_V1_STR,
    tags=["sentiment"]
)

app.include_router(
    wrestlers.router,
    prefix=settings.API_V1_STR,
    tags=["wrestlers"]
)

app.include_router(
    timeline.router,
    prefix=settings.API_V1_STR,
    tags=["timeline"]
)

app.include_router(
    posts.router,
    prefix=settings.API_V1_STR,
    tags=["posts"]
)

app.include_router(
    training.router,
    prefix=settings.API_V1_STR,
    tags=["training"]
)

app.include_router(
    phase2_training.router,
    prefix=settings.API_V1_STR,
    tags=["phase2-training"]
)

app.include_router(
    labeling.router,
    prefix=settings.API_V1_STR,
    tags=["labeling"]
)


# Cache management endpoints (admin only in production)
@app.get("/admin/cache/stats")
async def get_cache_stats():
    """Get cache statistics (admin endpoint)."""
    from app.core.cache import cache_manager
    return await cache_manager.get_cache_stats()


@app.delete("/admin/cache/flush")
async def flush_cache():
    """Flush all cache data (admin endpoint)."""
    from app.core.cache import cache_manager
    success = await cache_manager.flush_all_cache()
    return {"success": success, "message": "Cache flushed" if success else "Cache flush failed"}


@app.delete("/admin/cache/query/{query}")
async def clear_query_cache(query: str):
    """Clear cache for specific query (admin endpoint)."""
    from app.core.cache import cache_manager
    success = await cache_manager.clear_cache_for_query(query)
    return {"success": success, "query": query, "message": "Query cache cleared" if success else "Cache clear failed"}


@app.get("/admin/images/stats")
async def get_image_cache_stats():
    """Get image cache statistics (admin endpoint)."""
    from app.core.image_cache import image_cache_manager
    from app.core.database import get_db
    
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        stats = await image_cache_manager.get_cache_stats(db)
        return stats
    finally:
        db.close()


@app.delete("/admin/images/invalidate/{wrestler_name}")
async def invalidate_wrestler_image(wrestler_name: str):
    """Invalidate cached image for a wrestler (admin endpoint)."""
    from app.core.image_cache import image_cache_manager
    from app.core.database import get_db
    
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        success = await image_cache_manager.invalidate_cache(wrestler_name, db)
        return {"success": success, "wrestler": wrestler_name, "message": "Image cache invalidated" if success else "Invalidation failed"}
    finally:
        db.close()


# Database cleanup endpoints
@app.get("/admin/cleanup/stats")
async def get_cleanup_stats():
    """Get database storage statistics (admin endpoint)."""
    from app.core.cleanup import cleanup_service
    from app.core.database import get_db
    
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        stats = await cleanup_service.get_storage_stats(db)
        return stats
    finally:
        db.close()


@app.post("/admin/cleanup/run")
async def run_manual_cleanup():
    """Run manual database cleanup (admin endpoint)."""
    from app.core.cleanup import cleanup_service
    from app.core.database import get_db
    
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        results = await cleanup_service.run_full_cleanup(db)
        return {"success": True, "results": results}
    finally:
        db.close()


@app.delete("/admin/cleanup/old-posts")
async def cleanup_old_posts():
    """Remove posts older than retention period (admin endpoint)."""
    from app.core.cleanup import cleanup_service
    from app.core.database import get_db
    
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        results = await cleanup_service.cleanup_old_posts(db)
        return {"success": True, "results": results}
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )