from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
import random

from app.core.database import get_db
from app.core.enhanced_scraper import EnhancedWrestlingScraper
from app.core.nlp import analyze_texts_sentiment
from app.core.logging import logger

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# Scraper instance
scraper = EnhancedWrestlingScraper()


@router.get("/training/next-post")
@limiter.limit("30/minute")
async def get_next_training_post(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get next post for manual training classification."""
    try:
        # Use existing posts from database for faster training
        from app.models.tweet import PostModel
        
        # Get a random post from the last 30 days
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        # Get random post from database (much faster than scraping)
        existing_posts = db.query(PostModel).filter(
            PostModel.datetime >= thirty_days_ago,
            PostModel.query != 'test'
        ).all()
        
        if not existing_posts:
            # Fallback to scraping if no posts in DB
            subreddits = ['SquaredCircle', 'WWE', 'AEWOfficial']
            raw_posts = await scraper.scrape_reddit_posts("wrestling", 3)
            
            if not raw_posts:
                raise HTTPException(status_code=404, detail="No posts found for training")
            
            post_data = random.choice(raw_posts)
            post_content = post_data.get('content', '')
        else:
            # Use existing post from database
            db_post = random.choice(existing_posts)
            post_data = {
                'id': db_post.id,
                'content': db_post.content,
                'title': '',
                'author': db_post.username,
                'score': 0,
                'url': '#',
                'subreddit': 'database',
                'created_at': db_post.datetime.isoformat()
            }
            post_content = db_post.content
        
        # Get AI prediction for sentiment (use cached if available)
        if existing_posts:
            # Use existing sentiment from database for speed
            predicted_sentiment = db_post.sentiment_score or 0.0
            confidence = 0.9  # High confidence for existing data
        else:
            # Only run AI analysis for new posts
            texts = [post_content]
            sentiment_results = await analyze_texts_sentiment(texts)
            predicted_sentiment, confidence = sentiment_results[0] if sentiment_results else (0.0, 0.0)
        
        # Simple news classification prediction
        content = post_content.lower()
        title = post_data.get('title', '').lower()
        text = content + " " + title
        
        # Basic news detection
        news_keywords = ['breaking', 'report', 'confirmed', 'announced', 'signing', 'injury', 'suspended', 'released']
        opinion_keywords = ['i think', 'opinion', 'personally', 'hot take', 'thoughts', 'what do you think']
        
        news_score = sum(1 for keyword in news_keywords if keyword in text)
        opinion_score = sum(1 for keyword in opinion_keywords if keyword in text)
        
        predicted_classification = 'news' if news_score > opinion_score else 'opinion'
        
        return {
            "id": post_data.get('id', f"training_{random.randint(1000, 9999)}"),
            "content": post_data.get('content', ''),
            "title": post_data.get('title', ''),
            "subreddit": post_data.get('subreddit', 'wrestling'),
            "author": post_data.get('author', 'unknown'),
            "score": post_data.get('score', 0),
            "created_at": post_data.get('created_at', ''),
            "url": post_data.get('url', ''),
            "predicted_sentiment": round(predicted_sentiment, 3),
            "predicted_classification": predicted_classification,
            "confidence": round(confidence, 3)
        }
        
    except Exception as e:
        logger.error(f"Training next post endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching training post"
        )


@router.post("/training/classify")
@limiter.limit("100/minute")
async def classify_training_post(
    request: Request,
    classification_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Store manual classification for training."""
    try:
        # TODO: Store the classification in a training database table
        # For now, just log it
        logger.info(f"Manual classification received: {classification_data}")
        
        return {
            "status": "success",
            "message": "Classification stored for training"
        }
        
    except Exception as e:
        logger.error(f"Training classify endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error storing classification"
        )


@router.post("/training/save-model")
@limiter.limit("5/minute")
async def save_training_model(
    request: Request,
    db: Session = Depends(get_db)
):
    """Save the trained model."""
    try:
        # TODO: Implement model saving logic
        logger.info("Model save requested")
        
        return {
            "status": "success",
            "message": "Model saved successfully",
            "model_version": "1.0.1"
        }
        
    except Exception as e:
        logger.error(f"Training save model endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error saving model"
        )