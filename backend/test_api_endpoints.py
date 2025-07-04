#!/usr/bin/env python3
"""
Test API endpoints to populate some data
"""

import asyncio
import sys
import os
import requests
import json

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.scraper import RedditScraper
from app.core.nlp import analyze_texts_sentiment
from app.models.tweet import PostModel, QueryModel
from app.core.database import get_db
from sqlalchemy.orm import Session
from datetime import datetime


async def populate_test_data():
    """Populate the database with some test data for frontend testing."""
    
    print("🔄 Populating test data...")
    
    # Get database session
    db_gen = get_db()
    db: Session = next(db_gen)
    
    # Test wrestlers to scrape data for
    test_wrestlers = ["CM Punk", "Roman Reigns", "Cody Rhodes", "Seth Rollins", "Jon Moxley", "Rhea Ripley", "Bianca Belair", "Gunther"]
    
    scraper = RedditScraper()
    
    for wrestler in test_wrestlers:
        try:
            print(f"\n📝 Processing {wrestler}...")
            
            # Scrape posts
            posts = await scraper.scrape_reddit_posts(wrestler, 100)
            print(f"   Found {len(posts)} posts")
            
            if posts:
                # Analyze sentiment
                texts = [post.get('content', '') for post in posts]
                sentiment_results = await analyze_texts_sentiment(texts)
                print(f"   Analyzed sentiment for {len(sentiment_results)} posts")
                
                # Store in database
                stored_count = 0
                for post_data, (sentiment_score, confidence) in zip(posts, sentiment_results):
                    try:
                        post_model = PostModel(
                            id=post_data.get('id', ''),
                            content=post_data.get('content', ''),
                            username=post_data.get('author', 'unknown'),
                            datetime=post_data.get('created_at', datetime.utcnow()),
                            sentiment_score=sentiment_score,
                            query=wrestler,
                            extra_data=json.dumps({
                                "confidence": confidence, 
                                "url": post_data.get('url', ''),
                                "score": post_data.get('score', 0),
                                "subreddit": post_data.get('subreddit', '')
                            })
                        )
                        
                        # Check if already exists
                        existing = db.query(PostModel).filter(PostModel.id == post_model.id).first()
                        if not existing:
                            db.add(post_model)
                            stored_count += 1
                    
                    except Exception as e:
                        print(f"     Error storing post: {e}")
                        continue
                
                # Update query stats
                query_model = db.query(QueryModel).filter(QueryModel.query_text == wrestler).first()
                if not query_model:
                    avg_sentiment = sum(r[0] for r in sentiment_results) / len(sentiment_results)
                    query_model = QueryModel(
                        query_text=wrestler,
                        post_count=stored_count,
                        avg_sentiment=avg_sentiment
                    )
                    db.add(query_model)
                
                db.commit()
                print(f"   ✅ Stored {stored_count} new posts for {wrestler}")
        
        except Exception as e:
            print(f"   ❌ Error processing {wrestler}: {e}")
            db.rollback()
    
    db.close()
    print("\n✅ Test data population complete!")


def test_api_endpoints():
    """Test API endpoints to make sure they're working."""
    
    print("\n🧪 Testing API endpoints...")
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/",
        "/health",
        "/api/v1/popular?limit=3",
        "/api/v1/search?q=CM Punk&period=30d"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=30)
            if response.status_code == 200:
                print(f"✅ {endpoint} - Status: {response.status_code}")
                if endpoint == "/api/v1/popular?limit=3":
                    data = response.json()
                    print(f"     Popular wrestlers count: {len(data)}")
            else:
                print(f"⚠️  {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")


async def main():
    """Main function."""
    await populate_test_data()
    test_api_endpoints()


if __name__ == "__main__":
    asyncio.run(main())