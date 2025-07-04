#!/usr/bin/env python3
"""
Test Reddit scraping functionality directly
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.scraper import RedditScraper


async def test_reddit_scraping():
    """Test Reddit scraping functionality."""
    
    print("🔍 Testing Reddit Scraping...")
    print("=" * 50)
    
    # Initialize scraper
    scraper = RedditScraper()
    
    # Test queries
    test_queries = ["CM Punk", "Roman Reigns", "WWE"]
    
    for query in test_queries:
        try:
            print(f"\n📝 Testing query: '{query}'")
            posts = await scraper.scrape_reddit_posts(query, 5)
            
            print(f"✅ Found {len(posts)} posts")
            
            for i, post in enumerate(posts[:3]):  # Show first 3
                content = post.get('content', '')[:100]
                author = post.get('author', 'unknown')
                subreddit = post.get('subreddit', 'unknown')
                score = post.get('score', 0)
                
                print(f"   {i+1}. r/{subreddit} | Score: {score} | @{author}")
                print(f"      Content: {content}...")
                print()
                
        except Exception as e:
            print(f"❌ Error with '{query}': {e}")
    
    print("✅ Reddit scraping test complete!")


if __name__ == "__main__":
    asyncio.run(test_reddit_scraping())