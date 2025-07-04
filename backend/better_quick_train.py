#!/usr/bin/env python3
"""
Better Quick Training - Fetches more posts and works better
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import List, Dict

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.scraper import TwitterScraper


async def fetch_many_wrestling_posts(target_posts: int = 100) -> List[Dict]:
    """Fetch many wrestling posts from multiple queries."""
    
    scraper = TwitterScraper()
    
    # Use many different queries to get more diverse posts
    wrestling_queries = [
        "WWE",
        "AEW", 
        "wrestling",
        "Roman Reigns",
        "CM Punk",
        "Cody Rhodes",
        "Seth Rollins",
        "wrestling match",
        "wrestling promo",
        "wrestling storyline",
        "wrestling booking",
        "WrestleMania",
        "Monday Night Raw",
        "AEW Dynamite",
        "wrestling review",
        "wrestling opinion",
        "heel turn",
        "face turn",
        "wrestling character",
        "wrestling news"
    ]
    
    all_posts = []
    posts_per_query = max(target_posts // len(wrestling_queries), 5)
    
    print(f"🔍 Fetching posts from {len(wrestling_queries)} different queries...")
    
    for i, query in enumerate(wrestling_queries):
        try:
            print(f"   {i+1}/{len(wrestling_queries)}: '{query}'", end=" ")
            posts = await scraper.scrape_reddit_posts(query, posts_per_query)
            all_posts.extend(posts)
            print(f"-> {len(posts)} posts")
            
            if len(all_posts) >= target_posts * 2:  # Get extra to filter
                break
                
        except Exception as e:
            print(f"-> Error: {e}")
            continue
    
    # Filter for quality posts
    quality_posts = []
    seen_content = set()
    
    for post in all_posts:
        content = post.get('content', '').strip()
        title = post.get('title', '').strip()
        
        # Quality filters
        if (content and 
            len(content) > 25 and 
            len(content) < 400 and
            content not in seen_content and
            any(word in (content + title).lower() for word in [
                'wrestling', 'wwe', 'aew', 'match', 'wrestler', 'promo', 
                'storyline', 'booking', 'heel', 'face', 'champion'
            ])):
            quality_posts.append(post)
            seen_content.add(content)
    
    print(f"✅ Found {len(quality_posts)} quality wrestling posts")
    return quality_posts[:target_posts]


async def better_quick_training():
    """Better quick training session."""
    
    print("🎯 Better Quick Wrestling Sentiment Training")
    print("=" * 60)
    
    # Get number of posts
    while True:
        try:
            target = input("📝 How many posts to label (50-200)? [100]: ").strip()
            if not target:
                target_posts = 100
            else:
                target_posts = int(target)
            
            if 50 <= target_posts <= 200:
                break
            else:
                print("❌ Please enter 50-200")
        except ValueError:
            print("❌ Please enter a number")
        except EOFError:
            target_posts = 100
            break
    
    print(f"🎯 Target: {target_posts} posts")
    
    # Fetch posts
    posts = await fetch_many_wrestling_posts(target_posts)
    
    if len(posts) < 20:
        print("❌ Not enough posts found. Check internet connection.")
        return
    
    print(f"📚 Ready to label {len(posts)} posts")
    print("\n📝 LABELING GUIDE:")
    print("   P = Positive (great match, amazing wrestler, love this)")
    print("   N = Negative (terrible booking, boring, hate this)")
    print("   T = Neutral (okay match, decent show, nothing special)")
    print("   S = Skip (unclear, not wrestling, confusing)")
    print("   Q = Quit and save current progress")
    
    input("\n🚀 Press Enter to start labeling...")
    
    # Interactive labeling
    training_data = []
    
    for i, post in enumerate(posts):
        print(f"\n{'='*70}")
        print(f"📝 POST {i+1} of {len(posts)}")
        print(f"{'='*70}")
        
        title = post.get('title', 'No title')
        content = post.get('content', '')
        score = post.get('score', 0)
        subreddit = post.get('subreddit', 'unknown')
        
        print(f"🏷️  r/{subreddit} | Score: {score}")
        print(f"📰 Title: {title}")
        print(f"📄 Content:")
        print(f"   {content}")
        
        while True:
            try:
                choice = input(f"\n👉 Label (P/N/T/S/Q): ").strip().upper()
                
                if choice == 'Q':
                    print(f"\n✋ Stopping at {len(training_data)} labeled posts")
                    break
                elif choice == 'S':
                    print("⏭️  Skipped")
                    break
                elif choice in ['P', 'N', 'T']:
                    # Save labeled data
                    label_map = {
                        'P': {'label': 2, 'sentiment': 'positive'},
                        'N': {'label': 0, 'sentiment': 'negative'},
                        'T': {'label': 1, 'sentiment': 'neutral'}
                    }
                    
                    training_item = {
                        'text': content,
                        'title': title,
                        'original_score': score,
                        'subreddit': subreddit,
                        'user_label': choice,
                        'labeled_at': datetime.now().isoformat(),
                        **label_map[choice]
                    }
                    
                    training_data.append(training_item)
                    sentiment_name = label_map[choice]['sentiment'].upper()
                    print(f"✅ Labeled as {sentiment_name}")
                    
                    # Progress update
                    if len(training_data) % 10 == 0:
                        pos = len([d for d in training_data if d['sentiment'] == 'positive'])
                        neg = len([d for d in training_data if d['sentiment'] == 'negative'])
                        neu = len([d for d in training_data if d['sentiment'] == 'neutral'])
                        print(f"\n📊 Progress: {len(training_data)} total | +{pos} -{neg} ={neu}")
                    
                    break
                else:
                    print("❌ Please enter P, N, T, S, or Q")
            except (EOFError, KeyboardInterrupt):
                print(f"\n⚠️  Interrupted. Saving {len(training_data)} labeled posts...")
                choice = 'Q'
                break
        
        if choice == 'Q':
            break
    
    # Save results
    if training_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_training_data_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Summary
        pos_count = len([d for d in training_data if d['sentiment'] == 'positive'])
        neg_count = len([d for d in training_data if d['sentiment'] == 'negative']) 
        neu_count = len([d for d in training_data if d['sentiment'] == 'neutral'])
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Total labeled: {len(training_data)}")
        print(f"   Positive: {pos_count}")
        print(f"   Negative: {neg_count}")
        print(f"   Neutral: {neu_count}")
        print(f"   Saved to: {filename}")
        
        # Training recommendation
        if len(training_data) >= 30:
            print(f"\n🎉 Excellent! You have enough data for training")
            print(f"🚀 Next step: python train_with_user_data.py")
        elif len(training_data) >= 15:
            print(f"\n✅ Good! You have enough data for basic training")
            print(f"💡 Consider labeling more for better results")
        else:
            print(f"\n⚠️  Need at least 15 labeled posts for training")
            print(f"💡 Consider running this again to label more")
    
    else:
        print(f"\n❌ No posts labeled. No training data saved.")


if __name__ == "__main__":
    asyncio.run(better_quick_training())