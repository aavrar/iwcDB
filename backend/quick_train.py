#!/usr/bin/env python3
"""
Quick Interactive Wrestling Sentiment Training
Simplified version for fast custom model training
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


async def quick_training_session():
    """Quick training session with live wrestling posts."""
    
    print("🎯 Quick Wrestling Sentiment Training")
    print("=" * 50)
    print("📝 This will:")
    print("   1. Fetch 50 recent wrestling posts")
    print("   2. You label them as P(ositive), N(egative), or T(eutral)")
    print("   3. Save training data for model improvement")
    print("   4. Takes ~10-15 minutes")
    
    proceed = input("\n🚀 Ready to start? (Y/n): ").strip().lower()
    if proceed in ['n', 'no']:
        print("👋 Maybe next time!")
        return
    
    # Initialize scraper
    scraper = TwitterScraper()
    
    # Fetch posts
    print("\n🔍 Fetching wrestling posts...")
    queries = ["WWE", "AEW", "wrestling match", "wrestling promo", "Roman Reigns"]
    
    all_posts = []
    for query in queries:
        try:
            posts = await scraper.scrape_reddit_posts(query, 10)
            all_posts.extend(posts)
        except Exception as e:
            print(f"⚠️  Error with '{query}': {e}")
    
    # Filter and clean posts
    clean_posts = []
    seen = set()
    
    for post in all_posts:
        content = post.get('content', '').strip()
        if (content and 
            len(content) > 25 and 
            len(content) < 300 and
            content not in seen and
            any(word in content.lower() for word in ['wrestling', 'wwe', 'aew', 'match', 'wrestler'])):
            clean_posts.append(post)
            seen.add(content)
    
    if len(clean_posts) < 10:
        print("❌ Not enough posts found. Check internet connection.")
        return
    
    # Limit to 50 posts max
    posts_to_label = clean_posts[:50]
    print(f"✅ Found {len(posts_to_label)} posts to label")
    
    # Interactive labeling
    training_data = []
    
    print("\n📚 LABELING INSTRUCTIONS:")
    print("   P = Positive (great match, amazing promo, love this wrestler)")
    print("   N = Negative (terrible booking, boring match, hate this storyline)")
    print("   T = Neutral (okay match, standard show, decent wrestler)")
    print("   S = Skip (unclear, not wrestling related)")
    print("   Q = Quit and save current progress")
    
    input("\n🎯 Press Enter to start labeling...")
    
    for i, post in enumerate(posts_to_label):
        print(f"\n{'='*60}")
        print(f"📝 POST {i+1} of {len(posts_to_label)}")
        print(f"{'='*60}")
        
        content = post.get('content', '')
        title = post.get('title', 'No title')
        score = post.get('score', 0)
        
        print(f"Title: {title}")
        print(f"Score: {score}")
        print(f"Content: {content}")
        
        while True:
            choice = input(f"\n👉 Label this post (P/N/T/S/Q): ").strip().upper()
            
            if choice == 'Q':
                print(f"\n✋ Stopping. Labeled {len(training_data)} posts.")
                break
            elif choice == 'S':
                print("⏭️  Skipped")
                break
            elif choice in ['P', 'N', 'T']:
                # Convert to training format
                label_map = {
                    'P': {'label': 2, 'sentiment': 'positive'},
                    'N': {'label': 0, 'sentiment': 'negative'},
                    'T': {'label': 1, 'sentiment': 'neutral'}
                }
                
                training_item = {
                    'text': content,
                    'title': title,
                    'original_score': score,
                    'user_label': choice,
                    'labeled_at': datetime.now().isoformat(),
                    **label_map[choice]
                }
                
                training_data.append(training_item)
                print(f"✅ Labeled as {label_map[choice]['sentiment'].upper()}")
                
                # Show progress every 10 labels
                if len(training_data) % 10 == 0:
                    pos = len([d for d in training_data if d['sentiment'] == 'positive'])
                    neg = len([d for d in training_data if d['sentiment'] == 'negative'])
                    neu = len([d for d in training_data if d['sentiment'] == 'neutral'])
                    print(f"\n📊 Progress: {len(training_data)} labeled | +{pos} -{neg} ={neu}")
                
                break
            else:
                print("❌ Please enter P, N, T, S, or Q")
        
        if choice == 'Q':
            break
    
    # Save training data
    if training_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_training_data_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Show summary
        pos_count = len([d for d in training_data if d['sentiment'] == 'positive'])
        neg_count = len([d for d in training_data if d['sentiment'] == 'negative'])
        neu_count = len([d for d in training_data if d['sentiment'] == 'neutral'])
        
        print(f"\n📊 TRAINING DATA SUMMARY")
        print(f"   Total labeled: {len(training_data)}")
        print(f"   Positive: {pos_count}")
        print(f"   Negative: {neg_count}")
        print(f"   Neutral: {neu_count}")
        print(f"   Saved to: {filename}")
        
        if len(training_data) >= 20:
            print("\n✅ Great! You have enough data for training.")
            print("🔧 To train a custom model, run:")
            print("   python train_with_user_data.py")
        else:
            print("\n⚠️  Consider labeling more posts for better accuracy (aim for 30+)")
    
    else:
        print("\n❌ No data labeled. No training data saved.")


if __name__ == "__main__":
    asyncio.run(quick_training_session())