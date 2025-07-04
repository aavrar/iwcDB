#!/usr/bin/env python3
"""
Interactive Wrestling Sentiment Training System
Fetches live wrestling posts and lets you manually label them for training
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.scraper import TwitterScraper
from app.core.model_training import WrestlingModelTrainer, WrestlingDatasetCreator


class InteractiveTrainingSystem:
    """Interactive system for manually labeling wrestling posts for training."""
    
    def __init__(self):
        self.scraper = TwitterScraper()
        self.training_data = []
        self.current_post_index = 0
        self.total_posts = 0
        
    async def fetch_wrestling_posts(self, num_posts: int = 100) -> List[Dict]:
        """Fetch recent wrestling posts from Reddit."""
        print("🔍 Fetching latest wrestling posts...")
        
        wrestling_queries = [
            "WWE Monday Night Raw",
            "AEW Dynamite", 
            "Roman Reigns",
            "CM Punk",
            "Cody Rhodes",
            "Seth Rollins",
            "wrestling match review",
            "wrestling promo",
            "wrestling storyline",
            "wrestling booking"
        ]
        
        all_posts = []
        posts_per_query = max(1, num_posts // len(wrestling_queries))
        
        for query in wrestling_queries:
            try:
                posts = await self.scraper.scrape_reddit_posts(query, posts_per_query)
                all_posts.extend(posts)
                if len(all_posts) >= num_posts:
                    break
            except Exception as e:
                print(f"⚠️  Error fetching posts for '{query}': {e}")
                continue
        
        # Remove duplicates and filter for quality
        unique_posts = []
        seen_content = set()
        
        for post in all_posts:
            content = post.get('content', '').strip()
            if (content and 
                len(content) > 20 and 
                len(content) < 500 and
                content not in seen_content and
                any(word in content.lower() for word in ['wrestling', 'wwe', 'aew', 'match', 'wrestler', 'promo', 'storyline'])):
                unique_posts.append(post)
                seen_content.add(content)
        
        return unique_posts[:num_posts]
    
    def display_post(self, post: Dict, index: int, total: int) -> None:
        """Display a post for labeling."""
        print("\n" + "="*80)
        print(f"📝 POST {index + 1} of {total}")
        print("="*80)
        
        content = post.get('content', '')
        title = post.get('title', 'No title')
        score = post.get('score', 0)
        source = post.get('source', 'Unknown')
        
        print(f"🏷️  Title: {title}")
        print(f"📊 Score: {score} | Source: {source}")
        print(f"📄 Content:")
        print(f"   {content}")
        print("\n" + "-"*80)
    
    def get_user_label(self) -> str:
        """Get sentiment label from user."""
        while True:
            print("🎯 How would you classify this post's sentiment?")
            print("   [P]ositive - Good things about wrestling (great match, amazing promo, etc.)")
            print("   [N]egative - Bad things about wrestling (terrible booking, boring match, etc.)")  
            print("   [T]eutral - Neutral/mixed sentiment (okay match, average show, etc.)")
            print("   [S]kip - Skip this post (unclear, not wrestling-related, etc.)")
            print("   [Q]uit - Finish training with current data")
            
            choice = input("\n👉 Your choice (P/N/T/S/Q): ").strip().upper()
            
            if choice in ['P', 'POSITIVE']:
                return 'positive'
            elif choice in ['N', 'NEGATIVE']:
                return 'negative'
            elif choice in ['T', 'NEUTRAL']:
                return 'neutral'
            elif choice in ['S', 'SKIP']:
                return 'skip'
            elif choice in ['Q', 'QUIT']:
                return 'quit'
            else:
                print("❌ Invalid choice. Please enter P, N, T, S, or Q.")
    
    def save_training_data(self, filename: str = None) -> str:
        """Save training data to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wrestling_training_data_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        
        return filepath
    
    async def interactive_labeling_session(self, posts: List[Dict]) -> List[Dict]:
        """Run interactive labeling session."""
        self.total_posts = len(posts)
        labeled_data = []
        
        print(f"\n🎓 INTERACTIVE TRAINING SESSION")
        print(f"📚 You'll label {len(posts)} wrestling posts for sentiment")
        print(f"💡 This will create a custom model trained on your preferences")
        print(f"⏱️  Estimated time: {len(posts) * 0.5:.0f}-{len(posts) * 1:.0f} minutes")
        
        input("\n🚀 Press Enter to start labeling...")
        
        for i, post in enumerate(posts):
            self.display_post(post, i, len(posts))
            
            label = self.get_user_label()
            
            if label == 'quit':
                print(f"\n✋ Training stopped. Labeled {len(labeled_data)} posts so far.")
                break
            elif label == 'skip':
                print("⏭️  Skipping this post...")
                continue
            
            # Convert label to training format
            label_map = {
                'positive': {'label': 2, 'sentiment': 'positive'},
                'negative': {'label': 0, 'sentiment': 'negative'}, 
                'neutral': {'label': 1, 'sentiment': 'neutral'}
            }
            
            training_item = {
                'text': post.get('content', ''),
                'title': post.get('title', ''),
                'original_score': post.get('score', 0),
                'source': post.get('source', 'reddit'),
                'labeled_by': 'user',
                'labeled_at': datetime.now().isoformat(),
                **label_map[label]
            }
            
            labeled_data.append(training_item)
            self.training_data.append(training_item)
            
            print(f"✅ Labeled as {label.upper()}")
            
            # Show progress
            if len(labeled_data) % 10 == 0:
                pos = len([d for d in labeled_data if d['sentiment'] == 'positive'])
                neg = len([d for d in labeled_data if d['sentiment'] == 'negative'])
                neu = len([d for d in labeled_data if d['sentiment'] == 'neutral'])
                print(f"\n📊 Progress: {len(labeled_data)} labeled | Positive: {pos} | Negative: {neg} | Neutral: {neu}")
        
        return labeled_data
    
    def show_training_summary(self, labeled_data: List[Dict]) -> None:
        """Show summary of labeled data."""
        if not labeled_data:
            print("❌ No data labeled!")
            return
        
        pos_count = len([d for d in labeled_data if d['sentiment'] == 'positive'])
        neg_count = len([d for d in labeled_data if d['sentiment'] == 'negative'])
        neu_count = len([d for d in labeled_data if d['sentiment'] == 'neutral'])
        
        print(f"\n📊 TRAINING DATA SUMMARY")
        print(f"   Total labeled: {len(labeled_data)}")
        print(f"   Positive: {pos_count} ({pos_count/len(labeled_data)*100:.1f}%)")
        print(f"   Negative: {neg_count} ({neg_count/len(labeled_data)*100:.1f}%)")
        print(f"   Neutral: {neu_count} ({neu_count/len(labeled_data)*100:.1f}%)")
        
        # Check for balance
        if len(labeled_data) >= 30:
            print("✅ Good amount of data for training!")
        elif len(labeled_data) >= 15:
            print("⚠️  Minimal data - consider labeling more for better accuracy")
        else:
            print("❌ Too little data - need at least 15 samples for training")
    
    async def train_custom_model(self, labeled_data: List[Dict]) -> str:
        """Train a custom model with user-labeled data."""
        if len(labeled_data) < 15:
            print("❌ Need at least 15 labeled samples to train a model")
            return None
        
        print("\n🏋️ Training custom wrestling sentiment model...")
        print("⏱️  This will take 5-15 minutes depending on your hardware")
        
        # Create enhanced training data (combine user data with some base examples)
        dataset_creator = WrestlingDatasetCreator()
        base_data = dataset_creator.create_wrestling_training_data()
        
        # Combine user data with base examples (give user data higher weight)
        combined_data = labeled_data + base_data
        
        print(f"📚 Training on {len(combined_data)} samples:")
        print(f"   Your labeled data: {len(labeled_data)}")
        print(f"   Base examples: {len(base_data)}")
        
        # Train the model
        trainer = WrestlingModelTrainer()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./custom_wrestling_model_{timestamp}"
        
        try:
            output_path = trainer.fine_tune_model(combined_data, model_path)
            print(f"✅ Custom model trained successfully!")
            print(f"📁 Model saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None


async def main():
    """Main interactive training workflow."""
    print("🎯 Interactive Wrestling Sentiment Training System")
    print("=" * 60)
    print("📝 This system will:")
    print("   1. Fetch 100-200 latest wrestling posts")
    print("   2. Let you manually label them as positive/negative/neutral")
    print("   3. Train a custom model based on your preferences")
    print("   4. Test the model and save it for production use")
    
    # Get number of posts to label
    while True:
        try:
            num_posts = input("\n🔢 How many posts do you want to label? (50-200, default 100): ").strip()
            if not num_posts:
                num_posts = 100
            else:
                num_posts = int(num_posts)
            
            if 50 <= num_posts <= 200:
                break
            else:
                print("❌ Please enter a number between 50 and 200")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Initialize system
    training_system = InteractiveTrainingSystem()
    
    # Step 1: Fetch posts
    print(f"\n🔍 Fetching {num_posts} wrestling posts...")
    posts = await training_system.fetch_wrestling_posts(num_posts)
    
    if not posts:
        print("❌ No posts found! Check your internet connection.")
        return
    
    print(f"✅ Found {len(posts)} posts to label")
    
    # Step 2: Interactive labeling
    labeled_data = await training_system.interactive_labeling_session(posts)
    
    # Step 3: Save training data
    if labeled_data:
        data_file = training_system.save_training_data()
        print(f"\n💾 Training data saved to: {data_file}")
        
        # Show summary
        training_system.show_training_summary(labeled_data)
        
        # Step 4: Train model
        train_now = input("\n🤖 Train the model now? (Y/n): ").strip().lower()
        if train_now in ['', 'y', 'yes']:
            model_path = await training_system.train_custom_model(labeled_data)
            
            if model_path:
                print(f"\n🎉 SUCCESS! Your custom wrestling sentiment model is ready!")
                print(f"📁 Model location: {model_path}")
                print(f"\n🔧 To use your custom model, update your .env file:")
                print(f"MODEL_NAME={model_path}")
                print(f"\n🚀 Your model should have much better accuracy on wrestling content!")
        else:
            print("\n⏭️  Model training skipped. You can train later with your saved data.")
    
    else:
        print("\n❌ No data was labeled. Training cancelled.")


if __name__ == "__main__":
    asyncio.run(main())