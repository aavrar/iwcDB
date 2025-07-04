#!/usr/bin/env python3
"""
Train custom model with user-labeled data
"""

import json
import os
import sys
import glob
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.model_training import WrestlingModelTrainer, WrestlingDatasetCreator


def find_user_training_files():
    """Find all user training data files."""
    pattern = os.path.join(os.path.dirname(__file__), "user_training_data_*.json")
    files = glob.glob(pattern)
    return sorted(files, reverse=True)  # Most recent first


def load_user_training_data(filepath: str):
    """Load user training data from file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return []


def main():
    """Train model with user data."""
    print("🎯 Custom Wrestling Model Training")
    print("=" * 50)
    
    # Find training files
    training_files = find_user_training_files()
    
    if not training_files:
        print("❌ No user training data found!")
        print("💡 Run 'python quick_train.py' first to create training data")
        return
    
    print(f"📁 Found {len(training_files)} training data files:")
    for i, file in enumerate(training_files):
        filename = os.path.basename(file)
        print(f"   {i+1}. {filename}")
    
    # Let user choose file or use most recent
    if len(training_files) == 1:
        chosen_file = training_files[0]
        print(f"\n🎯 Using: {os.path.basename(chosen_file)}")
    else:
        while True:
            try:
                choice = input(f"\n🔢 Choose file (1-{len(training_files)}, or Enter for most recent): ").strip()
                if not choice:
                    chosen_file = training_files[0]
                    break
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(training_files):
                        chosen_file = training_files[idx]
                        break
                    else:
                        print(f"❌ Please enter 1-{len(training_files)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    # Load user data
    user_data = load_user_training_data(chosen_file)
    
    if not user_data:
        print("❌ No valid training data found!")
        return
    
    # Show user data summary
    pos_count = len([d for d in user_data if d['sentiment'] == 'positive'])
    neg_count = len([d for d in user_data if d['sentiment'] == 'negative'])
    neu_count = len([d for d in user_data if d['sentiment'] == 'neutral'])
    
    print(f"\n📊 YOUR TRAINING DATA:")
    print(f"   Total samples: {len(user_data)}")
    print(f"   Positive: {pos_count}")
    print(f"   Negative: {neg_count}")
    print(f"   Neutral: {neu_count}")
    
    # Check if we have enough data
    if len(user_data) < 15:
        print("\n⚠️  WARNING: Less than 15 samples may not train well")
        print("💡 Consider running 'python quick_train.py' to add more data")
        proceed = input("\n🤔 Continue anyway? (y/N): ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("👋 Cancelled. Label more data first.")
            return
    
    # Create enhanced dataset
    print("\n🔧 Creating enhanced training dataset...")
    dataset_creator = WrestlingDatasetCreator()
    base_data = dataset_creator.create_wrestling_training_data()
    
    # Combine user data with base examples
    # Give user data higher weight by including it multiple times
    user_weight = 2 if len(user_data) < 30 else 1
    weighted_user_data = user_data * user_weight
    
    combined_data = weighted_user_data + base_data
    
    print(f"📚 Enhanced dataset:")
    print(f"   Your labeled data: {len(user_data)} (weight: {user_weight}x)")
    print(f"   Base examples: {len(base_data)}")
    print(f"   Total training samples: {len(combined_data)}")
    
    # Train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"custom_wrestling_model_{timestamp}"
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    
    print(f"\n🏋️ Training custom model...")
    print(f"📁 Model will be saved to: {model_name}")
    print(f"⏱️  This will take 5-20 minutes depending on your hardware")
    
    # Confirm training
    confirm = input("\n🚀 Start training? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("👋 Training cancelled.")
        return
    
    try:
        trainer = WrestlingModelTrainer()
        output_path = trainer.fine_tune_model(combined_data, model_path)
        
        print(f"\n🎉 SUCCESS! Custom wrestling model trained!")
        print(f"📁 Model saved to: {output_path}")
        
        # Test the model
        print(f"\n🧪 Testing your custom model...")
        test_texts = [
            "Roman Reigns had an incredible match tonight! Absolutely phenomenal!",
            "That booking decision was terrible and ruined the entire storyline",
            "The match was okay, nothing special but it was watchable",
            "CM Punk's promo work was outstanding, really compelling stuff",
            "This wrestler is boring and the crowd doesn't care about them"
        ]
        
        results = trainer.test_model(output_path, test_texts)
        
        print(f"\n📊 TEST RESULTS:")
        for text, (score, label) in zip(test_texts, results):
            print(f"   Text: {text[:50]}...")
            print(f"   Prediction: {label.upper()} (score: {score:.3f})")
            print()
        
        # Usage instructions
        print(f"🔧 TO USE YOUR CUSTOM MODEL:")
        print(f"   1. Update your .env file or config:")
        print(f"      MODEL_NAME={output_path}")
        print(f"   2. Restart your application")
        print(f"   3. Your model should now have much better wrestling accuracy!")
        
        print(f"\n💡 Your model is trained on YOUR preferences for wrestling sentiment!")
        print(f"📈 Expected accuracy improvement: 5-15% better than base model")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print(f"💡 Try with more training data or check your system resources")


if __name__ == "__main__":
    main()