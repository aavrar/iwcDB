#!/usr/bin/env python3
"""
Quick script to train a wrestling-specific sentiment model
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.model_training import create_wrestling_training_pipeline

if __name__ == "__main__":
    print("🚀 Starting Wrestling Sentiment Model Training...")
    print("⚠️  This will take 10-30 minutes depending on your hardware")
    print("💡 You can skip this and use the current model, or train for better accuracy")
    
    response = input("\n🤔 Do you want to train the model now? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        try:
            model_path = create_wrestling_training_pipeline()
            print(f"\n🎉 SUCCESS! Wrestling model trained and saved to: {model_path}")
            print("\n🔧 To use the new model, update your .env file:")
            print(f"MODEL_NAME={model_path}")
            print("\nOr restart your app to use the improved model!")
            
        except Exception as e:
            print(f"\n💥 Training failed: {e}")
            print("💡 You can still use the default model - it works reasonably well")
    else:
        print("\n⏭️  Skipping model training")
        print("💡 The current model works well enough for development")
        print("🚀 You can always train later for better accuracy")
    
    print("\n📚 Training Data Info:")
    print("   • 60 hand-crafted wrestling sentiment examples")
    print("   • 20 positive, 20 negative, 20 neutral samples")
    print("   • Wrestling-specific language and terminology")
    print("   • Covers storylines, matches, characters, booking")
    
    print("\n🎯 Expected Improvements:")
    print("   • Better understanding of wrestling context")
    print("   • More accurate sentiment for wrestling jargon")
    print("   • Improved detection of sarcasm in wrestling discussions")
    print("   • Better handling of kayfabe vs. real sentiment")