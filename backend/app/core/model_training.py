#!/usr/bin/env python3
"""
Wrestling-specific sentiment model training and fine-tuning
"""

import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from app.core.config import settings
from app.core.logging import logger


class WrestlingDatasetCreator:
    """Create and manage wrestling-specific sentiment datasets."""
    
    def __init__(self):
        self.wrestling_samples = []
    
    def create_wrestling_training_data(self) -> List[Dict]:
        """Create comprehensive wrestling sentiment training data."""
        
        # Positive wrestling sentiment examples
        positive_samples = [
            "Roman Reigns absolutely killed it at WrestleMania! What a phenomenal match!",
            "That was the best match I've ever seen! Incredible storytelling and athleticism!",
            "CM Punk's return was absolutely perfect! The crowd went insane!",
            "Cody Rhodes finally finishing his story was beautiful and emotional",
            "The booking has been fantastic lately, really enjoying this storyline",
            "Amazing promo work from Seth Rollins tonight, he's so good on the mic",
            "This is why I love wrestling! What an incredible moment!",
            "Best wrestling show in years! Every match delivered!",
            "The wrestling tonight was absolutely spectacular, world-class athletes",
            "Perfect execution of that storyline, the writers nailed it",
            "Outstanding performance by both wrestlers, they have great chemistry",
            "This rivalry is must-see TV, incredible character development",
            "The crowd was electric tonight, what an atmosphere!",
            "Legendary performance, this will go down in wrestling history",
            "Brilliant match psychology and storytelling throughout",
            "The technical wrestling was flawless, these guys are artists",
            "Best heel turn I've seen in years, perfectly executed",
            "The emotional investment in this story is incredible",
            "What a comeback! Never gave up, true fighting spirit!",
            "This wrestler has improved so much, really impressed with their growth"
        ]
        
        # Negative wrestling sentiment examples
        negative_samples = [
            "That booking decision was terrible. Worst storyline ever, so disappointing",
            "This match was boring and went on way too long, couldn't wait for it to end",
            "The writing has been awful lately, makes no sense at all",
            "Such a waste of talent, this wrestler deserves so much better",
            "The crowd was dead silent, nobody cared about this match",
            "Terrible finish to an otherwise decent match, ruined everything",
            "This storyline is going nowhere, completely pointless and boring",
            "The wrestling was sloppy tonight, lots of botched moves",
            "Disappointing return, expected so much more from this wrestler",
            "The commentary was annoying and took away from the match",
            "This feud has been dragging on forever, time to move on",
            "Poor character development, this person has no personality",
            "The match felt rushed and thrown together, no chemistry",
            "Terrible promo work, couldn't deliver the lines convincingly",
            "This booking makes the champion look weak and ineffective",
            "The pacing of this show was all wrong, badly structured",
            "Overrated wrestler, don't understand all the hype around them",
            "The storyline doesn't make sense, too many plot holes",
            "Boring match with no excitement or drama whatsoever",
            "This wrestler phone it in, looked like they didn't care"
        ]
        
        # Neutral wrestling sentiment examples
        neutral_samples = [
            "The match was okay, nothing special but not bad either",
            "Standard wrestling show tonight, some good moments and some not so good",
            "This wrestler is decent, has potential but needs more development",
            "The storyline is progressing at a reasonable pace, we'll see where it goes",
            "Average match quality tonight, about what you'd expect",
            "This feud is fine I guess, not amazing but watchable",
            "The wrestling was competent, both wrestlers did their jobs",
            "Solid performance overall, no major complaints or praise",
            "The show was fine, had some entertainment value",
            "This wrestler is alright, nothing wrong but nothing spectacular",
            "The booking is predictable but makes sense for the story",
            "Standard episode, moved some storylines forward",
            "The match was watchable, served its purpose in the show",
            "Decent promo work, got the point across effectively",
            "The crowd reaction was mixed, some liked it and some didn't",
            "This storyline is developing slowly but steadily",
            "The wrestling was technically sound, no major issues",
            "Adequate performance from both wrestlers involved",
            "The show was what it was, nothing more nothing less",
            "This segment served its purpose in advancing the plot"
        ]
        
        # Create labeled dataset
        training_data = []
        
        for text in positive_samples:
            training_data.append({"text": text, "label": 2, "sentiment": "positive"})
        
        for text in negative_samples:
            training_data.append({"text": text, "label": 0, "sentiment": "negative"})
        
        for text in neutral_samples:
            training_data.append({"text": text, "label": 1, "sentiment": "neutral"})
        
        return training_data
    
    def collect_reddit_training_data(self, reddit_posts: List[Dict]) -> List[Dict]:
        """Convert Reddit posts to training data format."""
        training_data = []
        
        for post in reddit_posts:
            # Use Reddit score as a rough sentiment indicator
            score = post.get('score', 0)
            content = post.get('content', '')
            
            if len(content) < 10:  # Skip very short content
                continue
            
            # Rough labeling based on Reddit score and keywords
            if score > 50:  # High score usually means positive
                label = 2
                sentiment = "positive"
            elif score < -5:  # Negative score means negative
                label = 0
                sentiment = "negative"
            else:  # Neutral
                label = 1
                sentiment = "neutral"
            
            training_data.append({
                "text": content,
                "label": label,
                "sentiment": sentiment,
                "source": "reddit",
                "original_score": score
            })
        
        return training_data


class WrestlingModelTrainer:
    """Train and fine-tune models for wrestling sentiment analysis."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_base_model(self):
        """Load the base pre-trained model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3  # negative, neutral, positive
            )
            self.model.to(self.device)
            logger.info(f"Base model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def prepare_dataset(self, training_data: List[Dict]) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Split into train/validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label'].tolist()
        )
        
        # Tokenize
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': val_labels
        })
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
        }
    
    def fine_tune_model(self, training_data: List[Dict], output_dir: str = "./wrestling_sentiment_model"):
        """Fine-tune the model on wrestling data."""
        
        if not self.model or not self.tokenizer:
            self.load_base_model()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_dataset(training_data)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Evaluate
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        return output_dir
    
    def test_model(self, model_path: str, test_texts: List[str]) -> List[Tuple[float, str]]:
        """Test the fine-tuned model."""
        
        # Load fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        
        results = []
        
        for text in test_texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            # Convert to sentiment score [-1, 1]
            sentiment_score = (-1 * probabilities[0]) + (0 * probabilities[1]) + (1 * probabilities[2])
            
            # Get label
            predicted_label = ["negative", "neutral", "positive"][np.argmax(probabilities)]
            
            results.append((float(sentiment_score), predicted_label))
        
        return results


def create_wrestling_training_pipeline():
    """Complete pipeline to create and train a wrestling sentiment model."""
    
    print("🎯 Wrestling Sentiment Model Training Pipeline")
    print("=" * 50)
    
    # Step 1: Create training data
    print("📝 Creating wrestling training data...")
    dataset_creator = WrestlingDatasetCreator()
    training_data = dataset_creator.create_wrestling_training_data()
    
    print(f"✅ Created {len(training_data)} training samples")
    print(f"   Positive: {len([d for d in training_data if d['label'] == 2])}")
    print(f"   Neutral:  {len([d for d in training_data if d['label'] == 1])}")
    print(f"   Negative: {len([d for d in training_data if d['label'] == 0])}")
    
    # Step 2: Initialize trainer
    print("\n🤖 Initializing model trainer...")
    trainer = WrestlingModelTrainer()
    
    # Step 3: Fine-tune model
    print("\n🏋️ Fine-tuning model on wrestling data...")
    output_dir = trainer.fine_tune_model(training_data)
    
    print(f"✅ Model saved to: {output_dir}")
    
    # Step 4: Test the model
    print("\n🧪 Testing fine-tuned model...")
    test_texts = [
        "Roman Reigns had an amazing match! What a phenomenal performance!",
        "That booking was terrible and made no sense at all",
        "The match was okay, nothing special but watchable",
        "CM Punk's return was absolutely perfect! The crowd went crazy!",
        "This storyline is boring and going nowhere fast"
    ]
    
    results = trainer.test_model(output_dir, test_texts)
    
    for text, (score, label) in zip(test_texts, results):
        print(f"   Text: {text[:60]}...")
        print(f"   Prediction: {label} (score: {score:.3f})")
        print()
    
    return output_dir


if __name__ == "__main__":
    model_path = create_wrestling_training_pipeline()