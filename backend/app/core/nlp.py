import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import re
from app.core.config import settings
from app.core.logging import logger
import os
import warnings
warnings.filterwarnings("ignore")


class SentimentAnalyzer:
    """Enhanced sentiment analysis service for wrestling content."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.is_classification_model = False
        self.sentiment_classifier = None
        
        # Wrestling-specific keywords for context enhancement
        self.wrestling_keywords = {
            'positive': [
                'amazing', 'incredible', 'awesome', 'perfect', 'legendary', 'goat', 'phenomenal',
                'masterpiece', 'outstanding', 'brilliant', 'spectacular', 'epic', 'beautiful',
                'flawless', 'perfect', 'stunning', 'magnificent', 'wonderful', 'excellent'
            ],
            'negative': [
                'terrible', 'awful', 'horrible', 'disappointing', 'boring', 'trash', 'garbage',
                'worst', 'pathetic', 'cringe', 'embarrassing', 'shameful', 'disgusting',
                'unwatchable', 'disaster', 'failure', 'mess', 'joke', 'ridiculous'
            ],
            'neutral': [
                'okay', 'average', 'fine', 'decent', 'normal', 'standard', 'typical',
                'regular', 'usual', 'moderate', 'mediocre', 'so-so'
            ]
        }
        
        # Wrestling-specific expressions
        self.wrestling_expressions = {
            'positive': ['fire', 'target', '100', 'clap', 'hands', 'trophy', 'star', 'muscle', 'top', 'party'],
            'negative': ['sleep', 'zzz', 'thumbsdown', 'angry', 'rage', 'huff', 'eyeroll', 'tired', 'sad'],
            'neutral': ['shrug', 'neutral', 'meh', 'thinking', 'blank', 'smile', 'happy']
        }
    
    async def load_model(self):
        """Load the optimized sentiment analysis model with quantization."""
        if self.model_loaded:
            return
        
        try:
            # Check if we should use fine-tuned model
            if settings.USE_FINE_TUNED_MODEL and os.path.exists(settings.FINE_TUNED_MODEL_PATH):
                logger.info(f"Loading fine-tuned wrestling model from: {settings.FINE_TUNED_MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(settings.FINE_TUNED_MODEL_PATH)
                self.model = AutoModelForSequenceClassification.from_pretrained(settings.FINE_TUNED_MODEL_PATH)
                self.is_classification_model = True
                logger.info("Fine-tuned wrestling model loaded successfully")
            else:
                # Fallback to cached or base model
                model_path = os.path.join(settings.MODEL_CACHE_DIR, settings.MODEL_NAME.replace('/', '_'))
                
                # Try to load as classification model first
                try:
                    if os.path.exists(model_path):
                        logger.info(f"Loading classification model from cache: {model_path}")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    else:
                        logger.info(f"Downloading classification model: {settings.MODEL_NAME}")
                        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
                        self.model = AutoModelForSequenceClassification.from_pretrained(settings.MODEL_NAME)
                        
                        # Save model to cache
                        os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
                        self.tokenizer.save_pretrained(model_path)
                        self.model.save_pretrained(model_path)
                        logger.info(f"Classification model cached to: {model_path}")
                    
                    self.is_classification_model = True
                    
                except Exception as e:
                    logger.warning(f"Could not load as classification model: {e}")
                    logger.info("Falling back to DistilBERT for sentiment analysis...")
                    
                    # Fallback to proven working model
                    fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
                    fallback_path = os.path.join(settings.MODEL_CACHE_DIR, fallback_model.replace('/', '_'))
                    
                    if os.path.exists(fallback_path):
                        self.tokenizer = AutoTokenizer.from_pretrained(fallback_path)
                        self.model = AutoModelForSequenceClassification.from_pretrained(fallback_path)
                    else:
                        self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                        self.model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
                        
                        os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
                        self.tokenizer.save_pretrained(fallback_path)
                        self.model.save_pretrained(fallback_path)
                    
                    self.is_classification_model = True
                    logger.info(f"Loaded fallback model: {fallback_model}")
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            # Apply quantization if enabled
            if settings.USE_QUANTIZATION and settings.QUANTIZATION_METHOD == "pytorch":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
                logger.info("Applied PyTorch dynamic quantization")
            
            self.model_loaded = True
            
            # Log model info
            model_size = sum(p.numel() for p in self.model.parameters()) / 1_000_000
            logger.info(f"Model loaded successfully on {self.device}, {model_size:.1f}M parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove @mentions but keep the context
        text = re.sub(r'@\w+', '', text)
        
        # Clean up hashtags (keep the text, remove #)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        return text.strip()
    
    def enhance_with_wrestling_context(self, text: str) -> float:
        """Enhance sentiment with wrestling-specific context."""
        text_lower = text.lower()
        
        # Count wrestling-specific keywords
        positive_count = sum(1 for word in self.wrestling_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.wrestling_keywords['negative'] if word in text_lower)
        neutral_count = sum(1 for word in self.wrestling_keywords['neutral'] if word in text_lower)
        
        # Count wrestling-specific expressions
        positive_emojis = sum(1 for expr in self.wrestling_expressions['positive'] if expr in text_lower)
        negative_emojis = sum(1 for expr in self.wrestling_expressions['negative'] if expr in text_lower)
        
        # Calculate context enhancement factor
        total_keywords = positive_count + negative_count + neutral_count
        total_emojis = positive_emojis + negative_emojis
        
        if total_keywords > 0 or total_emojis > 0:
            context_score = (positive_count + positive_emojis - negative_count - negative_emojis) / max(total_keywords + total_emojis, 1)
            return context_score * 0.2  # Weight the context enhancement
        
        return 0.0
    
    async def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment of a single text."""
        if not self.model_loaded:
            await self.load_model()
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return 0.0, 0.0
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            # Convert to sentiment score [-1, 1]
            if len(probabilities) == 3:  # negative, neutral, positive
                sentiment_score = (-1 * probabilities[0]) + (0 * probabilities[1]) + (1 * probabilities[2])
                confidence = max(probabilities)
            else:  # binary classification
                sentiment_score = (probabilities[1] - probabilities[0])
                confidence = max(probabilities)
            
            # Enhance with wrestling context
            context_enhancement = self.enhance_with_wrestling_context(text)
            final_score = np.clip(sentiment_score + context_enhancement, -1, 1)
            
            return float(final_score), float(confidence)
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for text: {text[:50]}... Error: {e}")
            return 0.0, 0.0
    
    async def analyze_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Analyze sentiment for multiple texts."""
        if not self.model_loaded:
            await self.load_model()
        
        results = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), settings.BATCH_SIZE):
            batch_texts = texts[i:i + settings.BATCH_SIZE]
            batch_results = []
            
            for text in batch_texts:
                score, confidence = await self.analyze_sentiment(text)
                batch_results.append((score, confidence))
            
            results.extend(batch_results)
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
        
        return results
    
    def get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def analyze_sentiment_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Analyze distribution of sentiment scores."""
        distribution = {"positive": 0, "negative": 0, "neutral": 0}
        
        for score in scores:
            label = self.get_sentiment_label(score)
            distribution[label] += 1
        
        return distribution
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        return {
            "model_name": settings.MODEL_NAME,
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "cache_dir": settings.MODEL_CACHE_DIR
        }


# Singleton instance
sentiment_analyzer = SentimentAnalyzer()


async def analyze_text_sentiment(text: str) -> Tuple[float, float]:
    """Convenient function to analyze sentiment of a single text."""
    return await sentiment_analyzer.analyze_sentiment(text)


async def analyze_texts_sentiment(texts: List[str]) -> List[Tuple[float, float]]:
    """Convenient function to analyze sentiment of multiple texts."""
    return await sentiment_analyzer.analyze_batch(texts)