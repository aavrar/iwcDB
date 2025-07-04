import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import re
from app.core.config import settings
from app.core.logging import logger
import os


class ImprovedWrestlingSentimentAnalyzer:
    """Improved sentiment analyzer with wrestling-specific adjustments."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
        # Enhanced wrestling-specific keywords
        self.wrestling_keywords = {
            'positive': [
                'amazing', 'incredible', 'awesome', 'perfect', 'legendary', 'goat', 'phenomenal',
                'masterpiece', 'outstanding', 'brilliant', 'spectacular', 'epic', 'beautiful',
                'flawless', 'stunning', 'magnificent', 'wonderful', 'excellent', 'fantastic',
                # Wrestling-specific positive terms
                'worked', 'over', 'popped', 'elevated', 'delivered', 'nailed', 'executed',
                'chemistry', 'storytelling', 'psychology', 'compelling', 'engaging'
            ],
            'negative': [
                'terrible', 'awful', 'horrible', 'disappointing', 'boring', 'trash', 'garbage',
                'worst', 'pathetic', 'cringe', 'embarrassing', 'shameful', 'disgusting',
                'unwatchable', 'disaster', 'failure', 'mess', 'joke', 'ridiculous',
                # Wrestling-specific negative terms
                'buried', 'jobbed', 'botched', 'squashed', 'ruined', 'wasted', 'rushed'
            ],
            'neutral': [
                'okay', 'average', 'fine', 'decent', 'normal', 'standard', 'typical',
                'regular', 'usual', 'moderate', 'mediocre', 'so-so', 'watchable',
                'progressing', 'developing', 'building', 'potential'
            ]
        }
        
        # Wrestling context patterns that are actually positive
        self.positive_wrestling_patterns = [
            r'heel.*work',  # "heel work" is positive
            r'worked.*crowd',  # "worked the crowd" is positive
            r'got.*over',  # "got over" is positive
            r'heel.*turn',  # "heel turn" is usually positive storytelling
            r'face.*turn',  # "face turn" is usually positive storytelling
            r'good.*heel',  # "good heel" is positive
            r'great.*heel',  # "great heel" is positive
        ]
        
        # Patterns that indicate negative sentiment
        self.negative_wrestling_patterns = [
            r'buried.*wrestler',
            r'bad.*booking',
            r'terrible.*writing',
            r'wasted.*talent',
            r'pushing.*too.*hard',
        ]
    
    async def load_model(self):
        """Load the sentiment analysis model."""
        if self.model_loaded:
            return
        
        try:
            model_path = os.path.join(settings.MODEL_CACHE_DIR, settings.MODEL_NAME.replace('/', '_'))
            
            if os.path.exists(model_path):
                logger.info(f"Loading model from cache: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                logger.info(f"Downloading model: {settings.MODEL_NAME}")
                self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(settings.MODEL_NAME)
                
                # Save model to cache
                os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
                self.tokenizer.save_pretrained(model_path)
                self.model.save_pretrained(model_path)
                logger.info(f"Model cached to: {model_path}")
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            logger.info(f"Improved model loaded successfully on device: {self.device}")
            
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
        
        # Convert to lowercase for analysis (but keep original for pattern matching)
        return text.strip()
    
    def analyze_wrestling_context(self, text: str) -> float:
        """Enhanced wrestling context analysis."""
        text_lower = text.lower()
        context_adjustment = 0.0
        
        # Check for positive wrestling patterns
        for pattern in self.positive_wrestling_patterns:
            if re.search(pattern, text_lower):
                context_adjustment += 0.3  # Strong positive adjustment
                logger.debug(f"Found positive wrestling pattern: {pattern}")
        
        # Check for negative wrestling patterns
        for pattern in self.negative_wrestling_patterns:
            if re.search(pattern, text_lower):
                context_adjustment -= 0.3  # Strong negative adjustment
                logger.debug(f"Found negative wrestling pattern: {pattern}")
        
        # Count wrestling-specific keywords
        positive_count = sum(1 for word in self.wrestling_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.wrestling_keywords['negative'] if word in text_lower)
        neutral_count = sum(1 for word in self.wrestling_keywords['neutral'] if word in text_lower)
        
        # Keyword-based adjustment
        total_keywords = positive_count + negative_count + neutral_count
        if total_keywords > 0:
            keyword_score = (positive_count - negative_count) / total_keywords
            context_adjustment += keyword_score * 0.15
        
        # Special handling for neutral keywords
        if neutral_count > positive_count and neutral_count > negative_count:
            context_adjustment -= 0.1  # Nudge towards neutral
        
        return context_adjustment
    
    async def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment with improved wrestling context."""
        if not self.model_loaded:
            await self.load_model()
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return 0.0, 0.0
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text.lower(),  # Model expects lowercase
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
                base_sentiment = (-1 * probabilities[0]) + (0 * probabilities[1]) + (1 * probabilities[2])
                confidence = max(probabilities)
            else:  # binary classification
                base_sentiment = (probabilities[1] - probabilities[0])
                confidence = max(probabilities)
            
            # Apply wrestling context adjustments
            context_adjustment = self.analyze_wrestling_context(text)
            final_score = np.clip(base_sentiment + context_adjustment, -1, 1)
            
            # Improved neutral detection
            if abs(final_score) < 0.2 and any(word in text.lower() for word in self.wrestling_keywords['neutral']):
                final_score *= 0.5  # Reduce magnitude for likely neutral content
            
            logger.debug(f"Sentiment analysis: '{text[:50]}...' -> base: {base_sentiment:.3f}, context: {context_adjustment:.3f}, final: {final_score:.3f}")
            
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
        """Convert sentiment score to label with improved thresholds."""
        if score > 0.15:  # Slightly higher threshold for positive
            return "positive"
        elif score < -0.15:  # Slightly higher threshold for negative
            return "negative"
        else:
            return "neutral"


# Create improved analyzer instance
improved_sentiment_analyzer = ImprovedWrestlingSentimentAnalyzer()


async def analyze_text_sentiment_improved(text: str) -> Tuple[float, float]:
    """Improved sentiment analysis function."""
    return await improved_sentiment_analyzer.analyze_sentiment(text)


async def analyze_texts_sentiment_improved(texts: List[str]) -> List[Tuple[float, float]]:
    """Improved batch sentiment analysis function."""
    return await improved_sentiment_analyzer.analyze_batch(texts)