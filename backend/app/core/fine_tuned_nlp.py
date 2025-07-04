"""
Fine-tuned multi-task wrestling sentiment analysis service.
Handles the custom multi-task model with content type + sentiment classification.
"""
import torch
import asyncio
from transformers import AutoTokenizer
from typing import Tuple
import numpy as np
from datetime import datetime
import re
from app.core.config import settings
from app.core.logging import logger
from app.core.fine_tuning import MultiTaskWrestlingModel
import os
import warnings
warnings.filterwarnings("ignore")


class FineTunedWrestlingSentimentAnalyzer:
    """Fine-tuned multi-task sentiment analysis service for wrestling content."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
        # Label mappings from training
        self.content_type_to_id = {'news': 0, 'rumor': 1, 'opinion': 2}
        self.id_to_content_type = {0: 'news', 1: 'rumor', 2: 'opinion'}
        self.sentiment_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id_to_sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
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
    
    async def load_model(self):
        """Load the fine-tuned multi-task wrestling model."""
        if self.model_loaded:
            return
        
        try:
            if not os.path.exists(settings.FINE_TUNED_MODEL_PATH):
                raise FileNotFoundError(f"Fine-tuned model not found at {settings.FINE_TUNED_MODEL_PATH}")
            
            logger.info(f"Loading fine-tuned multi-task wrestling model from: {settings.FINE_TUNED_MODEL_PATH}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(settings.FINE_TUNED_MODEL_PATH)
            
            # Load our custom multi-task model
            self.model = MultiTaskWrestlingModel(
                base_model_name=settings.MODEL_NAME,
                num_content_labels=3,
                num_sentiment_labels=3
            )
            
            # Load the fine-tuned weights from safetensors
            from safetensors.torch import load_file
            checkpoint = load_file(
                os.path.join(settings.FINE_TUNED_MODEL_PATH, 'model.safetensors'),
                device=str(self.device)
            )
            
            # Load state dict (handle potential key mismatches)
            try:
                self.model.load_state_dict(checkpoint, strict=False)
                logger.info("Loaded fine-tuned model weights successfully")
            except Exception as e:
                logger.warning(f"Could not load state dict strictly: {e}")
                # Try loading with prefix adjustments
                state_dict = {}
                for key, value in checkpoint.items():
                    # Remove 'roberta.' prefix if present
                    new_key = key.replace('roberta.', '') if key.startswith('roberta.') else key
                    state_dict[new_key] = value
                
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded fine-tuned model weights with key adjustments")
            
            # Move to device and set eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            
            # Log model info
            model_size = sum(p.numel() for p in self.model.parameters()) / 1_000_000
            logger.info(f"Fine-tuned multi-task model loaded successfully on {self.device}, {model_size:.1f}M parameters")
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
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
        
        # Calculate context enhancement factor
        total_keywords = positive_count + negative_count + neutral_count
        
        if total_keywords > 0:
            context_score = (positive_count - negative_count) / total_keywords
            return context_score * 0.2  # Weight the context enhancement
        
        return 0.0
    
    async def analyze_content_and_sentiment(self, text: str) -> Tuple[str, float, str, float]:
        """Analyze both content type and sentiment with the multi-task model."""
        if not self.model_loaded:
            await self.load_model()
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return "opinion", 0.0, "neutral", 0.0
            
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
            
            # Predict with multi-task model
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get content type prediction
                content_probs = torch.nn.functional.softmax(outputs['content_logits'], dim=-1).cpu().numpy()[0]
                content_pred_id = np.argmax(content_probs)
                content_type = self.id_to_content_type[content_pred_id]
                content_confidence = float(content_probs[content_pred_id])
                
                # Get sentiment prediction
                sentiment_probs = torch.nn.functional.softmax(outputs['sentiment_logits'], dim=-1).cpu().numpy()[0]
                sentiment_pred_id = np.argmax(sentiment_probs)
                sentiment_type = self.id_to_sentiment[sentiment_pred_id]
                sentiment_confidence = float(sentiment_probs[sentiment_pred_id])
            
            # Enhance sentiment with wrestling context
            context_enhancement = self.enhance_with_wrestling_context(text)
            
            # Adjust sentiment if context suggests different sentiment
            if abs(context_enhancement) > 0.1:
                if context_enhancement > 0.1 and sentiment_type != 'positive':
                    sentiment_type = 'positive'
                    sentiment_confidence = min(sentiment_confidence + abs(context_enhancement), 1.0)
                elif context_enhancement < -0.1 and sentiment_type != 'negative':
                    sentiment_type = 'negative'
                    sentiment_confidence = min(sentiment_confidence + abs(context_enhancement), 1.0)
            
            return content_type, content_confidence, sentiment_type, sentiment_confidence
            
        except Exception as e:
            logger.error(f"Multi-task analysis failed for text: {text[:50]}... Error: {e}")
            return "opinion", 0.0, "neutral", 0.0
    
    async def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment of a single text (for compatibility with existing API)."""
        try:
            _, _, sentiment_type, sentiment_confidence = await self.analyze_content_and_sentiment(text)
            
            # Convert sentiment type to score
            if sentiment_type == 'positive':
                sentiment_score = 0.5
            elif sentiment_type == 'negative':
                sentiment_score = -0.5
            else:
                sentiment_score = 0.0
            
            return sentiment_score, sentiment_confidence
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0, 0.0
    
    def classify_content_type(self, text: str) -> Tuple[str, float]:
        """Classify content type (for compatibility with weak labeling service)."""
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, run in thread pool
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    # Create new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.analyze_content_and_sentiment(text)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    content_type, content_confidence, _, _ = future.result(timeout=30)
                    return content_type, content_confidence
                    
            except RuntimeError:
                # No running loop, safe to use run_until_complete
                content_type, content_confidence, _, _ = asyncio.run(
                    self.analyze_content_and_sentiment(text)
                )
                return content_type, content_confidence
        except Exception as e:
            logger.error(f"Content classification failed: {e}")
            return "opinion", 0.0
    
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": "Fine-tuned Multi-task Wrestling Model",
            "base_model": settings.MODEL_NAME,
            "device": str(self.device),
            "model_loaded": self.model_loaded,
            "model_path": settings.FINE_TUNED_MODEL_PATH,
            "capabilities": ["content_type_classification", "sentiment_analysis", "multi_task"]
        }


# Singleton instance
fine_tuned_analyzer = FineTunedWrestlingSentimentAnalyzer()


async def analyze_wrestling_content(text: str) -> Tuple[str, float, str, float]:
    """Convenient function to analyze both content type and sentiment."""
    return await fine_tuned_analyzer.analyze_content_and_sentiment(text)