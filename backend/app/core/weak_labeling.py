"""
Multi-model weak labeling service for content type and sentiment classification.
Uses current models to generate initial labels with confidence scores.
"""
import asyncio
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import re
import numpy as np

from app.core.nlp import sentiment_analyzer
from app.core.fine_tuned_nlp import fine_tuned_analyzer
from app.core.enhanced_scraper import ScrapedPost
from app.core.logging import logger
from app.core.config import settings


@dataclass
class WeakLabel:
    """Weak label with confidence scores."""
    post_id: str
    content_type: str  # news, rumor, opinion
    content_confidence: float
    sentiment: Optional[str]  # positive, negative, neutral (only for opinions)
    sentiment_confidence: Optional[float]
    needs_review: bool  # True if confidence is low


class WeakLabelingService:
    """Service for generating weak labels using multiple classification models."""
    
    def __init__(self):
        self.content_type_patterns = {
            'news': [
                r'\b(announces?|confirmed?|officially|statement|released?)\b',
                r'\b(signs?|debuts?|returns?|injured?|surgery)\b',
                r'\b(breaking|official|press release)\b',
                r'\b(wwe|aew|njpw|tna)\s+(announces?|confirms?)\b',
                r'^EXCLUSIVE\b',  # Posts starting with EXCLUSIVE
                r'^Live\s+(SmackDown|RAW|NXT|Dynamite|Rampage|Collision|Impact).*Discussion.*Thread',  # Show discussion threads
                r'Discussion.*Thread.*\d{4}',  # Discussion threads with dates
            ],
            'rumor': [
                r'\b(rumou?rs?|reportedly|allegedly|sources?|backstage)\b',
                r'\b(might|could|possibly|speculation|unconfirmed)\b',
                r'\b(hearing|word is|sources say|apparently|insider)\b',
                r'\[rumou?r\]|\[speculation\]|\[unconfirmed\]'
            ],
            'opinion': [
                r'\b(think|feel|believe|opinion|imo|personally)\b',
                r'\b(love|hate|amazing|terrible|best|worst)\b',
                r'\b(should|would|hope|wish|want)\b',
                r'^(am i|does anyone|why do|unpopular opinion)\b'
            ]
        }
        
        # News/rumor distinguishing patterns
        self.news_certainty_indicators = [
            'official', 'confirmed', 'announced', 'press release', 
            'statement', 'breaking', 'reports'
        ]
        
        self.rumor_uncertainty_indicators = [
            'rumor', 'allegedly', 'reportedly', 'sources say',
            'hearing', 'might', 'could', 'speculation'
        ]
        
        # Wrestling news sources - if mentioned, very likely to be news
        self.news_sources = {
            "fightful.com",
            "fightfulselect.com", 
            "wrestlingobserver.com",
            "wrestlingobserver.net",
            "pwinsider.com",
            "bodyslam.net",
            "prowrestling.net",
            "ringsidenews.com",
            "ringsidenews.net",
            "wrestlinginc.com",
            "wrestlinginc.net",
            "cagesideseats.com",
            "postwrestling.com",
            "wrestletalk.com",
            "wrestletalk.tv",
            "wrestlingnews.co",
            "sportskeeda.com/wwe",
            "sportskeeda.com",
            "411mania.com/wrestling",
            "411mania.com",
            "wrestling-edge.com",
            "dailywrestlingnews.com",
            "wrestlingheadlines.com",
            # Common URL shorteners or redirects
            "bit.ly",
            "tinyurl.com",
        }
    
    def classify_content_type(self, text: str) -> Tuple[str, float]:
        """
        Classify content type using fine-tuned model or pattern matching.
        Returns (content_type, confidence)
        """
        # Try using fine-tuned model first
        if settings.USE_FINE_TUNED_MODEL:
            try:
                return fine_tuned_analyzer.classify_content_type(text)
            except Exception as e:
                logger.warning(f"Fine-tuned model failed, falling back to rule-based: {e}")
        
        # Fallback to rule-based classification
        text_lower = text.lower()
        
        # First check for news sources - if found, very high confidence it's news
        news_source_mentions = 0
        for source in self.news_sources:
            if source.lower() in text_lower:
                news_source_mentions += 1
        
        if news_source_mentions > 0:
            # High confidence news classification based on source citation
            confidence = min(0.95, 0.8 + (news_source_mentions * 0.05))  # Cap at 95%
            return 'news', confidence
        
        # Count pattern matches for other classification
        scores = {}
        for content_type, patterns in self.content_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[content_type] = score
        
        # Special logic for news vs rumor disambiguation
        news_indicators = sum(1 for indicator in self.news_certainty_indicators 
                             if indicator in text_lower)
        rumor_indicators = sum(1 for indicator in self.rumor_uncertainty_indicators 
                              if indicator in text_lower)
        
        if news_indicators > rumor_indicators:
            scores['news'] += 2
        elif rumor_indicators > news_indicators:
            scores['rumor'] += 2
        
        # Additional heuristics
        if text.count('?') > 1:  # Multiple questions suggest opinion
            scores['opinion'] += 1
        
        if any(phrase in text_lower for phrase in ['what do you think', 'change my mind', 'unpopular opinion']):
            scores['opinion'] += 3
        
        # Determine winner
        if max(scores.values()) == 0:
            return 'opinion', 0.2  # Default with low confidence
        
        predicted_type = max(scores, key=scores.get)
        max_score = scores[predicted_type]
        total_score = sum(scores.values())
        
        # Calculate confidence
        if total_score == 0:
            confidence = 0.2
        else:
            confidence = min(max_score / max(total_score, 1), 0.8)  # Cap at 0.8 for weak labeling
        
        return predicted_type, confidence
    
    async def generate_sentiment_label(self, text: str) -> Tuple[str, float]:
        """
        Generate sentiment label using fine-tuned model or base sentiment analyzer.
        Returns (sentiment, confidence)
        """
        # Try using fine-tuned model first
        if settings.USE_FINE_TUNED_MODEL:
            try:
                _, _, sentiment_type, sentiment_confidence = await fine_tuned_analyzer.analyze_content_and_sentiment(text)
                return sentiment_type, sentiment_confidence
            except Exception as e:
                logger.warning(f"Fine-tuned sentiment failed, falling back to base model: {e}")
        
        # Fallback to base sentiment analyzer
        try:
            sentiment_score, confidence = await sentiment_analyzer.analyze_sentiment(text)
            
            # Convert score to label
            if sentiment_score > 0.1:
                sentiment_label = 'positive'
            elif sentiment_score < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return sentiment_label, confidence
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 'neutral', 0.1
    
    async def generate_weak_labels(self, posts: List[ScrapedPost]) -> List[WeakLabel]:
        """
        Generate weak labels for a list of posts.
        """
        logger.info(f"Generating weak labels for {len(posts)} posts")
        
        weak_labels = []
        
        for i, post in enumerate(posts):
            # Classify content type
            content_type, content_confidence = self.classify_content_type(post.content)
            
            # Generate sentiment label only for opinions
            sentiment = None
            sentiment_confidence = None
            
            if content_type == 'opinion':
                sentiment, sentiment_confidence = await self.generate_sentiment_label(post.content)
            
            # Determine if needs manual review
            needs_review = (
                content_confidence < 0.5 or  # Low content classification confidence
                (content_type == 'opinion' and sentiment_confidence and sentiment_confidence < 0.6) or  # Low sentiment confidence
                content_type in ['news', 'rumor'] and content_confidence < 0.7  # Uncertain news/rumor distinction
            )
            
            weak_label = WeakLabel(
                post_id=post.id,
                content_type=content_type,
                content_confidence=content_confidence,
                sentiment=sentiment,
                sentiment_confidence=sentiment_confidence,
                needs_review=needs_review
            )
            
            weak_labels.append(weak_label)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(posts)} posts")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        # Log statistics
        content_type_counts = {}
        sentiment_counts = {}
        needs_review_count = 0
        
        for label in weak_labels:
            content_type_counts[label.content_type] = content_type_counts.get(label.content_type, 0) + 1
            if label.sentiment:
                sentiment_counts[label.sentiment] = sentiment_counts.get(label.sentiment, 0) + 1
            if label.needs_review:
                needs_review_count += 1
        
        logger.info(f"Weak labeling complete:")
        logger.info(f"  Content types: {content_type_counts}")
        logger.info(f"  Sentiments: {sentiment_counts}")
        logger.info(f"  Needs review: {needs_review_count}/{len(weak_labels)} ({needs_review_count/len(weak_labels)*100:.1f}%)")
        
        return weak_labels
    
    def get_review_candidates(self, weak_labels: List[WeakLabel], limit: int = 200) -> List[WeakLabel]:
        """
        Get the most important posts for manual review.
        Prioritizes uncertain classifications and balanced representation.
        """
        # Separate by review need
        needs_review = [label for label in weak_labels if label.needs_review]
        
        # Sort by uncertainty (lowest confidence first)
        needs_review.sort(key=lambda x: min(
            x.content_confidence,
            x.sentiment_confidence or 1.0
        ))
        
        # Ensure balanced representation across content types
        review_candidates = []
        content_type_targets = {'news': limit // 4, 'rumor': limit // 4, 'opinion': limit // 2}
        content_type_counts = {'news': 0, 'rumor': 0, 'opinion': 0}
        
        # First pass: fill targets
        for label in needs_review:
            if len(review_candidates) >= limit:
                break
            
            content_type = label.content_type
            if content_type_counts[content_type] < content_type_targets[content_type]:
                review_candidates.append(label)
                content_type_counts[content_type] += 1
        
        # Second pass: fill remaining slots
        for label in needs_review:
            if len(review_candidates) >= limit:
                break
            
            if label not in review_candidates:
                review_candidates.append(label)
        
        logger.info(f"Selected {len(review_candidates)} posts for manual review")
        logger.info(f"  Review distribution: {content_type_counts}")
        
        return review_candidates
    
    def get_high_confidence_labels(self, weak_labels: List[WeakLabel], 
                                  min_content_confidence: float = 0.7,
                                  min_sentiment_confidence: float = 0.7) -> List[WeakLabel]:
        """
        Get high-confidence labels that can be used directly for training.
        """
        high_confidence = []
        
        for label in weak_labels:
            # Check content type confidence
            if label.content_confidence >= min_content_confidence:
                # For opinions, also check sentiment confidence
                if label.content_type == 'opinion':
                    if (label.sentiment_confidence and 
                        label.sentiment_confidence >= min_sentiment_confidence):
                        high_confidence.append(label)
                else:
                    # News and rumors don't need sentiment confidence
                    high_confidence.append(label)
        
        logger.info(f"Found {len(high_confidence)} high-confidence labels")
        return high_confidence


# Singleton instance
weak_labeling_service = WeakLabelingService()


# Convenience functions
async def generate_weak_labels_for_posts(posts: List[ScrapedPost]) -> List[WeakLabel]:
    """Generate weak labels for scraped posts."""
    return await weak_labeling_service.generate_weak_labels(posts)


def get_posts_for_manual_review(weak_labels: List[WeakLabel], limit: int = 200) -> List[WeakLabel]:
    """Get posts that need manual review."""
    return weak_labeling_service.get_review_candidates(weak_labels, limit)


def get_auto_training_labels(weak_labels: List[WeakLabel]) -> List[WeakLabel]:
    """Get high-confidence labels for automatic training."""
    return weak_labeling_service.get_high_confidence_labels(weak_labels)