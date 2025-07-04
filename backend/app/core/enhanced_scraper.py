"""
Enhanced scraping service for Phase 2 multi-label training data collection.
Targets diverse wrestling content: news, rumors, and opinions.
"""
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from app.core.logging import logger
from app.core.config import settings
import re


@dataclass
class ScrapedPost:
    """Data structure for scraped wrestling posts."""
    id: str
    content: str
    username: str
    datetime: datetime
    subreddit: str
    url: str
    upvotes: int
    comments_count: int
    post_type_hint: str  # Initial classification hint
    confidence: float


class EnhancedWrestlingScraper:
    """Enhanced scraper for collecting diverse wrestling content types."""
    
    def __init__(self):
        self.session = None
        
        # Wrestling subreddits categorized by content type
        self.subreddit_sources = {
            'news_heavy': [
                'SquaredCircle',  # Mix but news-heavy
                'WWE',           # Official announcements
                'AEWOfficial',   # Company news
                'prowrestling',  # General wrestling news
                'NJPW'          # NJPW news
            ],
            'opinion_heavy': [
                'Wreddit',       # Fan discussions
                'WredditCountryClub',  # Opinion discussions
                'AEW',          # Fan opinions
                'IndyWrestling', # Indie wrestling discussions
                'SmackDown',    # Show discussions
                'RAW'           # Show discussions
            ],
            'rumor_heavy': [
                'SquaredCircle',  # Also has rumors
                'WWE',           # Backstage reports
                'AEWOfficial',   # Speculation
                'prowrestling'   # Wrestling rumors
            ]
        }
        
        # Wrestling news sources - if mentioned, very likely to be news
        self.news_sources = {
            "fightful.com", "fightfulselect.com", "wrestlingobserver.com", "wrestlingobserver.net",
            "pwinsider.com", "bodyslam.net", "prowrestling.net", "ringsidenews.com", "ringsidenews.net",
            "wrestlinginc.com", "wrestlinginc.net", "cagesideseats.com", "postwrestling.com",
            "wrestletalk.com", "wrestletalk.tv", "wrestlingnews.co", "sportskeeda.com/wwe", "sportskeeda.com",
            "411mania.com/wrestling", "411mania.com", "wrestling-edge.com", "dailywrestlingnews.com",
            "wrestlingheadlines.com", "bit.ly", "tinyurl.com"
        }
        
        # Patterns to identify content types
        self.content_patterns = {
            'news': [
                r'\b(announces?|confirmed?|officially|statement|released?)\b',
                r'\b(signs?|debuts?|returns?|injured?|surgery)\b',
                r'\b(wwe|aew|njpw|tna)\s+(announces?|confirms?)\b',
                r'\[official\]|\[confirmed\]|\[breaking\]',
                r'\b(press release|interview|backstage|report)\b',
                r'^EXCLUSIVE\b',  # Posts starting with EXCLUSIVE
                r'^Live\s+(SmackDown|RAW|NXT|Dynamite|Rampage|Collision|Impact).*Discussion.*Thread',  # Show discussion threads
                r'Discussion.*Thread.*\d{4}',  # Discussion threads with dates
            ],
            'rumor': [
                r'\b(rumou?rs?|reportedly|allegedly|sources?|backstage)\b',
                r'\b(might|could|possibly|speculation|unconfirmed)\b',
                r'\b(hearing|word is|sources say|apparently)\b',
                r'\[rumou?r\]|\[speculation\]|\[unconfirmed\]',
                r'\b(insider|dirt sheet|newsletter)\b'
            ],
            'opinion': [
                r'\b(think|feel|believe|opinion|imo|personally)\b',
                r'\b(love|hate|amazing|terrible|best|worst)\b',
                r'\b(should|would|hope|wish|want)\b',
                r'\b(unpopular opinion|hot take|change my mind)\b',
                r'^(am i|does anyone|why do)\b'
            ]
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': 'HeatMeter/1.0 Wrestling Sentiment Tracker'},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def classify_content_type(self, text: str, title: str = "") -> Tuple[str, float]:
        """
        Classify content type based on text patterns.
        Returns (content_type, confidence)
        """
        full_text = f"{title} {text}".lower()
        
        # First check for news sources - if found, very high confidence it's news
        news_source_mentions = 0
        for source in self.news_sources:
            if source.lower() in full_text:
                news_source_mentions += 1
        
        if news_source_mentions > 0:
            # High confidence news classification based on source citation
            confidence = min(0.95, 0.8 + (news_source_mentions * 0.05))  # Cap at 95%
            return 'news', confidence
        
        scores = {'news': 0, 'rumor': 0, 'opinion': 0}
        
        # Score based on pattern matches
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, full_text, re.IGNORECASE))
                scores[content_type] += matches
        
        # Additional heuristics
        if '?' in title:
            scores['opinion'] += 1
            
        if any(word in full_text for word in ['breaking', 'official', 'confirmed']):
            scores['news'] += 2
            
        if any(word in full_text for word in ['sources', 'hearing', 'apparently']):
            scores['rumor'] += 2
        
        # Determine winner
        if max(scores.values()) == 0:
            return 'opinion', 0.3  # Default to opinion with low confidence
        
        predicted_type = max(scores, key=scores.get)
        confidence = min(scores[predicted_type] / 5.0, 0.9)  # Cap at 0.9
        
        return predicted_type, confidence
    
    async def scrape_subreddit_posts(self, subreddit: str, limit: int = 100) -> List[ScrapedPost]:
        """Scrape posts from a specific subreddit."""
        if not self.session:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")
        
        posts = []
        url = f"https://www.reddit.com/r/{subreddit}/hot.json"
        
        try:
            async with self.session.get(url, params={'limit': limit}) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch r/{subreddit}: {response.status}")
                    return posts
                
                data = await response.json()
                
                for post_data in data.get('data', {}).get('children', []):
                    post = post_data.get('data', {})
                    
                    # Skip removed/deleted posts
                    if post.get('removed_by_category') or not post.get('selftext'):
                        continue
                    
                    # Extract post info
                    post_id = post.get('id', '')
                    title = post.get('title', '')
                    content = post.get('selftext', '')
                    
                    # Skip if no meaningful content
                    if len(content.strip()) < 20:
                        continue
                    
                    # Create post object
                    post_datetime = datetime.fromtimestamp(post.get('created_utc', 0))
                    
                    # Classify content type
                    content_type, confidence = self.classify_content_type(content, title)
                    
                    scraped_post = ScrapedPost(
                        id=post_id,
                        content=f"{title}\\n\\n{content}",
                        username=post.get('author', 'unknown'),
                        datetime=post_datetime,
                        subreddit=subreddit,
                        url=f"https://reddit.com{post.get('permalink', '')}",
                        upvotes=post.get('ups', 0),
                        comments_count=post.get('num_comments', 0),
                        post_type_hint=content_type,
                        confidence=confidence
                    )
                    
                    posts.append(scraped_post)
                    
                    # Small delay between processing
                    await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit}: {e}")
        
        return posts
    
    async def collect_training_data(self, target_count: int = 1000) -> Dict[str, List[ScrapedPost]]:
        """
        Collect balanced training data across content types.
        
        Args:
            target_count: Total posts to collect
            
        Returns:
            Dict with posts categorized by initial classification
        """
        logger.info(f"Starting enhanced data collection for {target_count} posts")
        
        # Target distribution
        target_per_type = target_count // 3
        collected = {'news': [], 'rumor': [], 'opinion': []}
        
        # Collect from all subreddits
        all_subreddits = set()
        for subreddit_list in self.subreddit_sources.values():
            all_subreddits.update(subreddit_list)
        
        for subreddit in all_subreddits:
            if sum(len(posts) for posts in collected.values()) >= target_count:
                break
            
            logger.info(f"Scraping r/{subreddit}...")
            posts = await self.scrape_subreddit_posts(subreddit, limit=250)
            
            # Categorize posts
            for post in posts:
                content_type = post.post_type_hint
                
                if len(collected[content_type]) < target_per_type:
                    collected[content_type].append(post)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        # Log collection results
        total_collected = sum(len(posts) for posts in collected.values())
        logger.info(f"Collection complete: {total_collected} posts")
        for content_type, posts in collected.items():
            logger.info(f"  {content_type}: {len(posts)} posts")
        
        return collected
    
    async def collect_high_confidence_examples(self, content_type: str, count: int = 50) -> List[ScrapedPost]:
        """Collect high-confidence examples of a specific content type."""
        logger.info(f"Collecting {count} high-confidence {content_type} examples")
        
        # Target specific subreddits for content type
        target_subreddits = {
            'news': ['WWE', 'AEWOfficial'],
            'rumor': ['SquaredCircle'],
            'opinion': ['Wreddit', 'SCJerk']
        }.get(content_type, ['SquaredCircle'])
        
        high_confidence_posts = []
        
        for subreddit in target_subreddits:
            if len(high_confidence_posts) >= count:
                break
            
            posts = await self.scrape_subreddit_posts(subreddit, limit=200)
            
            # Filter for high confidence matches
            for post in posts:
                if (post.post_type_hint == content_type and 
                    post.confidence > 0.6 and 
                    len(high_confidence_posts) < count):
                    high_confidence_posts.append(post)
            
            await asyncio.sleep(2)
        
        logger.info(f"Collected {len(high_confidence_posts)} high-confidence {content_type} posts")
        return high_confidence_posts
    
    def filter_quality_posts(self, posts: List[ScrapedPost], min_engagement: int = 5) -> List[ScrapedPost]:
        """Filter posts by quality metrics."""
        quality_posts = []
        
        for post in posts:
            # Quality criteria
            has_engagement = post.upvotes >= min_engagement or post.comments_count >= 3
            recent_enough = post.datetime > datetime.now() - timedelta(days=30)
            good_length = 50 <= len(post.content) <= 2000
            
            if has_engagement and recent_enough and good_length:
                quality_posts.append(post)
        
        logger.info(f"Quality filter: {len(quality_posts)}/{len(posts)} posts passed")
        return quality_posts


# Convenience functions
async def collect_diverse_training_data(target_count: int = 1000) -> Dict[str, List[ScrapedPost]]:
    """Collect diverse wrestling posts for training."""
    async with EnhancedWrestlingScraper() as scraper:
        return await scraper.collect_training_data(target_count)


async def collect_content_type_examples(content_type: str, count: int = 50) -> List[ScrapedPost]:
    """Collect high-confidence examples of specific content type."""
    async with EnhancedWrestlingScraper() as scraper:
        return await scraper.collect_high_confidence_examples(content_type, count)