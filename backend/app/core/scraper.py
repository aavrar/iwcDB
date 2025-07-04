import asyncio
import random
import json
from typing import List, Dict, Optional
from datetime import datetime
import aiohttp
import time
from app.core.config import settings
from app.core.logging import logger
from urllib.parse import quote


class RedditScraper:
    """Reddit scraper for wrestling content."""
    
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
    
    async def scrape_reddit_posts(self, query: str, max_results: int = 100) -> List[Dict]:
        """Scrape Reddit posts from wrestling subreddits."""
        try:
            all_posts = []
            subreddits = [
                'SquaredCircle',     # Main wrestling discussion (2.7M members)
                'WWE',               # WWE official (1.2M members)  
                'AEWOfficial',       # AEW official (400K members)
                'njpw',              # New Japan Pro Wrestling (200K members)
                'ROH',               # Ring of Honor (50K members)
                'Wreddit',           # Wrestling discussion (100K members)
                'WredditCountryClub', # Smaller wrestling community (50K members)
                'prowrestling',      # General pro wrestling (300K members)
                'stardomjoshi',      # Women's wrestling (50K members)
                'indiewrestling',    # Independent wrestling (80K members)
                'MLW',               # Major League Wrestling (20K members)
                'IMPACTWRESTLING'    # Impact Wrestling (60K members)
            ]
            
            posts_per_subreddit = max(max_results // len(subreddits), 15)
            
            async with aiohttp.ClientSession(
                headers={'User-Agent': random.choice(self.user_agents)},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                for subreddit in subreddits:
                    try:
                        url = f"https://www.reddit.com/r/{subreddit}/search.json"
                        params = {
                            'q': query,
                            'restrict_sr': 'on',
                            'sort': 'new',
                            'limit': posts_per_subreddit
                        }
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                posts = []
                                
                                if 'data' in data and 'children' in data['data']:
                                    for child in data['data']['children']:
                                        if child.get('kind') == 't3':  # Post type
                                            post_data = child.get('data', {})
                                            
                                            # Extract text content
                                            content = post_data.get('selftext', '') or post_data.get('title', '')
                                            if content and len(content.strip()) > 10:
                                                posts.append({
                                                    'id': f"reddit_{post_data.get('id')}",
                                                    'content': content.strip(),
                                                    'title': post_data.get('title', ''),
                                                    'score': post_data.get('score', 0),
                                                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                                    'source': 'reddit',
                                                    'subreddit': subreddit,
                                                    'created_at': datetime.fromtimestamp(post_data.get('created_utc', time.time())).isoformat(),
                                                    'author': post_data.get('author', 'unknown')
                                                })
                                
                                all_posts.extend(posts)
                                logger.debug(f"Reddit r/{subreddit}: {len(posts)} posts")
                            else:
                                logger.warning(f"Reddit r/{subreddit} returned status {response.status}")
                        
                        # Small delay between subreddit requests
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.warning(f"Error scraping r/{subreddit}: {e}")
                        continue
            
            # Remove duplicates based on content
            unique_posts = []
            seen_content = set()
            
            for post in all_posts:
                content_key = post['content'][:100].lower()
                if content_key not in seen_content:
                    unique_posts.append(post)
                    seen_content.add(content_key)
            
            logger.info(f"Reddit scraping successful: {len(unique_posts)} unique posts from {len(subreddits)} subreddits")
            return unique_posts[:max_results]
            
        except Exception as e:
            logger.error(f"Reddit scraping failed: {e}")
            return []
    
    async def scrape_posts(self, query: str, max_results: int = 50) -> List[Dict]:
        """Main scraping method - uses Reddit as primary source."""
        logger.info(f"Starting Reddit scraping for query: '{query}'")
        
        try:
            tweets = await self.scrape_reddit_posts(query, max_results)
            
            if tweets:
                logger.info(f"Successfully scraped {len(tweets)} posts")
                return tweets
            else:
                logger.warning("No posts found")
                return []
                
        except Exception as e:
            logger.error(f"Scraping failed completely: {e}")
            return []


# Create a scraper instance
scraper = RedditScraper()

# For backwards compatibility
TwitterScraper = RedditScraper