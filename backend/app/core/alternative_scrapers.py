import asyncio
import aiohttp
import praw
from typing import List, Dict, Optional
from datetime import datetime
from app.core.config import settings
from app.core.logging import logger
from bs4 import BeautifulSoup
import re


class RedditScraper:
    """Reddit scraper for wrestling discussions."""
    
    def __init__(self):
        self.reddit = None
        self.client_id = getattr(settings, 'REDDIT_CLIENT_ID', None)
        self.client_secret = getattr(settings, 'REDDIT_CLIENT_SECRET', None)
        self.user_agent = f"{settings.APP_NAME}/1.0"
        
    def init_reddit(self):
        """Initialize Reddit client."""
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit credentials not found. Using read-only access.")
            try:
                # Read-only Reddit instance
                self.reddit = praw.Reddit(
                    client_id="dummy",
                    client_secret="dummy", 
                    user_agent=self.user_agent
                )
                return True
            except Exception as e:
                logger.error(f"Failed to initialize read-only Reddit: {e}")
                return False
        
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            logger.info("Reddit client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            return False
    
    async def search_posts(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search Reddit posts about wrestling."""
        if not self.reddit:
            if not self.init_reddit():
                return []
        
        try:
            # Wrestling subreddits
            subreddits = ['SquaredCircle', 'WWE', 'AEWOfficial', 'njpw', 'ROH', 'Wreddit']
            posts = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search in subreddit
                    search_results = subreddit.search(
                        query, 
                        sort='hot', 
                        time_filter='week',
                        limit=max_results // len(subreddits)
                    )
                    
                    for post in search_results:
                        post_data = {
                            'id': f"reddit_{post.id}",
                            'content': f"{post.title} {post.selftext}".strip(),
                            'username': str(post.author) if post.author else 'deleted',
                            'datetime': datetime.fromtimestamp(post.created_utc).isoformat(),
                            'url': f"https://reddit.com{post.permalink}",
                            'subreddit': subreddit_name,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio
                        }
                        posts.append(post_data)
                        
                        if len(posts) >= max_results:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error searching subreddit {subreddit_name}: {e}")
                    continue
            
            logger.info(f"Reddit search successful: {len(posts)} posts")
            return posts[:max_results]
            
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
            return []


class YouTubeScraper:
    """YouTube comments scraper for wrestling videos."""
    
    def __init__(self):
        self.api_key = getattr(settings, 'YOUTUBE_API_KEY', None)
        
    async def search_comments(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search YouTube comments about wrestling."""
        if not self.api_key:
            logger.warning("YouTube API key not found")
            return await self._scrape_without_api(query, max_results)
        
        try:
            async with aiohttp.ClientSession() as session:
                # Search for videos first
                search_url = "https://www.googleapis.com/youtube/v3/search"
                search_params = {
                    'key': self.api_key,
                    'q': f"{query} wrestling",
                    'part': 'snippet',
                    'type': 'video',
                    'maxResults': 5,
                    'order': 'relevance'
                }
                
                async with session.get(search_url, params=search_params) as response:
                    if response.status != 200:
                        logger.error(f"YouTube search failed: {response.status}")
                        return []
                    
                    search_data = await response.json()
                    video_ids = [item['id']['videoId'] for item in search_data.get('items', [])]
                
                # Get comments for each video
                comments = []
                for video_id in video_ids:
                    video_comments = await self._get_video_comments(session, video_id, max_results // len(video_ids))
                    comments.extend(video_comments)
                    
                    if len(comments) >= max_results:
                        break
                
                logger.info(f"YouTube search successful: {len(comments)} comments")
                return comments[:max_results]
                
        except Exception as e:
            logger.error(f"YouTube API search failed: {e}")
            return []
    
    async def _get_video_comments(self, session: aiohttp.ClientSession, video_id: str, max_results: int) -> List[Dict]:
        """Get comments for a specific video."""
        try:
            comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
            comments_params = {
                'key': self.api_key,
                'videoId': video_id,
                'part': 'snippet',
                'maxResults': max_results,
                'order': 'relevance'
            }
            
            async with session.get(comments_url, params=comments_params) as response:
                if response.status != 200:
                    return []
                
                comments_data = await response.json()
                comments = []
                
                for item in comments_data.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']
                    comment_data = {
                        'id': f"youtube_{item['id']}",
                        'content': comment['textDisplay'],
                        'username': comment['authorDisplayName'],
                        'datetime': comment['publishedAt'],
                        'url': f"https://youtube.com/watch?v={video_id}&lc={item['id']}",
                        'likes': comment.get('likeCount', 0)
                    }
                    comments.append(comment_data)
                
                return comments
                
        except Exception as e:
            logger.warning(f"Error getting comments for video {video_id}: {e}")
            return []
    
    async def _scrape_without_api(self, query: str, max_results: int) -> List[Dict]:
        """Fallback scraping without API (limited functionality)."""
        logger.warning("YouTube API not available, using limited scraping")
        # This would require more complex scraping and may violate ToS
        return []


class MastodonScraper:
    """Mastodon scraper for wrestling discussions."""
    
    async def search_toots(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search Mastodon toots about wrestling."""
        try:
            # Major Mastodon instances
            instances = ['mastodon.social', 'mas.to', 'hachyderm.io']
            toots = []
            
            async with aiohttp.ClientSession() as session:
                for instance in instances:
                    try:
                        search_url = f"https://{instance}/api/v2/search"
                        params = {
                            'q': f"{query} wrestling",
                            'type': 'statuses',
                            'limit': max_results // len(instances)
                        }
                        
                        async with session.get(search_url, params=params) as response:
                            if response.status != 200:
                                continue
                            
                            data = await response.json()
                            
                            for status in data.get('statuses', []):
                                toot_data = {
                                    'id': f"mastodon_{status['id']}",
                                    'content': BeautifulSoup(status['content'], 'html.parser').get_text(),
                                    'username': status['account']['username'],
                                    'datetime': status['created_at'],
                                    'url': status['url'],
                                    'instance': instance,
                                    'reblogs': status.get('reblogs_count', 0),
                                    'favourites': status.get('favourites_count', 0)
                                }
                                toots.append(toot_data)
                        
                    except Exception as e:
                        logger.warning(f"Error searching {instance}: {e}")
                        continue
            
            logger.info(f"Mastodon search successful: {len(toots)} toots")
            return toots[:max_results]
            
        except Exception as e:
            logger.error(f"Mastodon search failed: {e}")
            return []


class AlternativeDataAggregator:
    """Aggregates data from multiple alternative sources."""
    
    def __init__(self):
        self.reddit_scraper = RedditScraper()
        self.youtube_scraper = YouTubeScraper()
        self.mastodon_scraper = MastodonScraper()
    
    async def search_all_sources(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search across all alternative data sources."""
        results_per_source = max_results // 3
        
        # Run all searches concurrently
        reddit_task = self.reddit_scraper.search_posts(query, results_per_source)
        youtube_task = self.youtube_scraper.search_comments(query, results_per_source)
        mastodon_task = self.mastodon_scraper.search_toots(query, results_per_source)
        
        reddit_results, youtube_results, mastodon_results = await asyncio.gather(
            reddit_task, youtube_task, mastodon_task, return_exceptions=True
        )
        
        # Combine results
        all_results = []
        
        if isinstance(reddit_results, list):
            all_results.extend(reddit_results)
        if isinstance(youtube_results, list):
            all_results.extend(youtube_results)
        if isinstance(mastodon_results, list):
            all_results.extend(mastodon_results)
        
        # Sort by recency
        all_results.sort(key=lambda x: x.get('datetime', ''), reverse=True)
        
        logger.info(f"Alternative sources search: {len(all_results)} total results")
        return all_results[:max_results]


# Alternative data source costs:
# Reddit API: FREE (with rate limits)
# YouTube API: FREE tier available (10,000 requests/day)
# Mastodon: FREE (public APIs)