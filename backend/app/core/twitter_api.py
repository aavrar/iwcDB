import tweepy
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from app.core.config import settings
from app.core.logging import logger


class TwitterAPIClient:
    """Twitter API v2 client for authenticated access."""
    
    def __init__(self):
        self.client = None
        self.bearer_token = getattr(settings, 'TWITTER_BEARER_TOKEN', None)
        
    def init_client(self):
        """Initialize Twitter API client."""
        if not self.bearer_token:
            logger.error("Twitter Bearer Token not found. Set TWITTER_BEARER_TOKEN in environment.")
            return False
        
        try:
            self.client = tweepy.Client(bearer_token=self.bearer_token)
            logger.info("Twitter API client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API client: {e}")
            return False
    
    async def search_tweets(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search tweets using Twitter API v2."""
        if not self.client:
            if not self.init_client():
                return []
        
        try:
            # Twitter API v2 search
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=f"{query} lang:en -is:retweet",
                tweet_fields=['created_at', 'author_id', 'public_metrics'],
                user_fields=['username'],
                expansions=['author_id'],
                max_results=min(max_results, 100)  # API limit
            ).flatten(limit=max_results)
            
            # Process tweets
            processed_tweets = []
            for tweet in tweets:
                try:
                    # Get user info from includes
                    author_username = 'unknown'
                    if hasattr(tweet, 'includes') and 'users' in tweet.includes:
                        for user in tweet.includes['users']:
                            if user.id == tweet.author_id:
                                author_username = user.username
                                break
                    
                    processed_tweet = {
                        'id': str(tweet.id),
                        'content': tweet.text,
                        'username': author_username,
                        'datetime': tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
                        'url': f"https://twitter.com/{author_username}/status/{tweet.id}",
                        'metrics': {
                            'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                            'like_count': tweet.public_metrics.get('like_count', 0),
                            'reply_count': tweet.public_metrics.get('reply_count', 0),
                        } if hasattr(tweet, 'public_metrics') else {}
                    }
                    processed_tweets.append(processed_tweet)
                    
                except Exception as e:
                    logger.warning(f"Error processing tweet {tweet.id}: {e}")
                    continue
            
            logger.info(f"Twitter API search successful: {len(processed_tweets)} tweets")
            return processed_tweets
            
        except tweepy.TooManyRequests:
            logger.error("Twitter API rate limit exceeded")
            return []
        except tweepy.Unauthorized:
            logger.error("Twitter API unauthorized - check bearer token")
            return []
        except Exception as e:
            logger.error(f"Twitter API search failed: {e}")
            return []


# Twitter API costs (as of 2025):
# Basic Plan: $100/month - 10,000 tweets/month
# Pro Plan: $5,000/month - 1M tweets/month
# Enterprise: Custom pricing