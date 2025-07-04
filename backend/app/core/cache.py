import json
import pickle
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta
import redis
from app.core.database import get_redis
from app.core.config import settings
from app.core.logging import logger
import hashlib


class CacheManager:
    """Redis-based cache manager for tweets and sentiment data."""
    
    def __init__(self):
        self.redis_client = get_redis()
        self.default_ttl = 3600  # 1 hour
        self.prefix = "iwc_sentiment:"
    
    def _get_key(self, key: str) -> str:
        """Generate cache key with prefix."""
        return f"{self.prefix}{key}"
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for Redis storage."""
        if isinstance(data, (dict, list)):
            return json.dumps(data, default=str)
        return str(data)
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from Redis."""
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    
    def _generate_query_key(self, query: str, params: Dict = None) -> str:
        """Generate a unique cache key for a query."""
        key_data = {"query": query.lower().strip()}
        if params:
            key_data.update(params)
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"query:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(self._get_key(key))
            if cached_data:
                return self._deserialize_data(cached_data)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set data in cache."""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized_data = self._serialize_data(value)
            result = self.redis_client.setex(
                self._get_key(key),
                ttl,
                serialized_data
            )
            return result
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete data from cache."""
        if not self.redis_client:
            return False
        
        try:
            result = self.redis_client.delete(self._get_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.redis_client:
            return False
        
        try:
            return self.redis_client.exists(self._get_key(key)) > 0
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def get_tweets_for_query(self, query: str, max_results: int = 50) -> Optional[List[Dict]]:
        """Get cached tweets for a query."""
        key = self._generate_query_key(query, {"max_results": max_results})
        return await self.get(key)
    
    async def cache_tweets_for_query(self, query: str, tweets: List[Dict], max_results: int = 50, ttl: int = 1800) -> bool:
        """Cache tweets for a query."""
        key = self._generate_query_key(query, {"max_results": max_results})
        cache_data = {
            "tweets": tweets,
            "cached_at": datetime.utcnow().isoformat(),
            "query": query,
            "count": len(tweets)
        }
        return await self.set(key, cache_data, ttl)
    
    async def get_sentiment_for_query(self, query: str) -> Optional[Dict]:
        """Get cached sentiment analysis for a query."""
        key = f"sentiment:{self._generate_query_key(query)}"
        return await self.get(key)
    
    async def cache_sentiment_for_query(self, query: str, sentiment_data: Dict, ttl: int = 3600) -> bool:
        """Cache sentiment analysis for a query."""
        key = f"sentiment:{self._generate_query_key(query)}"
        cache_data = {
            **sentiment_data,
            "cached_at": datetime.utcnow().isoformat(),
            "query": query
        }
        return await self.set(key, cache_data, ttl)
    
    async def get_timeline_for_query(self, query: str) -> Optional[List[Dict]]:
        """Get cached timeline data for a query."""
        key = f"timeline:{self._generate_query_key(query)}"
        return await self.get(key)
    
    async def cache_timeline_for_query(self, query: str, timeline_data: List[Dict], ttl: int = 1800) -> bool:
        """Cache timeline data for a query."""
        key = f"timeline:{self._generate_query_key(query)}"
        cache_data = {
            "timeline": timeline_data,
            "cached_at": datetime.utcnow().isoformat(),
            "query": query
        }
        return await self.set(key, cache_data, ttl)
    
    async def get_autocomplete_suggestions(self, prefix: str) -> Optional[List[Dict]]:
        """Get cached autocomplete suggestions."""
        key = f"autocomplete:{prefix.lower()}"
        return await self.get(key)
    
    async def cache_autocomplete_suggestions(self, prefix: str, suggestions: List[Dict], ttl: int = 7200) -> bool:
        """Cache autocomplete suggestions."""
        key = f"autocomplete:{prefix.lower()}"
        return await self.set(key, suggestions, ttl)
    
    async def increment_query_count(self, query: str) -> int:
        """Increment query usage count."""
        if not self.redis_client:
            return 0
        
        try:
            key = f"query_count:{query.lower()}"
            return self.redis_client.incr(self._get_key(key))
        except Exception as e:
            logger.error(f"Error incrementing query count: {e}")
            return 0
    
    async def get_popular_queries(self, limit: int = 10) -> List[Dict]:
        """Get popular queries based on usage count."""
        if not self.redis_client:
            return []
        
        try:
            pattern = self._get_key("query_count:*")
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return []
            
            # Get all counts
            counts = self.redis_client.mget(keys)
            
            # Create query-count pairs
            query_counts = []
            for key, count in zip(keys, counts):
                if count:
                    query = key.decode().replace(self._get_key("query_count:"), "")
                    query_counts.append({"query": query, "count": int(count)})
            
            # Sort by count and return top queries
            query_counts.sort(key=lambda x: x["count"], reverse=True)
            return query_counts[:limit]
            
        except Exception as e:
            logger.error(f"Error getting popular queries: {e}")
            return []
    
    async def clear_cache_for_query(self, query: str) -> bool:
        """Clear all cached data for a specific query."""
        if not self.redis_client:
            return False
        
        try:
            query_key = self._generate_query_key(query)
            keys_to_delete = [
                self._get_key(query_key),
                self._get_key(f"sentiment:{query_key}"),
                self._get_key(f"timeline:{query_key}")
            ]
            
            deleted_count = self.redis_client.delete(*keys_to_delete)
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error clearing cache for query {query}: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.redis_client:
            return {"status": "disconnected"}
        
        try:
            info = self.redis_client.info()
            pattern = self._get_key("*")
            total_keys = len(self.redis_client.keys(pattern))
            
            return {
                "status": "connected",
                "total_keys": total_keys,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def flush_all_cache(self) -> bool:
        """Flush all cache data (use with caution)."""
        if not self.redis_client:
            return False
        
        try:
            pattern = self._get_key("*")
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info(f"Flushed {deleted_count} keys from cache")
                return deleted_count > 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False


# Singleton instance
cache_manager = CacheManager()


# Convenience functions
async def get_cached_tweets(query: str, max_results: int = 50) -> Optional[List[Dict]]:
    """Get cached tweets for a query."""
    return await cache_manager.get_tweets_for_query(query, max_results)


async def cache_tweets(query: str, tweets: List[Dict], max_results: int = 50, ttl: int = 1800) -> bool:
    """Cache tweets for a query."""
    return await cache_manager.cache_tweets_for_query(query, tweets, max_results, ttl)


async def get_cached_sentiment(query: str) -> Optional[Dict]:
    """Get cached sentiment analysis for a query."""
    return await cache_manager.get_sentiment_for_query(query)


async def cache_sentiment(query: str, sentiment_data: Dict, ttl: int = 3600) -> bool:
    """Cache sentiment analysis for a query."""
    return await cache_manager.cache_sentiment_for_query(query, sentiment_data, ttl)