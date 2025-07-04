from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.real_image_scraper import image_scraper
from app.models.tweet import QueryModel
from app.core.logging import logger


class ImageCacheManager:
    """Manages image URL caching in the database."""
    
    def __init__(self, cache_duration_days: int = 30):
        self.cache_duration_days = cache_duration_days
    
    async def get_cached_image(self, wrestler_name: str, db: Session) -> Optional[str]:
        """Get cached image URL if it's still valid."""
        try:
            query_model = db.query(QueryModel).filter(
                QueryModel.query_text == wrestler_name
            ).first()
            
            if query_model and query_model.image_url and query_model.image_cached_at:
                # Check if cache is still valid
                cache_expiry = query_model.image_cached_at + timedelta(days=self.cache_duration_days)
                if datetime.utcnow() < cache_expiry:
                    logger.debug(f"Using cached image for {wrestler_name}")
                    return query_model.image_url
                else:
                    logger.debug(f"Image cache expired for {wrestler_name}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached image for {wrestler_name}: {e}")
            return None
    
    async def cache_image_url(self, wrestler_name: str, image_url: str, source: str, db: Session) -> bool:
        """Cache image URL in the database."""
        try:
            # Get or create query model
            query_model = db.query(QueryModel).filter(
                QueryModel.query_text == wrestler_name
            ).first()
            
            if not query_model:
                query_model = QueryModel(
                    query_text=wrestler_name,
                    post_count=0,
                    avg_sentiment=0.0
                )
                db.add(query_model)
            
            # Update image cache fields
            query_model.image_url = image_url
            query_model.image_cached_at = datetime.utcnow()
            query_model.image_source = source
            
            db.commit()
            logger.info(f"Cached image for {wrestler_name} from {source}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching image for {wrestler_name}: {e}")
            db.rollback()
            return False
    
    async def get_wrestler_image_with_cache(self, wrestler_name: str, db: Session) -> Optional[str]:
        """Get wrestler image with caching - main public method."""
        try:
            # First try to get from cache
            cached_url = await self.get_cached_image(wrestler_name, db)
            if cached_url:
                return cached_url
            
            # If not cached or expired, fetch new image
            logger.info(f"Fetching new image for {wrestler_name}")
            image_url = await image_scraper.get_wrestler_image(wrestler_name)
            
            if image_url:
                # Determine source based on URL
                source = self._determine_image_source(image_url)
                
                # Cache the new image URL
                await self.cache_image_url(wrestler_name, image_url, source, db)
                
                return image_url
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting wrestler image with cache for {wrestler_name}: {e}")
            return None
    
    def _determine_image_source(self, image_url: str) -> str:
        """Determine the source of an image URL."""
        if 'wikipedia' in image_url or 'wikimedia' in image_url:
            return 'wikipedia'
        elif 'cagematch' in image_url:
            return 'cagematch'
        elif 'placeholder' in image_url:
            return 'placeholder'
        else:
            return 'unknown'
    
    async def invalidate_cache(self, wrestler_name: str, db: Session) -> bool:
        """Invalidate cached image for a wrestler."""
        try:
            query_model = db.query(QueryModel).filter(
                QueryModel.query_text == wrestler_name
            ).first()
            
            if query_model:
                query_model.image_url = None
                query_model.image_cached_at = None
                query_model.image_source = None
                db.commit()
                logger.info(f"Invalidated image cache for {wrestler_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {wrestler_name}: {e}")
            db.rollback()
            return False
    
    async def get_cache_stats(self, db: Session) -> dict:
        """Get statistics about the image cache."""
        try:
            total_cached = db.query(QueryModel).filter(
                QueryModel.image_url.isnot(None)
            ).count()
            
            # Count by source
            wikipedia_count = db.query(QueryModel).filter(
                QueryModel.image_source == 'wikipedia'
            ).count()
            
            cagematch_count = db.query(QueryModel).filter(
                QueryModel.image_source == 'cagematch'
            ).count()
            
            placeholder_count = db.query(QueryModel).filter(
                QueryModel.image_source == 'placeholder'
            ).count()
            
            # Count expired entries
            expiry_date = datetime.utcnow() - timedelta(days=self.cache_duration_days)
            expired_count = db.query(QueryModel).filter(
                QueryModel.image_cached_at < expiry_date
            ).count()
            
            return {
                'total_cached_images': total_cached,
                'wikipedia_images': wikipedia_count,
                'cagematch_images': cagematch_count,
                'placeholder_images': placeholder_count,
                'expired_entries': expired_count,
                'cache_duration_days': self.cache_duration_days
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# Global instance
image_cache_manager = ImageCacheManager()