"""
Database cleanup service for storage optimization.
Automatically removes old posts and maintains efficient caching.
"""
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import delete, func
from app.core.database import get_db
from app.models.tweet import PostModel, TimelineModel, QueryModel
from app.core.logging import logger
from app.core.config import settings
from typing import Dict, Any


class DatabaseCleanupService:
    """Service for automatic database cleanup and storage optimization."""
    
    def __init__(self):
        self.retention_days = getattr(settings, 'DATA_RETENTION_DAYS', 60)
        self.cleanup_batch_size = getattr(settings, 'CLEANUP_BATCH_SIZE', 1000)
        self.image_cache_days = getattr(settings, 'IMAGE_CACHE_DAYS', 30)
    
    async def cleanup_old_posts(self, db: Session) -> Dict[str, int]:
        """Remove posts older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        try:
            # Count posts to be deleted
            old_posts_count = db.query(PostModel).filter(
                PostModel.datetime < cutoff_date
            ).count()
            
            if old_posts_count == 0:
                return {"deleted_posts": 0, "deleted_timeline": 0}
            
            logger.info(f"Cleaning up {old_posts_count} posts older than {self.retention_days} days")
            
            # Delete in batches to avoid memory issues
            deleted_posts = 0
            while True:
                # Get batch of old post IDs
                batch_ids = db.query(PostModel.id).filter(
                    PostModel.datetime < cutoff_date
                ).limit(self.cleanup_batch_size).all()
                
                if not batch_ids:
                    break
                
                # Delete batch
                batch_id_list = [id_tuple[0] for id_tuple in batch_ids]
                db.query(PostModel).filter(
                    PostModel.id.in_(batch_id_list)
                ).delete(synchronize_session=False)
                
                deleted_posts += len(batch_id_list)
                db.commit()
                
                # Small delay to prevent overwhelming the database
                await asyncio.sleep(0.1)
            
            # Clean up timeline data for deleted posts
            timeline_cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
            deleted_timeline = db.query(TimelineModel).filter(
                TimelineModel.timestamp < timeline_cutoff
            ).delete(synchronize_session=False)
            
            db.commit()
            
            logger.info(f"Cleanup completed: {deleted_posts} posts, {deleted_timeline} timeline entries")
            
            return {
                "deleted_posts": deleted_posts,
                "deleted_timeline": deleted_timeline
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            db.rollback()
            raise
    
    async def cleanup_expired_image_cache(self, db: Session) -> int:
        """Remove expired image cache entries."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.image_cache_days)
        
        try:
            # Clear expired image URLs
            expired_count = db.query(QueryModel).filter(
                QueryModel.image_cached_at < cutoff_date,
                QueryModel.image_url.isnot(None)
            ).update({
                QueryModel.image_url: None,
                QueryModel.image_cached_at: None,
                QueryModel.image_source: None
            }, synchronize_session=False)
            
            db.commit()
            
            if expired_count > 0:
                logger.info(f"Cleared {expired_count} expired image cache entries")
            
            return expired_count
            
        except Exception as e:
            logger.error(f"Image cache cleanup failed: {e}")
            db.rollback()
            raise
    
    async def optimize_database(self, db: Session) -> Dict[str, Any]:
        """Run database optimization tasks."""
        try:
            # Update query statistics
            queries_updated = await self._update_query_statistics(db)
            
            # Clean up orphaned timeline entries
            orphaned_cleaned = await self._cleanup_orphaned_timeline(db)
            
            # Vacuum database (for SQLite)
            if 'sqlite' in str(db.bind.url):
                db.execute("VACUUM")
                db.commit()
            
            return {
                "queries_updated": queries_updated,
                "orphaned_timeline_cleaned": orphaned_cleaned,
                "database_vacuumed": 'sqlite' in str(db.bind.url)
            }
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            raise
    
    async def _update_query_statistics(self, db: Session) -> int:
        """Update cached statistics for queries."""
        # Get all queries
        queries = db.query(QueryModel).all()
        updated_count = 0
        
        for query in queries:
            # Calculate current statistics
            stats = db.query(
                func.count(PostModel.id).label('post_count'),
                func.avg(PostModel.sentiment_score).label('avg_sentiment')
            ).filter(
                PostModel.query == query.query_text
            ).first()
            
            if stats.post_count != query.post_count or abs((stats.avg_sentiment or 0) - query.avg_sentiment) > 0.01:
                query.post_count = stats.post_count or 0
                query.avg_sentiment = stats.avg_sentiment or 0.0
                query.last_updated = datetime.utcnow()
                updated_count += 1
        
        if updated_count > 0:
            db.commit()
            logger.info(f"Updated statistics for {updated_count} queries")
        
        return updated_count
    
    async def _cleanup_orphaned_timeline(self, db: Session) -> int:
        """Remove timeline entries for non-existent queries."""
        orphaned_count = db.query(TimelineModel).filter(
            ~TimelineModel.query_id.in_(
                db.query(QueryModel.id)
            )
        ).delete(synchronize_session=False)
        
        if orphaned_count > 0:
            db.commit()
            logger.info(f"Cleaned up {orphaned_count} orphaned timeline entries")
        
        return orphaned_count
    
    async def get_storage_stats(self, db: Session) -> Dict[str, Any]:
        """Get current database storage statistics."""
        try:
            # Count records
            posts_count = db.query(func.count(PostModel.id)).scalar()
            queries_count = db.query(func.count(QueryModel.id)).scalar()
            timeline_count = db.query(func.count(TimelineModel.id)).scalar()
            
            # Count cached images
            cached_images = db.query(func.count(QueryModel.id)).filter(
                QueryModel.image_url.isnot(None)
            ).scalar()
            
            # Calculate data age
            oldest_post = db.query(func.min(PostModel.datetime)).scalar()
            newest_post = db.query(func.max(PostModel.datetime)).scalar()
            
            # Estimate storage usage (rough calculation)
            avg_post_size = 500  # bytes (estimated)
            estimated_storage_mb = (posts_count * avg_post_size) / (1024 * 1024)
            
            return {
                "posts_count": posts_count,
                "queries_count": queries_count,
                "timeline_count": timeline_count,
                "cached_images_count": cached_images,
                "oldest_post": oldest_post,
                "newest_post": newest_post,
                "estimated_storage_mb": round(estimated_storage_mb, 2),
                "retention_days": self.retention_days
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    async def run_full_cleanup(self, db: Session) -> Dict[str, Any]:
        """Run all cleanup operations."""
        logger.info("Starting full database cleanup...")
        
        results = {}
        
        # Cleanup old posts
        cleanup_results = await self.cleanup_old_posts(db)
        results.update(cleanup_results)
        
        # Cleanup image cache
        image_cleanup = await self.cleanup_expired_image_cache(db)
        results["expired_images_cleared"] = image_cleanup
        
        # Optimize database
        optimization_results = await self.optimize_database(db)
        results.update(optimization_results)
        
        # Get final stats
        storage_stats = await self.get_storage_stats(db)
        results["final_storage_stats"] = storage_stats
        
        logger.info(f"Full cleanup completed: {results}")
        
        return results


# Singleton instance
cleanup_service = DatabaseCleanupService()


async def run_scheduled_cleanup():
    """Run cleanup as a background task."""
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        results = await cleanup_service.run_full_cleanup(db)
        return results
    finally:
        db.close()


# Cleanup endpoint function
async def manual_cleanup_trigger():
    """Trigger manual cleanup (for admin use)."""
    return await run_scheduled_cleanup()