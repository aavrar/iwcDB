"""
Phase 2 training data collection endpoint.
Scrapes diverse wrestling posts and exports to CSV for manual labeling.
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
import csv
import os
from datetime import datetime

from app.core.database import get_db
from app.core.enhanced_scraper import collect_diverse_training_data
from app.core.weak_labeling import generate_weak_labels_for_posts, get_posts_for_manual_review
from app.core.logging import logger

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post("/training/phase2/collect-data")
@limiter.limit("5/minute")
async def collect_phase2_training_data(
    request: Request,
    target_count: int = 1000,
    db: Session = Depends(get_db)
):
    """
    Collect diverse wrestling posts for Phase 2 training.
    Exports to CSV file for manual labeling.
    """
    try:
        logger.info(f"Starting Phase 2 data collection for {target_count} posts")
        
        # Collect diverse posts using enhanced scraper
        categorized_posts = await collect_diverse_training_data(target_count)
        
        # Flatten all posts into single list
        all_posts = []
        for content_type, posts in categorized_posts.items():
            all_posts.extend(posts)
        
        if not all_posts:
            raise HTTPException(status_code=404, detail="No posts collected")
        
        # Generate weak labels
        weak_labels = await generate_weak_labels_for_posts(all_posts)
        
        # Create posts lookup for easy access
        posts_lookup = {post.id: post for post in all_posts}
        
        # Get posts that need manual review
        review_candidates = get_posts_for_manual_review(weak_labels, limit=min(len(weak_labels), 500))
        
        # Export to CSV file in root directory
        csv_filename = f"wrestling_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_filepath = os.path.join("/Users/aahadvakani/IWCScraper", csv_filename)
        
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'post_id',
                'content',
                'username', 
                'subreddit',
                'datetime',
                'upvotes',
                'comments_count',
                'url',
                'predicted_content_type',
                'content_confidence',
                'predicted_sentiment',
                'sentiment_confidence',
                'needs_review',
                'manual_content_type',  # Empty for manual labeling
                'manual_sentiment',     # Empty for manual labeling
                'notes'                 # Empty for manual notes
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write posts that need review first (highest priority)
            written_ids = set()
            for weak_label in review_candidates:
                post = posts_lookup.get(weak_label.post_id)
                if post and post.id not in written_ids:
                    writer.writerow({
                        'post_id': post.id,
                        'content': post.content.replace('\n', ' ').replace('\r', ' '),
                        'username': post.username,
                        'subreddit': post.subreddit,
                        'datetime': post.datetime.isoformat(),
                        'upvotes': post.upvotes,
                        'comments_count': post.comments_count,
                        'url': post.url,
                        'predicted_content_type': weak_label.content_type,
                        'content_confidence': round(weak_label.content_confidence, 3),
                        'predicted_sentiment': weak_label.sentiment or '',
                        'sentiment_confidence': round(weak_label.sentiment_confidence, 3) if weak_label.sentiment_confidence else '',
                        'needs_review': weak_label.needs_review,
                        'manual_content_type': '',  # For manual labeling
                        'manual_sentiment': '',     # For manual labeling
                        'notes': ''                 # For manual notes
                    })
                    written_ids.add(post.id)
            
            # Write remaining posts
            for weak_label in weak_labels:
                post = posts_lookup.get(weak_label.post_id)
                if post and post.id not in written_ids:
                    writer.writerow({
                        'post_id': post.id,
                        'content': post.content.replace('\n', ' ').replace('\r', ' '),
                        'username': post.username,
                        'subreddit': post.subreddit,
                        'datetime': post.datetime.isoformat(),
                        'upvotes': post.upvotes,
                        'comments_count': post.comments_count,
                        'url': post.url,
                        'predicted_content_type': weak_label.content_type,
                        'content_confidence': round(weak_label.content_confidence, 3),
                        'predicted_sentiment': weak_label.sentiment or '',
                        'sentiment_confidence': round(weak_label.sentiment_confidence, 3) if weak_label.sentiment_confidence else '',
                        'needs_review': weak_label.needs_review,
                        'manual_content_type': '',
                        'manual_sentiment': '',
                        'notes': ''
                    })
                    written_ids.add(post.id)
        
        # Statistics
        total_posts = len(all_posts)
        review_count = len(review_candidates)
        content_type_stats = {}
        sentiment_stats = {}
        
        for weak_label in weak_labels:
            content_type_stats[weak_label.content_type] = content_type_stats.get(weak_label.content_type, 0) + 1
            if weak_label.sentiment:
                sentiment_stats[weak_label.sentiment] = sentiment_stats.get(weak_label.sentiment, 0) + 1
        
        logger.info(f"Phase 2 data collection complete: {total_posts} posts exported to {csv_filename}")
        
        return {
            "status": "success",
            "message": f"Training data exported to {csv_filename}",
            "csv_file": csv_filename,
            "csv_path": csv_filepath,
            "statistics": {
                "total_posts": total_posts,
                "posts_needing_review": review_count,
                "content_type_distribution": content_type_stats,
                "sentiment_distribution": sentiment_stats
            },
            "instructions": {
                "manual_labeling": [
                    "1. Open the CSV file in Excel/Google Sheets",
                    "2. Focus on posts where 'needs_review' is True first",
                    "3. Fill in 'manual_content_type' column with: news, rumor, or opinion",
                    "4. For opinion posts, fill 'manual_sentiment' with: positive, negative, or neutral",
                    "5. Add any notes in the 'notes' column",
                    "6. Save the file when complete"
                ],
                "content_type_examples": {
                    "news": "WWE announces new champion, Wrestler signs contract, Injury report",
                    "rumor": "Sources say, Backstage reports, Allegedly, Speculation about",
                    "opinion": "I think, Best/worst, Should/shouldn't, Personal take"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Phase 2 data collection error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error collecting training data: {str(e)}"
        )


@router.post("/training/phase2/train-model")
@limiter.limit("2/minute")
async def train_phase2_model(
    request: Request,
    csv_filename: str,
    db: Session = Depends(get_db)
):
    """Train Phase 2 model on labeled CSV data."""
    try:
        csv_path = os.path.join("/Users/aahadvakani/IWCScraper", csv_filename)
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"CSV file not found: {csv_filename}")
        
        logger.info(f"Starting Phase 2 model training with {csv_filename}")
        
        # Import here to avoid circular imports
        from app.core.fine_tuning import train_wrestling_model
        
        # Train model
        results = train_wrestling_model(csv_path)
        
        return {
            "status": "success",
            "message": "Phase 2 model training completed",
            "csv_file": csv_filename,
            "training_results": results,
            "model_path": "./wrestling_fine_tuned_model"
        }
        
    except Exception as e:
        logger.error(f"Phase 2 training error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )


@router.post("/training/phase2/active-learning")
@limiter.limit("3/minute")
async def run_active_learning(
    request: Request,
    iteration: int,
    db: Session = Depends(get_db)
):
    """Run active learning iteration for continuous improvement."""
    try:
        logger.info(f"Starting active learning iteration {iteration}")
        
        # Import here to avoid circular imports
        from app.core.active_learning import run_active_learning_iteration
        
        # Run active learning iteration
        results = await run_active_learning_iteration(iteration)
        
        return {
            "status": "success",
            "message": f"Active learning iteration {iteration} completed",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Active learning error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running active learning: {str(e)}"
        )


@router.get("/training/phase2/status")
async def get_phase2_status():
    """Get status of Phase 2 training data collection."""
    try:
        # Check for existing CSV files
        root_dir = "/Users/aahadvakani/IWCScraper"
        csv_files = [f for f in os.listdir(root_dir) if f.startswith('wrestling_training_data_') and f.endswith('.csv')]
        active_learning_files = [f for f in os.listdir(root_dir) if f.startswith('active_learning_iteration_') and f.endswith('.csv')]
        
        csv_files.sort(reverse=True)  # Most recent first
        active_learning_files.sort(reverse=True)
        
        return {
            "status": "ready",
            "training_data_files": csv_files[:5],  # Show last 5 files
            "active_learning_files": active_learning_files[:3],
            "available_endpoints": {
                "collect_data": "POST /training/phase2/collect-data",
                "train_model": "POST /training/phase2/train-model",
                "active_learning": "POST /training/phase2/active-learning"
            },
            "instructions": [
                "1. Collect training data: POST to /training/phase2/collect-data",
                "2. Label the generated CSV file manually",
                "3. Train model: POST to /training/phase2/train-model with csv_filename",
                "4. Run active learning: POST to /training/phase2/active-learning with iteration number"
            ]
        }
        
    except Exception as e:
        logger.error(f"Phase 2 status error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status: {str(e)}"
        )