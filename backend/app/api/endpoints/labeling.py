"""
Web-based labeling interface endpoints.
Load CSV data and provide easy labeling workflow.
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
import pandas as pd
import os
import json
from datetime import datetime

from app.core.database import get_db
from app.core.logging import logger

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.get("/labeling/csv-files")
async def get_available_csv_files():
    """Get list of available CSV files for labeling."""
    try:
        root_dir = "/Users/aahadvakani/IWCScraper"
        csv_files = []
        
        # Look for training data CSVs
        for filename in os.listdir(root_dir):
            if (filename.startswith('wrestling_training_data_') or 
                filename.startswith('active_learning_iteration_')) and filename.endswith('.csv'):
                
                filepath = os.path.join(root_dir, filename)
                file_stats = os.stat(filepath)
                
                # Try to get post count
                try:
                    df = pd.read_csv(filepath)
                    post_count = len(df)
                    labeled_count = len(df[df['manual_content_type'].notna() & (df['manual_content_type'] != '')])
                except:
                    post_count = 0
                    labeled_count = 0
                
                csv_files.append({
                    'filename': filename,
                    'size_mb': round(file_stats.st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'post_count': post_count,
                    'labeled_count': labeled_count,
                    'progress': round((labeled_count / post_count * 100) if post_count > 0 else 0, 1)
                })
        
        # Sort by modification time (newest first)
        csv_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return {
            "status": "success",
            "csv_files": csv_files
        }
        
    except Exception as e:
        logger.error(f"Error getting CSV files: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting CSV files: {str(e)}")


@router.get("/labeling/load/{filename}")
async def load_csv_for_labeling(filename: str):
    """Load CSV file data for web-based labeling."""
    try:
        csv_path = os.path.join("/Users/aahadvakani/IWCScraper", filename)
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"CSV file not found: {filename}")
        
        logger.info(f"Loading CSV for labeling: {filename}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Convert to format needed by frontend
        posts = []
        for _, row in df.iterrows():
            # Handle NaN values
            def safe_str(val):
                return str(val) if pd.notna(val) else ""
            
            def safe_float(val):
                return float(val) if pd.notna(val) else 0.0
            
            post = {
                "post_id": safe_str(row.get('post_id', '')),
                "content": safe_str(row.get('content', '')),
                "username": safe_str(row.get('username', '')),
                "subreddit": safe_str(row.get('subreddit', '')),
                "datetime": safe_str(row.get('datetime', '')),
                "upvotes": int(row.get('upvotes', 0)) if pd.notna(row.get('upvotes')) else 0,
                "comments_count": int(row.get('comments_count', 0)) if pd.notna(row.get('comments_count')) else 0,
                "url": safe_str(row.get('url', '')),
                "predicted_content_type": safe_str(row.get('predicted_content_type', '')),
                "predicted_sentiment": safe_str(row.get('predicted_sentiment', '')),
                "content_confidence": safe_float(row.get('content_confidence', 0)),
                "sentiment_confidence": safe_float(row.get('sentiment_confidence', 0)),
                "needs_review": bool(row.get('needs_review', False)),
                "manual_content_type": safe_str(row.get('manual_content_type', '')),
                "manual_sentiment": safe_str(row.get('manual_sentiment', '')),
                "notes": safe_str(row.get('notes', ''))
            }
            posts.append(post)
        
        # Calculate statistics
        total_posts = len(posts)
        labeled_posts = len([p for p in posts if p['manual_content_type']])
        needs_review_count = len([p for p in posts if p['needs_review']])
        
        # Content type distribution (predictions)
        content_type_dist = {}
        for post in posts:
            ct = post['predicted_content_type']
            content_type_dist[ct] = content_type_dist.get(ct, 0) + 1
        
        # Sort posts: needs_review first, then by confidence (lowest first)
        posts.sort(key=lambda p: (
            not p['needs_review'],  # needs_review posts first
            p['content_confidence']  # then lowest confidence first
        ))
        
        return {
            "status": "success",
            "filename": filename,
            "posts": posts,
            "statistics": {
                "total_posts": total_posts,
                "labeled_posts": labeled_posts,
                "remaining_posts": total_posts - labeled_posts,
                "needs_review_count": needs_review_count,
                "content_type_distribution": content_type_dist,
                "progress_percentage": round((labeled_posts / total_posts * 100) if total_posts > 0 else 0, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading CSV for labeling: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading CSV: {str(e)}")


@router.post("/labeling/save/{filename}")
@limiter.limit("10/minute")
async def save_labeling_progress(
    filename: str,
    request: Request,
    labels: Dict[str, Dict[str, str]]  # {post_id: {content_type: str, sentiment: str, notes: str}}
):
    """Save labeling progress back to CSV."""
    try:
        csv_path = os.path.join("/Users/aahadvakani/IWCScraper", filename)
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"CSV file not found: {filename}")
        
        logger.info(f"Saving labeling progress for {filename}: {len(labels)} labels")
        
        # Load current CSV
        df = pd.read_csv(csv_path)
        
        # Update labels
        for post_id, label_data in labels.items():
            mask = df['post_id'] == post_id
            if mask.any():
                if label_data.get('content_type'):
                    df.loc[mask, 'manual_content_type'] = label_data['content_type']
                if label_data.get('sentiment'):
                    df.loc[mask, 'manual_sentiment'] = label_data['sentiment']
                if label_data.get('notes'):
                    df.loc[mask, 'notes'] = label_data['notes']
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        
        # Calculate updated statistics
        labeled_count = len(df[df['manual_content_type'].notna() & (df['manual_content_type'] != '')])
        total_count = len(df)
        
        return {
            "status": "success",
            "message": f"Saved {len(labels)} labels",
            "filename": filename,
            "progress": {
                "labeled": labeled_count,
                "total": total_count,
                "percentage": round((labeled_count / total_count * 100) if total_count > 0 else 0, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error saving labeling progress: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving progress: {str(e)}")


@router.post("/labeling/export/{filename}")
async def export_labeled_data(filename: str, format: str = "csv"):
    """Export completed labeled data."""
    try:
        csv_path = os.path.join("/Users/aahadvakani/IWCScraper", filename)
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"CSV file not found: {filename}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Filter for labeled data only
        labeled_df = df[df['manual_content_type'].notna() & (df['manual_content_type'] != '')].copy()
        
        if len(labeled_df) == 0:
            raise HTTPException(status_code=400, detail="No labeled data found")
        
        # Create export filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = filename.replace('.csv', '')
        export_filename = f"{base_name}_labeled_{timestamp}.{format}"
        export_path = os.path.join("/Users/aahadvakani/IWCScraper", export_filename)
        
        if format == "csv":
            labeled_df.to_csv(export_path, index=False)
        elif format == "json":
            labeled_df.to_json(export_path, orient='records', indent=2)
        else:
            raise HTTPException(status_code=400, detail="Format must be 'csv' or 'json'")
        
        # Calculate final statistics
        content_type_counts = labeled_df['manual_content_type'].value_counts().to_dict()
        sentiment_counts = labeled_df[labeled_df['manual_sentiment'].notna()]['manual_sentiment'].value_counts().to_dict()
        
        logger.info(f"Exported {len(labeled_df)} labeled posts to {export_filename}")
        
        return {
            "status": "success",
            "message": f"Exported {len(labeled_df)} labeled posts",
            "export_filename": export_filename,
            "export_path": export_path,
            "statistics": {
                "total_labeled": len(labeled_df),
                "content_type_distribution": content_type_counts,
                "sentiment_distribution": sentiment_counts
            }
        }
        
    except Exception as e:
        logger.error(f"Error exporting labeled data: {e}")
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@router.get("/labeling/stats/{filename}")
async def get_labeling_stats(filename: str):
    """Get current labeling statistics for a CSV file."""
    try:
        csv_path = os.path.join("/Users/aahadvakani/IWCScraper", filename)
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"CSV file not found: {filename}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Calculate statistics
        total_posts = len(df)
        labeled_posts = len(df[df['manual_content_type'].notna() & (df['manual_content_type'] != '')])
        needs_review = len(df[df['needs_review'] == True])
        
        # Label distributions
        content_type_counts = df[df['manual_content_type'].notna()]['manual_content_type'].value_counts().to_dict()
        sentiment_counts = df[df['manual_sentiment'].notna()]['manual_sentiment'].value_counts().to_dict()
        
        # Confidence analysis
        avg_content_confidence = df['content_confidence'].mean() if 'content_confidence' in df.columns else 0
        low_confidence_count = len(df[df['content_confidence'] < 0.5]) if 'content_confidence' in df.columns else 0
        
        return {
            "status": "success",
            "filename": filename,
            "statistics": {
                "total_posts": total_posts,
                "labeled_posts": labeled_posts,
                "remaining_posts": total_posts - labeled_posts,
                "progress_percentage": round((labeled_posts / total_posts * 100) if total_posts > 0 else 0, 1),
                "needs_review_count": needs_review,
                "low_confidence_count": low_confidence_count,
                "average_confidence": round(avg_content_confidence, 3),
                "content_type_distribution": content_type_counts,
                "sentiment_distribution": sentiment_counts
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting labeling stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")