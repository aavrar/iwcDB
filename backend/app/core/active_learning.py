"""
Active learning system for iterative improvement of wrestling sentiment classification.
Implements uncertainty sampling across both content type and sentiment dimensions.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import torch
from sklearn.metrics import accuracy_score
import os

from app.core.enhanced_scraper import collect_diverse_training_data, ScrapedPost
from app.core.weak_labeling import generate_weak_labels_for_posts, WeakLabel
from app.core.fine_tuning import wrestling_fine_tuner
from app.core.logging import logger


@dataclass
class ActiveLearningCandidate:
    """Candidate post for active learning labeling."""
    post: ScrapedPost
    weak_label: WeakLabel
    content_uncertainty: float
    sentiment_uncertainty: float
    combined_uncertainty: float
    priority_score: float


class ActiveLearningSystem:
    """Active learning system for wrestling sentiment classification."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.iteration_count = 0
        self.performance_history = []
        
        # Active learning parameters
        self.uncertainty_threshold = 0.6
        self.batch_size = 100  # Number of samples to label per iteration
        self.max_iterations = 5
        self.min_improvement = 0.01  # Minimum accuracy improvement to continue
        
    def calculate_prediction_uncertainty(self, logits: np.ndarray) -> float:
        """Calculate uncertainty from model logits using entropy."""
        # Convert logits to probabilities
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        # Calculate entropy (higher entropy = more uncertainty)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        # Normalize to 0-1 range (log(n_classes) is max entropy)
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def predict_with_uncertainty(self, texts: List[str]) -> List[Tuple[int, int, float, float]]:
        """
        Predict content type and sentiment with uncertainty scores.
        Returns list of (content_pred, sentiment_pred, content_uncertainty, sentiment_uncertainty)
        """
        if not self.model or not self.tokenizer:
            logger.warning("Model not loaded, using dummy predictions")
            return [(1, 1, 0.5, 0.5) for _ in texts]  # Dummy predictions
        
        predictions = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                content_logits = outputs['content_logits'].cpu().numpy()
                sentiment_logits = outputs['sentiment_logits'].cpu().numpy()
            
            # Calculate predictions and uncertainties
            for j in range(len(batch_texts)):
                content_pred = np.argmax(content_logits[j])
                sentiment_pred = np.argmax(sentiment_logits[j])
                
                content_uncertainty = self.calculate_prediction_uncertainty(content_logits[j])
                sentiment_uncertainty = self.calculate_prediction_uncertainty(sentiment_logits[j])
                
                predictions.append((content_pred, sentiment_pred, content_uncertainty, sentiment_uncertainty))
        
        return predictions
    
    def identify_uncertain_samples(self, posts: List[ScrapedPost], 
                                 weak_labels: List[WeakLabel]) -> List[ActiveLearningCandidate]:
        """Identify samples with high uncertainty for manual labeling."""
        logger.info(f"Identifying uncertain samples from {len(posts)} posts")
        
        # Get model predictions with uncertainty
        texts = [post.content for post in posts]
        predictions = self.predict_with_uncertainty(texts)
        
        candidates = []
        
        for i, (post, weak_label) in enumerate(zip(posts, weak_labels)):
            content_pred, sentiment_pred, content_uncertainty, sentiment_uncertainty = predictions[i]
            
            # Calculate combined uncertainty
            if weak_label.content_type == 'opinion':
                # For opinions, consider both content and sentiment uncertainty
                combined_uncertainty = 0.6 * content_uncertainty + 0.4 * sentiment_uncertainty
            else:
                # For news/rumors, only content uncertainty matters
                combined_uncertainty = content_uncertainty
            
            # Calculate priority score (higher = more important to label)
            priority_score = combined_uncertainty
            
            # Boost priority for disagreement between weak labels and model predictions
            if weak_label.content_type != ['news', 'rumor', 'opinion'][content_pred]:
                priority_score += 0.2
            
            # Boost priority for low confidence weak labels
            if weak_label.content_confidence < 0.5:
                priority_score += 0.1
            
            # Boost priority for high engagement posts
            engagement_score = (post.upvotes + post.comments_count) / 100
            priority_score += min(engagement_score, 0.1)
            
            candidate = ActiveLearningCandidate(
                post=post,
                weak_label=weak_label,
                content_uncertainty=content_uncertainty,
                sentiment_uncertainty=sentiment_uncertainty,
                combined_uncertainty=combined_uncertainty,
                priority_score=priority_score
            )
            
            candidates.append(candidate)
        
        # Sort by priority score (highest first)
        candidates.sort(key=lambda x: x.priority_score, reverse=True)
        
        logger.info(f"Identified {len(candidates)} candidates for active learning")
        return candidates
    
    def select_diverse_batch(self, candidates: List[ActiveLearningCandidate], 
                           batch_size: int) -> List[ActiveLearningCandidate]:
        """Select a diverse batch of candidates for labeling."""
        if len(candidates) <= batch_size:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # Ensure representation across content types and uncertainty levels
        content_type_targets = {
            'news': batch_size // 3,
            'rumor': batch_size // 3,
            'opinion': batch_size // 3
        }
        
        content_type_counts = {'news': 0, 'rumor': 0, 'opinion': 0}
        
        # First pass: fill content type targets
        for candidate in remaining[:]:
            if len(selected) >= batch_size:
                break
            
            content_type = candidate.weak_label.content_type
            if content_type_counts[content_type] < content_type_targets[content_type]:
                selected.append(candidate)
                content_type_counts[content_type] += 1
                remaining.remove(candidate)
        
        # Second pass: fill remaining slots with highest priority
        remaining.sort(key=lambda x: x.priority_score, reverse=True)
        while len(selected) < batch_size and remaining:
            selected.append(remaining.pop(0))
        
        logger.info(f"Selected {len(selected)} diverse samples for labeling")
        logger.info(f"Content type distribution: {content_type_counts}")
        
        return selected
    
    def export_active_learning_batch(self, candidates: List[ActiveLearningCandidate], 
                                   iteration: int) -> str:
        """Export active learning batch to CSV for manual labeling."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"active_learning_iteration_{iteration}_{timestamp}.csv"
        csv_filepath = os.path.join("/Users/aahadvakani/IWCScraper", csv_filename)
        
        # Prepare data for CSV
        data = []
        for candidate in candidates:
            data.append({
                'post_id': candidate.post.id,
                'content': candidate.post.content.replace('\n', ' ').replace('\r', ' '),
                'username': candidate.post.username,
                'subreddit': candidate.post.subreddit,
                'datetime': candidate.post.datetime.isoformat(),
                'upvotes': candidate.post.upvotes,
                'comments_count': candidate.post.comments_count,
                'url': candidate.post.url,
                'weak_content_type': candidate.weak_label.content_type,
                'weak_sentiment': candidate.weak_label.sentiment or '',
                'content_uncertainty': round(candidate.content_uncertainty, 3),
                'sentiment_uncertainty': round(candidate.sentiment_uncertainty, 3),
                'combined_uncertainty': round(candidate.combined_uncertainty, 3),
                'priority_score': round(candidate.priority_score, 3),
                'manual_content_type': '',  # For manual labeling
                'manual_sentiment': '',     # For manual labeling
                'notes': '',                # For manual notes
                'iteration': iteration
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_filepath, index=False)
        
        logger.info(f"Exported {len(candidates)} candidates to {csv_filename}")
        return csv_filepath
    
    def run_active_learning_iteration(self, iteration: int) -> Dict[str, any]:
        """Run a single active learning iteration."""
        logger.info(f"Starting active learning iteration {iteration}")
        
        # Collect new diverse posts
        new_posts_by_type = await collect_diverse_training_data(target_count=2000)
        all_new_posts = []
        for posts in new_posts_by_type.values():
            all_new_posts.extend(posts)
        
        if not all_new_posts:
            logger.warning("No new posts collected for active learning")
            return {"status": "error", "message": "No new posts collected"}
        
        # Generate weak labels
        weak_labels = await generate_weak_labels_for_posts(all_new_posts)
        
        # Identify uncertain samples
        candidates = self.identify_uncertain_samples(all_new_posts, weak_labels)
        
        # Select diverse batch
        selected_batch = self.select_diverse_batch(candidates, self.batch_size)
        
        # Export for manual labeling
        csv_path = self.export_active_learning_batch(selected_batch, iteration)
        
        # Calculate statistics
        uncertainty_stats = {
            'avg_content_uncertainty': np.mean([c.content_uncertainty for c in selected_batch]),
            'avg_sentiment_uncertainty': np.mean([c.sentiment_uncertainty for c in selected_batch]),
            'avg_combined_uncertainty': np.mean([c.combined_uncertainty for c in selected_batch]),
            'avg_priority_score': np.mean([c.priority_score for c in selected_batch])
        }
        
        content_type_dist = {}
        for candidate in selected_batch:
            content_type = candidate.weak_label.content_type
            content_type_dist[content_type] = content_type_dist.get(content_type, 0) + 1
        
        result = {
            "status": "success",
            "iteration": iteration,
            "csv_file": os.path.basename(csv_path),
            "csv_path": csv_path,
            "batch_size": len(selected_batch),
            "uncertainty_stats": uncertainty_stats,
            "content_type_distribution": content_type_dist,
            "instructions": [
                f"1. Open {os.path.basename(csv_path)} and label the posts",
                "2. Fill 'manual_content_type' and 'manual_sentiment' columns",
                "3. Focus on high uncertainty/priority posts first",
                "4. When complete, use the CSV for retraining"
            ]
        }
        
        logger.info(f"Active learning iteration {iteration} complete")
        return result
    
    def evaluate_improvement(self, old_model_path: str, new_model_path: str, 
                           test_csv_path: str) -> Dict[str, float]:
        """Evaluate improvement between model versions."""
        logger.info("Evaluating model improvement")
        
        # Load test data
        test_df = pd.read_csv(test_csv_path)
        labeled_test = test_df[test_df['manual_content_type'].notna()].copy()
        
        if len(labeled_test) == 0:
            logger.warning("No labeled test data available")
            return {"improvement": 0.0}
        
        # Evaluate old model
        old_results = wrestling_fine_tuner.evaluate_on_test_set(test_csv_path, old_model_path)
        
        # Evaluate new model
        new_results = wrestling_fine_tuner.evaluate_on_test_set(test_csv_path, new_model_path)
        
        # Calculate improvement
        old_content_acc = old_results['overall_accuracy']['content']
        new_content_acc = new_results['overall_accuracy']['content']
        
        old_sentiment_acc = old_results['overall_accuracy']['sentiment']
        new_sentiment_acc = new_results['overall_accuracy']['sentiment']
        
        content_improvement = new_content_acc - old_content_acc
        sentiment_improvement = new_sentiment_acc - old_sentiment_acc
        
        overall_improvement = (content_improvement + sentiment_improvement) / 2
        
        improvement_results = {
            "content_improvement": content_improvement,
            "sentiment_improvement": sentiment_improvement,
            "overall_improvement": overall_improvement,
            "old_content_accuracy": old_content_acc,
            "new_content_accuracy": new_content_acc,
            "old_sentiment_accuracy": old_sentiment_acc,
            "new_sentiment_accuracy": new_sentiment_acc
        }
        
        logger.info(f"Model improvement: {improvement_results}")
        return improvement_results


# Singleton instance
active_learning_system = ActiveLearningSystem()


# Convenience functions
async def run_active_learning_iteration(iteration: int) -> Dict[str, any]:
    """Run an active learning iteration."""
    return await active_learning_system.run_active_learning_iteration(iteration)


def evaluate_model_improvement(old_model_path: str, new_model_path: str, test_csv_path: str) -> Dict[str, float]:
    """Evaluate improvement between model versions."""
    return active_learning_system.evaluate_improvement(old_model_path, new_model_path, test_csv_path)