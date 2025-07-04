"""
Fine-tuning script for multi-task wrestling sentiment classification.
Trains MiniLM on labeled wrestling dataset with dual output heads.
"""
import os
import json
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch.nn as nn

from app.core.config import settings
from app.core.logging import logger
from app.core.wrestling_preprocessor import prepare_texts_for_training


@dataclass
class TrainingData:
    """Container for training dataset."""
    texts: List[str]
    content_labels: List[int]  # 0=news, 1=rumor, 2=opinion
    sentiment_labels: List[int]  # 0=negative, 1=neutral, 2=positive, -1=not applicable
    
    
class MultiTaskWrestlingModel(nn.Module):
    """Multi-task model for content type and sentiment classification."""
    
    def __init__(self, base_model_name: str, num_content_labels: int = 3, num_sentiment_labels: int = 3):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, 
            num_labels=num_content_labels  # Will modify this
        )
        
        # Get hidden size from base model
        hidden_size = self.base_model.config.hidden_size
        
        # Remove the original classifier
        self.base_model.classifier = nn.Identity()
        
        # Add custom classification heads
        self.content_classifier = nn.Linear(hidden_size, num_content_labels)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_labels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None, content_labels=None, sentiment_labels=None):
        # Get base model outputs (handle both BERT and RoBERTa)
        if hasattr(self.base_model, 'bert'):
            outputs = self.base_model.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
        elif hasattr(self.base_model, 'roberta'):
            outputs = self.base_model.roberta(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        else:
            # Fallback to base model forward pass
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            pooled_output = outputs.hidden_states[-1][:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Get logits for both tasks
        content_logits = self.content_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        total_loss = None
        if content_labels is not None and sentiment_labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            
            # Content classification loss
            content_loss = loss_fn(content_logits, content_labels)
            
            # Sentiment classification loss (only for opinion posts)
            opinion_mask = content_labels == 2  # opinion posts
            if opinion_mask.sum() > 0:
                sentiment_loss = loss_fn(
                    sentiment_logits[opinion_mask], 
                    sentiment_labels[opinion_mask]
                )
            else:
                sentiment_loss = torch.tensor(0.0, device=content_logits.device)
            
            # Combined loss with weighting
            total_loss = content_loss + 0.5 * sentiment_loss
        
        return {
            'loss': total_loss,
            'content_logits': content_logits,
            'sentiment_logits': sentiment_logits
        }


class WrestlingFineTuner:
    """Fine-tuning service for wrestling sentiment analysis."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.training_args = None
        
        # Label mappings
        self.content_type_to_id = {'news': 0, 'rumor': 1, 'opinion': 2}
        self.id_to_content_type = {v: k for k, v in self.content_type_to_id.items()}
        
        self.sentiment_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id_to_sentiment = {v: k for k, v in self.sentiment_to_id.items()}
        
    def load_training_data_from_csv(self, csv_path: str) -> TrainingData:
        """Load training data from labeled CSV file."""
        logger.info(f"Loading training data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Filter for manually labeled rows
        labeled_df = df[
            df['manual_content_type'].notna() & 
            (df['manual_content_type'] != '')
        ].copy()
        
        if len(labeled_df) == 0:
            raise ValueError("No manually labeled data found in CSV")
        
        logger.info(f"Found {len(labeled_df)} labeled examples")
        
        # Preprocess texts and track which ones are kept
        original_texts = labeled_df['content'].tolist()
        preprocessed_texts = prepare_texts_for_training(original_texts)
        
        # Find which texts were kept after preprocessing
        kept_indices = []
        for i, orig_text in enumerate(original_texts):
            # Simple length filter check (same as in preprocessor)
            cleaned_length = len(orig_text.strip())
            if 10 <= cleaned_length <= 2000:
                kept_indices.append(i)
        
        logger.info(f"Preprocessing kept {len(kept_indices)} out of {len(original_texts)} texts")
        
        # Filter dataframe to match preprocessed texts
        filtered_df = labeled_df.iloc[kept_indices].copy()
        texts = preprocessed_texts
        
        # Convert content type labels for filtered data
        content_labels = []
        for content_type in filtered_df['manual_content_type']:
            if content_type in self.content_type_to_id:
                content_labels.append(self.content_type_to_id[content_type])
            else:
                logger.warning(f"Unknown content type: {content_type}, defaulting to opinion")
                content_labels.append(2)  # Default to opinion
        
        # Convert sentiment labels for filtered data
        sentiment_labels = []
        for idx, row in filtered_df.iterrows():
            content_type = row['manual_content_type']
            sentiment = row['manual_sentiment']
            
            if pd.notna(sentiment) and sentiment != '':
                if sentiment in self.sentiment_to_id:
                    sentiment_labels.append(self.sentiment_to_id[sentiment])
                else:
                    logger.warning(f"Unknown sentiment: {sentiment}, defaulting to neutral")
                    sentiment_labels.append(1)  # Neutral
            else:
                sentiment_labels.append(1)  # Default to neutral (0 contribution)
        
        # Validate data
        assert len(texts) == len(content_labels) == len(sentiment_labels)
        
        # Log distribution
        content_dist = {self.id_to_content_type[i]: content_labels.count(i) for i in range(3)}
        sentiment_dist = {self.id_to_sentiment[i]: sentiment_labels.count(i) for i in range(3)}
        
        logger.info(f"Content type distribution: {content_dist}")
        logger.info(f"Sentiment distribution: {sentiment_dist}")
        
        return TrainingData(
            texts=texts,
            content_labels=content_labels,
            sentiment_labels=sentiment_labels
        )
    
    def prepare_datasets(self, training_data: TrainingData, test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """Prepare train and validation datasets."""
        logger.info("Preparing datasets for training")
        
        # Split data
        train_texts, val_texts, train_content, val_content, train_sentiment, val_sentiment = train_test_split(
            training_data.texts,
            training_data.content_labels,
            training_data.sentiment_labels,
            test_size=test_size,
            random_state=42,
            stratify=training_data.content_labels
        )
        
        # Tokenize texts
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'content_labels': train_content,
            'sentiment_labels': train_sentiment
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'content_labels': val_content,
            'sentiment_labels': val_sentiment
        })
        
        # Apply tokenization
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def setup_model_and_tokenizer(self, base_model_name: str = None):
        """Setup model and tokenizer for training."""
        if base_model_name is None:
            base_model_name = settings.MODEL_NAME
        logger.info(f"Setting up model and tokenizer: {base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create multi-task model
        self.model = MultiTaskWrestlingModel(base_model_name)
        
        logger.info("Model and tokenizer setup complete")
    
    def setup_training_args(self, output_dir: str = "./wrestling_fine_tuned_model"):
        """Setup training arguments."""
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_steps=100,
            save_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to=None,  # Disable wandb
            dataloader_drop_last=False,
            fp16=torch.cuda.is_available(),
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        # Get content and sentiment predictions
        content_preds = np.argmax(predictions[0], axis=1)
        sentiment_preds = np.argmax(predictions[1], axis=1)
        
        content_labels = labels[:, 0]
        sentiment_labels = labels[:, 1]
        
        # Content type accuracy
        content_accuracy = accuracy_score(content_labels, content_preds)
        
        # Sentiment accuracy (only for opinion posts)
        opinion_mask = content_labels == 2
        if opinion_mask.sum() > 0:
            sentiment_accuracy = accuracy_score(
                sentiment_labels[opinion_mask], 
                sentiment_preds[opinion_mask]
            )
        else:
            sentiment_accuracy = 0.0
        
        return {
            'content_accuracy': content_accuracy,
            'sentiment_accuracy': sentiment_accuracy,
            'combined_accuracy': (content_accuracy + sentiment_accuracy) / 2
        }
    
    def train(self, csv_path: str) -> Dict[str, float]:
        """Complete training pipeline."""
        logger.info("Starting wrestling sentiment fine-tuning")
        
        # Load and prepare data
        training_data = self.load_training_data_from_csv(csv_path)
        
        # Setup model
        self.setup_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(training_data)
        
        # Setup training arguments
        self.setup_training_args()
        
        # Custom data collator for multi-task learning
        def multi_task_data_collator(features):
            batch = {}
            
            # Tokenization features
            batch['input_ids'] = torch.stack([torch.tensor(f['input_ids']) for f in features])
            batch['attention_mask'] = torch.stack([torch.tensor(f['attention_mask']) for f in features])
            
            # Labels
            batch['content_labels'] = torch.tensor([f['content_labels'] for f in features])
            batch['sentiment_labels'] = torch.tensor([f['sentiment_labels'] for f in features])
            
            return batch
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=multi_task_data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        # Final evaluation
        eval_result = trainer.evaluate()
        
        logger.info("Training complete!")
        logger.info(f"Final metrics: {eval_result}")
        
        return eval_result
    
    def evaluate_on_test_set(self, csv_path: str, model_path: str) -> Dict[str, any]:
        """Evaluate trained model on test set."""
        # Load test data
        test_data = self.load_training_data_from_csv(csv_path)
        
        # Load trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MultiTaskWrestlingModel.from_pretrained(model_path)
        
        # Prepare test dataset
        test_dataset = Dataset.from_dict({
            'text': test_data.texts,
            'content_labels': test_data.content_labels,
            'sentiment_labels': test_data.sentiment_labels
        })
        
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
        
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        # Run evaluation
        trainer = Trainer(model=self.model, tokenizer=self.tokenizer)
        predictions = trainer.predict(test_dataset)
        
        # Calculate detailed metrics
        content_preds = np.argmax(predictions.predictions[0], axis=1)
        sentiment_preds = np.argmax(predictions.predictions[1], axis=1)
        
        content_labels = predictions.label_ids[:, 0]
        sentiment_labels = predictions.label_ids[:, 1]
        
        # Content classification report
        content_report = classification_report(
            content_labels, content_preds,
            target_names=['news', 'rumor', 'opinion'],
            output_dict=True
        )
        
        # Sentiment classification report (opinion posts only)
        opinion_mask = content_labels == 2
        if opinion_mask.sum() > 0:
            sentiment_report = classification_report(
                sentiment_labels[opinion_mask], 
                sentiment_preds[opinion_mask],
                target_names=['negative', 'neutral', 'positive'],
                output_dict=True
            )
        else:
            sentiment_report = {}
        
        return {
            'content_classification': content_report,
            'sentiment_classification': sentiment_report,
            'overall_accuracy': {
                'content': content_report['accuracy'],
                'sentiment': sentiment_report.get('accuracy', 0.0)
            }
        }


# Singleton instance
wrestling_fine_tuner = WrestlingFineTuner()


# Convenience functions
def train_wrestling_model(csv_path: str) -> Dict[str, float]:
    """Train wrestling sentiment model from labeled CSV."""
    return wrestling_fine_tuner.train(csv_path)


def evaluate_wrestling_model(csv_path: str, model_path: str) -> Dict[str, any]:
    """Evaluate trained wrestling model."""
    return wrestling_fine_tuner.evaluate_on_test_set(csv_path, model_path)