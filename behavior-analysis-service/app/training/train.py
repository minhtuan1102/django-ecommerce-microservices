"""
Training Script cho Behavior Analysis Model
"""
import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup Django (if running standalone)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'behavior_service.settings')

try:
    import django
    django.setup()
except:
    pass

from app.services.data_collector import collect_or_generate_data, SyntheticDataGenerator
from app.ml_models.data_processor import DataProcessor
from app.ml_models.behavior_model import BehaviorAnalysisModel, MultiTaskLoss, create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training pipeline cho Behavior Analysis Model"""
    
    def __init__(
        self,
        model_dir: str = 'data/models',
        n_customers: int = 1000,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        device: str = None
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_customers = n_customers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.processor = None
        self.history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
    def load_data(self):
        """Load hoặc generate data"""
        logger.info(f"Loading/generating data for {self.n_customers} customers...")
        
        # Generate synthetic data
        data = collect_or_generate_data(use_synthetic=True, n_customers=self.n_customers)
        
        logger.info(f"Data loaded: {len(data['customers'])} customers, "
                   f"{len(data['orders'])} orders, {len(data['events'])} events")
        
        return data
    
    def prepare_data(self, data):
        """Process và prepare data cho training"""
        logger.info("Processing data...")
        
        self.processor = DataProcessor(sequence_length=20)
        processed = self.processor.fit_transform(data)
        
        logger.info(f"Processed {len(processed['customer_ids'])} samples")
        logger.info(f"Vocab sizes: {self.processor.vocab_sizes}")
        
        return processed
    
    def create_model(self, processed_data):
        """Create model với config từ processed data"""
        vocab_sizes = self.processor.vocab_sizes
        
        n_event_types = getattr(self.processor.sequence_builder, 'n_event_types', 10)
        n_categories = getattr(self.processor.sequence_builder, 'n_categories', 10)
        
        self.model = create_model(
            vocab_sizes=vocab_sizes,
            n_event_types=n_event_types,
            n_categories=n_categories,
            embedding_dim=32,
            lstm_hidden_size=128,
            lstm_layers=2,
            attention_size=64,
            numerical_features_dim=processed_data['numerical'].shape[1],
            n_segments=4,
            dropout=0.3
        )
        
        self.model.to(self.device)
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Save config
        self.model_config = {
            'vocab_sizes': vocab_sizes,
            'n_event_types': n_event_types,
            'n_categories': n_categories,
            'embedding_dim': 32,
            'lstm_hidden_size': 128,
            'lstm_layers': 2,
            'numerical_features_dim': processed_data['numerical'].shape[1],
            'created_at': datetime.now().isoformat()
        }
        
        return self.model
    
    def split_data(self, processed_data, val_ratio=0.2):
        """Split data into train/val"""
        n_samples = len(processed_data['customer_ids'])
        indices = np.arange(n_samples)
        
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=42,
            stratify=processed_data['labels']['segment']
        )
        
        def subset(data_dict, idx):
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    result[key] = value[idx]
                elif isinstance(value, dict):
                    result[key] = subset(value, idx)
                else:
                    result[key] = value
            return result
        
        train_data = subset(processed_data, train_idx)
        val_data = subset(processed_data, val_idx)
        
        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        return train_data, val_data
    
    def create_dataloader(self, data, shuffle=True):
        """Create PyTorch DataLoader"""
        return self.processor.create_dataloader(
            data, batch_size=self.batch_size, shuffle=shuffle
        )
    
    def train_epoch(self, model, dataloader, optimizer, loss_fn):
        """Train one epoch"""
        model.train()
        total_loss = 0
        n_batches = 0
        
        for features, labels in dataloader:
            # Move to device
            features = {k: v.to(self.device) for k, v in features.items()}
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward
            outputs = model(
                categorical=features['categorical'],
                numerical=features['numerical'],
                event_seq=features['event_seq'],
                category_seq=features['category_seq'],
                amount_seq=features['amount_seq'],
                seq_lengths=features['seq_lengths']
            )
            
            # Loss
            loss, loss_dict = loss_fn(
                segment_logits=outputs['segment_logits'],
                segment_labels=labels['segment'],
                category_logits=outputs['category_logits'],
                category_labels=labels['next_category'],
                churn_pred=outputs['churn_prob'],
                churn_labels=labels['churn']
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self, model, dataloader, loss_fn):
        """Evaluate model"""
        model.eval()
        total_loss = 0
        n_batches = 0
        
        all_segment_preds = []
        all_segment_labels = []
        all_churn_preds = []
        all_churn_labels = []
        all_category_preds = []
        all_category_labels = []
        
        for features, labels in dataloader:
            features = {k: v.to(self.device) for k, v in features.items()}
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            outputs = model(
                categorical=features['categorical'],
                numerical=features['numerical'],
                event_seq=features['event_seq'],
                category_seq=features['category_seq'],
                amount_seq=features['amount_seq'],
                seq_lengths=features['seq_lengths']
            )
            
            loss, _ = loss_fn(
                segment_logits=outputs['segment_logits'],
                segment_labels=labels['segment'],
                category_logits=outputs['category_logits'],
                category_labels=labels['next_category'],
                churn_pred=outputs['churn_prob'],
                churn_labels=labels['churn']
            )
            
            total_loss += loss.item()
            n_batches += 1
            
            # Collect predictions
            all_segment_preds.extend(outputs['segment_logits'].argmax(dim=1).cpu().numpy())
            all_segment_labels.extend(labels['segment'].cpu().numpy())
            all_churn_preds.extend(outputs['churn_prob'].cpu().numpy())
            all_churn_labels.extend(labels['churn'].cpu().numpy())
            all_category_preds.extend(outputs['category_logits'].argmax(dim=1).cpu().numpy())
            all_category_labels.extend(labels['next_category'].cpu().numpy())
        
        # Calculate metrics
        segment_acc = accuracy_score(all_segment_labels, all_segment_preds)
        category_acc = accuracy_score(all_category_labels, all_category_preds)
        
        try:
            churn_auc = roc_auc_score(all_churn_labels, all_churn_preds)
        except:
            churn_auc = 0.5
        
        metrics = {
            'segment_accuracy': segment_acc,
            'category_accuracy': category_acc,
            'churn_auc': churn_auc
        }
        
        return total_loss / n_batches, metrics
    
    def train(self):
        """Full training pipeline"""
        start_time = datetime.now()
        logger.info("=" * 50)
        logger.info("Starting training pipeline")
        logger.info("=" * 50)
        
        # 1. Load data
        data = self.load_data()
        
        # 2. Process data
        processed = self.prepare_data(data)
        
        # 3. Split data
        train_data, val_data = self.split_data(processed)
        
        # 4. Create dataloaders
        train_loader = self.create_dataloader(train_data, shuffle=True)
        val_loader = self.create_dataloader(val_data, shuffle=False)
        
        # 5. Create model
        self.create_model(processed)
        
        # 6. Setup training
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        loss_fn = MultiTaskLoss(n_tasks=3, use_uncertainty=True)
        loss_fn.to(self.device)
        
        best_val_loss = float('inf')
        best_metrics = {}
        patience = 10
        patience_counter = 0
        
        # 7. Training loop
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(self.model, train_loader, optimizer, loss_fn)
            val_loss, metrics = self.evaluate(self.model, val_loader, loss_fn)
            scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(metrics)
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Segment Acc: {metrics['segment_accuracy']:.4f}, "
                       f"Category Acc: {metrics['category_accuracy']:.4f}, "
                       f"Churn AUC: {metrics['churn_auc']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                patience_counter = 0
                self.save_model()
                logger.info(f"  -> New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 8. Final report
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 50)
        logger.info("Training completed!")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best metrics: {best_metrics}")
        logger.info(f"Model saved to: {self.model_dir}")
        logger.info("=" * 50)
        
        return best_metrics
    
    def save_model(self):
        """Save model, processor và config"""
        # Model weights
        model_path = self.model_dir / 'behavior_model.pt'
        torch.save(self.model.state_dict(), model_path)
        
        # Processor
        processor_path = self.model_dir / 'data_processor.pkl'
        with open(processor_path, 'wb') as f:
            pickle.dump(self.processor, f)
        
        # Config
        config_path = self.model_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f, indent=2)
        
        # Training history
        history_path = self.model_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Model saved to {self.model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Behavior Analysis Model')
    parser.add_argument('--customers', type=int, default=1000,
                       help='Number of customers to generate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--model-dir', type=str, default='data/models',
                       help='Directory to save model')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    trainer = Trainer(
        model_dir=args.model_dir,
        n_customers=args.customers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )
    
    metrics = trainer.train()
    print(f"\nTraining completed with metrics: {metrics}")


if __name__ == '__main__':
    main()
