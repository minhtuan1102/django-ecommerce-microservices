"""
Training Pipeline cho Chatbot
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from typing import Dict, Optional, List, Callable
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

from ..dl_models.chatbot_model import ChatbotModel, ChatbotLoss
from ..dl_models.config import ModelConfig


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    warmup_epochs: int = 3
    teacher_forcing_ratio: float = 0.5
    tf_decay: float = 0.95  # Decay teacher forcing each epoch
    clip_grad: float = 1.0
    weight_decay: float = 1e-5
    patience: int = 5
    min_delta: float = 0.001
    save_every: int = 5
    log_every: int = 10


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class ChatbotTrainer:
    """
    Trainer cho Chatbot Model
    
    Features:
    - Teacher forcing với decay
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Logging
    """
    
    def __init__(
        self,
        model: ChatbotModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        save_dir: str = "saved_models",
        device: str = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss
        self.criterion = ChatbotLoss(
            vocab_size=model.config.vocab_size,
            pad_token_id=0,
            label_smoothing=0.1,
            intent_weight=0.1
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'teacher_forcing_ratio': []
        }
        
    def train_epoch(self, teacher_forcing_ratio: float) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_response_loss = 0
        total_intent_loss = 0
        n_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            intent_labels = batch['intent_labels'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            
            # Optional context features
            rag_embeddings = batch.get('rag_embeddings')
            behavior_features = batch.get('behavior_features')
            
            if rag_embeddings is not None:
                rag_embeddings = rag_embeddings.to(self.device)
            if behavior_features is not None:
                behavior_features = behavior_features.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                target_ids=target_ids,
                lengths=input_lengths,
                rag_embeddings=rag_embeddings,
                behavior_features=behavior_features,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            # Compute loss
            loss_dict = self.criterion(
                outputs=outputs['outputs'],
                targets=target_ids,
                intent_logits=outputs['intent_probs'],
                intent_labels=intent_labels
            )
            
            loss = loss_dict['total_loss']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_grad
            )
            
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_response_loss += loss_dict['response_loss'].item()
            if 'intent_loss' in loss_dict:
                total_intent_loss += loss_dict['intent_loss'].item()
            
            n_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % self.config.log_every == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return {
            'loss': total_loss / n_batches,
            'response_loss': total_response_loss / n_batches,
            'intent_loss': total_intent_loss / n_batches if total_intent_loss > 0 else 0
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        total_response_loss = 0
        total_intent_loss = 0
        n_batches = 0
        
        correct_intents = 0
        total_intents = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            intent_labels = batch['intent_labels'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            
            rag_embeddings = batch.get('rag_embeddings')
            behavior_features = batch.get('behavior_features')
            
            if rag_embeddings is not None:
                rag_embeddings = rag_embeddings.to(self.device)
            if behavior_features is not None:
                behavior_features = behavior_features.to(self.device)
            
            # Forward (no teacher forcing)
            outputs = self.model(
                input_ids=input_ids,
                target_ids=target_ids,
                lengths=input_lengths,
                rag_embeddings=rag_embeddings,
                behavior_features=behavior_features,
                teacher_forcing_ratio=0.0  # No teacher forcing during validation
            )
            
            # Loss
            loss_dict = self.criterion(
                outputs=outputs['outputs'],
                targets=target_ids,
                intent_logits=outputs['intent_probs'],
                intent_labels=intent_labels
            )
            
            total_loss += loss_dict['total_loss'].item()
            total_response_loss += loss_dict['response_loss'].item()
            if 'intent_loss' in loss_dict:
                total_intent_loss += loss_dict['intent_loss'].item()
            
            # Intent accuracy
            predicted_intents = outputs['intent_probs'].argmax(dim=-1)
            correct_intents += (predicted_intents == intent_labels).sum().item()
            total_intents += intent_labels.size(0)
            
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'response_loss': total_response_loss / n_batches,
            'intent_loss': total_intent_loss / n_batches if total_intent_loss > 0 else 0,
            'intent_accuracy': correct_intents / total_intents if total_intents > 0 else 0
        }
    
    def train(self, num_epochs: int = None) -> Dict:
        """Full training loop"""
        num_epochs = num_epochs or self.config.num_epochs
        teacher_forcing_ratio = self.config.teacher_forcing_ratio
        
        print(f"Training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Teacher forcing ratio: {teacher_forcing_ratio:.3f}")
            print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_metrics = self.train_epoch(teacher_forcing_ratio)
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Response: {train_metrics['response_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Response: {val_metrics['response_loss']:.4f}, "
                  f"Intent Acc: {val_metrics['intent_accuracy']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['teacher_forcing_ratio'].append(teacher_forcing_ratio)
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model saved!")
            
            # Periodic save
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Decay teacher forcing
            teacher_forcing_ratio *= self.config.tf_decay
            teacher_forcing_ratio = max(0.1, teacher_forcing_ratio)  # Minimum 0.1
            
            elapsed = time.time() - start_time
            print(f"  Time: {elapsed:.1f}s")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        self.save_history()
        
        print("\n" + "=" * 60)
        print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        path = self.save_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config.to_dict(),
            'training_config': asdict(self.config),
            'history': self.history
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {self.current_epoch}, Best val loss: {self.best_val_loss:.4f}")
    
    def save_history(self):
        """Save training history to JSON"""
        path = self.save_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_chatbot(
    model: ChatbotModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str = "saved_models",
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = None
) -> Dict:
    """
    Convenient function to train chatbot
    """
    config = TrainingConfig(
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    trainer = ChatbotTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=save_dir,
        device=device
    )
    
    return trainer.train()
