"""
PyTorch Dataset classes cho training
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import json
import random

from ..dl_models.tokenizer import VietnameseTokenizer
from .data_generator import ConversationSample


class ChatbotDataset(Dataset):
    """
    Dataset cho training Seq2Seq chatbot
    
    Returns:
        - input_ids: tokenized query
        - target_ids: tokenized response
        - intent_label: intent index
        - input_length: actual query length
        - target_length: actual response length
    """
    
    INTENT_MAP = {
        "greeting": 0,
        "product_query": 1,
        "policy_query": 2,
        "order_support": 3,
        "recommendation": 4,
        "general_chat": 5
    }
    
    def __init__(
        self,
        samples: List[ConversationSample],
        tokenizer: VietnameseTokenizer,
        max_input_length: int = 64,
        max_target_length: int = 128
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Tokenize query
        input_ids, input_length = self.tokenizer.encode(
            sample.query,
            max_length=self.max_input_length,
            return_length=True
        )
        
        # Tokenize response
        target_ids, target_length = self.tokenizer.encode(
            sample.response,
            max_length=self.max_target_length,
            return_length=True
        )
        
        # Intent label
        intent_label = self.INTENT_MAP.get(sample.intent, 5)  # default to general_chat
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'intent_label': torch.tensor(intent_label, dtype=torch.long),
            'input_length': torch.tensor(input_length, dtype=torch.long),
            'target_length': torch.tensor(target_length, dtype=torch.long)
        }


class IntentDataset(Dataset):
    """
    Dataset cho training Intent Classifier riêng
    """
    
    INTENT_MAP = {
        "greeting": 0,
        "product_query": 1,
        "policy_query": 2,
        "order_support": 3,
        "recommendation": 4,
        "general_chat": 5
    }
    
    def __init__(
        self,
        samples: List[ConversationSample],
        tokenizer: VietnameseTokenizer,
        max_length: int = 64
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        input_ids, length = self.tokenizer.encode(
            sample.query,
            max_length=self.max_length,
            return_length=True
        )
        
        intent_label = self.INTENT_MAP.get(sample.intent, 5)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'intent_label': torch.tensor(intent_label, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }


class ChatbotCollator:
    """
    Custom collator để handle variable length sequences
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack all tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        target_ids = torch.stack([item['target_ids'] for item in batch])
        intent_labels = torch.stack([item['intent_label'] for item in batch])
        input_lengths = torch.stack([item['input_length'] for item in batch])
        target_lengths = torch.stack([item['target_length'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'intent_labels': intent_labels,
            'input_lengths': input_lengths,
            'target_lengths': target_lengths
        }


def load_samples_from_json(path: str) -> List[ConversationSample]:
    """Load samples from JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [
        ConversationSample(
            query=d["query"],
            response=d["response"],
            intent=d["intent"],
            context=d.get("context"),
            metadata=d.get("metadata", {})
        )
        for d in data
    ]


def split_data(
    samples: List[ConversationSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List, List, List]:
    """Split data into train/val/test"""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return (
        shuffled[:train_end],
        shuffled[train_end:val_end],
        shuffled[val_end:]
    )


def create_dataloaders(
    samples: List[ConversationSample],
    tokenizer: VietnameseTokenizer,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    max_input_length: int = 64,
    max_target_length: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders
    """
    # Split data
    train_samples, val_samples, test_samples = split_data(
        samples, train_ratio, val_ratio
    )
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Create datasets
    train_dataset = ChatbotDataset(
        train_samples, tokenizer, max_input_length, max_target_length
    )
    val_dataset = ChatbotDataset(
        val_samples, tokenizer, max_input_length, max_target_length
    )
    test_dataset = ChatbotDataset(
        test_samples, tokenizer, max_input_length, max_target_length
    )
    
    # Create collator
    collator = ChatbotCollator(tokenizer.pad_token_id)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class ContextAugmentedDataset(ChatbotDataset):
    """
    Dataset với context augmentation
    Simulate RAG và behavior features cho training
    """
    
    def __init__(
        self,
        samples: List[ConversationSample],
        tokenizer: VietnameseTokenizer,
        rag_dim: int = 384,
        behavior_dim: int = 128,
        n_rag_docs: int = 3,
        **kwargs
    ):
        super().__init__(samples, tokenizer, **kwargs)
        self.rag_dim = rag_dim
        self.behavior_dim = behavior_dim
        self.n_rag_docs = n_rag_docs
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get base features
        item = super().__getitem__(idx)
        
        # Simulate RAG embeddings (in real usage, these come from retriever)
        # Use random but seeded by intent for some consistency
        intent = item['intent_label'].item()
        
        # RAG embeddings - simulated
        rag_embeddings = torch.randn(self.n_rag_docs, self.rag_dim)
        
        # Behavior features - simulated based on intent
        behavior_features = torch.randn(self.behavior_dim)
        
        item['rag_embeddings'] = rag_embeddings
        item['behavior_features'] = behavior_features
        
        return item
