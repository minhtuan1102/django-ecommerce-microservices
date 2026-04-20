"""
Model Configuration cho AI Chatbot
"""
from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class ModelConfig:
    """Configuration cho tất cả models"""
    
    # Tokenizer
    vocab_size: int = 10000
    max_seq_length: int = 128
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    sos_token: str = "<SOS>"
    eos_token: str = "<EOS>"
    
    # Embedding
    embedding_dim: int = 256
    
    # Intent Classifier
    intent_hidden_size: int = 128
    intent_num_layers: int = 2
    intents: List[str] = field(default_factory=lambda: [
        "product_query",      # Hỏi về sản phẩm
        "policy_query",       # Hỏi về chính sách
        "order_support",      # Hỗ trợ đơn hàng
        "greeting",           # Chào hỏi
        "recommendation",     # Yêu cầu gợi ý
        "general_chat"        # Chat chung
    ])
    
    # Encoder
    encoder_hidden_size: int = 256
    encoder_num_layers: int = 2
    encoder_bidirectional: bool = True
    
    # Context Fusion
    rag_context_dim: int = 384  # sentence-transformers output
    behavior_context_dim: int = 128  # từ behavior model
    fusion_hidden_size: int = 512
    
    # Decoder
    decoder_hidden_size: int = 512
    decoder_num_layers: int = 2
    attention_dim: int = 256
    
    # Training
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    teacher_forcing_ratio: float = 0.5
    clip_grad: float = 1.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def num_intents(self) -> int:
        return len(self.intents)
    
    @property
    def encoder_output_dim(self) -> int:
        return self.encoder_hidden_size * (2 if self.encoder_bidirectional else 1)
    
    def to_dict(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'embedding_dim': self.embedding_dim,
            'encoder_hidden_size': self.encoder_hidden_size,
            'encoder_num_layers': self.encoder_num_layers,
            'decoder_hidden_size': self.decoder_hidden_size,
            'decoder_num_layers': self.decoder_num_layers,
            'intent_hidden_size': self.intent_hidden_size,
            'intent_num_layers': self.intent_num_layers,
            'dropout': self.dropout,
            'num_intents': self.num_intents,
            'intents': self.intents
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        # Filter out computed properties
        exclude_keys = {'num_intents', 'encoder_output_dim'}
        
        # Defaults matching the trained model architecture
        defaults = {
            'encoder_num_layers': 1,
            'decoder_num_layers': 1,
            'intent_hidden_size': 128,  # Match trained model
            'intent_num_layers': 2,     # Match trained model (has l0 and l1)
            'dropout': 0.2,
            'encoder_bidirectional': True,
        }
        
        # Start with defaults, update with saved config
        merged = defaults.copy()
        merged.update({k: v for k, v in config_dict.items() if k not in exclude_keys})
        
        # Only keep keys that are valid init params
        import inspect
        valid_params = set(inspect.signature(cls).parameters.keys())
        filtered = {k: v for k, v in merged.items() if k in valid_params}
        
        return cls(**filtered)


# Predefined configurations
SMALL_CONFIG = ModelConfig(
    vocab_size=5000,
    embedding_dim=128,
    encoder_hidden_size=128,
    decoder_hidden_size=256,
    encoder_num_layers=1,
    decoder_num_layers=1
)

DEFAULT_CONFIG = ModelConfig()

LARGE_CONFIG = ModelConfig(
    vocab_size=20000,
    embedding_dim=512,
    encoder_hidden_size=512,
    decoder_hidden_size=1024,
    encoder_num_layers=3,
    decoder_num_layers=3
)
