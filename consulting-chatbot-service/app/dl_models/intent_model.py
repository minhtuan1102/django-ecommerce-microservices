"""
Intent Classification Model
LSTM-based classifier để phân loại ý định của người dùng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class IntentClassifier(nn.Module):
    """
    LSTM-based Intent Classifier
    
    Architecture:
    - Embedding Layer
    - Bidirectional LSTM
    - Attention pooling
    - Classification head
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_intents: int = 6,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_intents = num_intents
        
        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_intents)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def attention_pooling(
        self,
        lstm_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention-weighted pooling
        
        Args:
            lstm_output: (batch, seq_len, hidden*2)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            pooled: (batch, hidden*2)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Weighted sum
        pooled = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden*2)
        ).squeeze(1)  # (batch, hidden*2)
        
        return pooled, attention_weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) token IDs
            lengths: (batch,) actual lengths for packing
            return_attention: Return attention weights
            
        Returns:
            Dict with logits, probs, and optionally attention_weights
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        
        # LSTM
        if lengths is not None:
            # Pack padded sequences
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            lstm_output, _ = self.lstm(packed)
            lstm_output, output_lengths = nn.utils.rnn.pad_packed_sequence(
                lstm_output, batch_first=True, total_length=seq_len
            )
        else:
            lstm_output, _ = self.lstm(embedded)
        
        # Get actual output seq_len
        output_seq_len = lstm_output.size(1)
        
        # Create mask for attention - match lstm_output size
        if lengths is not None:
            mask = torch.arange(output_seq_len, device=input_ids.device).unsqueeze(0) < lengths.unsqueeze(1).to(input_ids.device)
        else:
            # Use padding from input_ids, but match output length
            mask = (input_ids[:, :output_seq_len] != 0) if output_seq_len <= seq_len else (input_ids != 0)
        
        # Attention pooling
        pooled, attention_weights = self.attention_pooling(lstm_output, mask)
        
        # Classification
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        
        outputs = {
            'logits': logits,
            'probs': probs,
            'predicted': torch.argmax(logits, dim=-1)
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def predict(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Inference mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids, return_attention=True, **kwargs)


class IntentWithContextClassifier(nn.Module):
    """
    Intent classifier that also considers conversation context
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_intents: int = 6,
        context_dim: int = 128,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()
        
        # Base intent classifier
        self.base_classifier = IntentClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_intents=num_intents,  # Temporary, we'll replace classifier
            dropout=dropout,
            padding_idx=padding_idx
        )
        
        # Replace classifier to include context
        self.base_classifier.classifier = nn.Identity()
        
        # Context-aware classifier
        self.context_projection = nn.Linear(context_dim, hidden_size)
        
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_intents)
        )
        
        self.num_intents = num_intents
    
    def forward(
        self,
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with optional context
        
        Args:
            input_ids: (batch, seq_len)
            context: (batch, context_dim) - conversation/behavior context
            lengths: (batch,)
        """
        # Get base representations
        base_out = self.base_classifier(input_ids, lengths, return_attention=True)
        
        # base_out['logits'] is now actually the pooled representation
        # because we replaced classifier with Identity
        pooled = base_out['logits']  # This is the pooled output now
        
        # Actually we need to access the attention pooling directly
        # Let me fix this...
        batch_size, seq_len = input_ids.shape
        
        embedded = self.base_classifier.embedding(input_ids)
        
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            lstm_output, _ = self.base_classifier.lstm(packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_output, batch_first=True
            )
        else:
            lstm_output, _ = self.base_classifier.lstm(embedded)
        
        if lengths is not None:
            mask = torch.arange(seq_len, device=input_ids.device).unsqueeze(0) < lengths.unsqueeze(1).to(input_ids.device)
        else:
            mask = (input_ids != 0)
        
        pooled, attention_weights = self.base_classifier.attention_pooling(lstm_output, mask)
        
        # Add context if provided
        if context is not None:
            context_proj = self.context_projection(context)
            combined = torch.cat([pooled, context_proj], dim=-1)
        else:
            # Zero context
            zero_context = torch.zeros(pooled.size(0), pooled.size(1) // 2, device=pooled.device)
            combined = torch.cat([pooled, zero_context], dim=-1)
        
        # Final classification
        logits = self.final_classifier(combined)
        probs = F.softmax(logits, dim=-1)
        
        outputs = {
            'logits': logits,
            'probs': probs,
            'predicted': torch.argmax(logits, dim=-1)
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
