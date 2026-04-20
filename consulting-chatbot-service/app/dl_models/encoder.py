"""
Encoder Models cho Seq2Seq Chatbot
- QueryEncoder: Encode user query
- ContextEncoder: Encode RAG + Behavior context
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, hidden_size: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % n_heads == 0
        
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            mask: (batch, seq_len)
            
        Returns:
            output: (batch, seq_len, hidden_size)
            attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.output(context), attention_weights


class QueryEncoder(nn.Module):
    """
    Encode user query với LSTM + Self-Attention
    
    Output: encoder_outputs (for attention) + final_state (for decoder init)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size * self.num_directions,
            n_heads=4,
            dropout=dropout
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = hidden_size * self.num_directions
        
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            lengths: (batch,)
            
        Returns:
            Dict with:
                - encoder_outputs: (batch, seq_len, hidden*2)
                - final_state: (batch, hidden*2)
                - attention_weights: (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        embedded = self.dropout(self.embedding(input_ids))
        
        # LSTM
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            lstm_output, (h_n, c_n) = self.lstm(packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_output, batch_first=True, total_length=seq_len
            )
        else:
            lstm_output, (h_n, c_n) = self.lstm(embedded)
        
        # Get output seq_len
        output_seq_len = lstm_output.size(1)
        
        # Create mask - match lstm_output size
        if lengths is not None:
            mask = torch.arange(output_seq_len, device=input_ids.device).unsqueeze(0) < lengths.unsqueeze(1).to(input_ids.device)
        else:
            mask = (input_ids[:, :output_seq_len] != 0) if output_seq_len <= seq_len else (input_ids != 0)
        
        # Self-attention with residual
        attended, attention_weights = self.self_attention(lstm_output, mask)
        encoder_outputs = self.layer_norm(lstm_output + attended)
        
        # Final state: concatenate last hidden states from both directions
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden)
            # Take last layer from both directions
            final_state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final_state = h_n[-1]
        
        return {
            'encoder_outputs': encoder_outputs,
            'final_state': final_state,
            'attention_weights': attention_weights,
            'mask': mask
        }


class ContextEncoder(nn.Module):
    """
    Encode multiple context sources:
    - RAG retrieved documents (already embedded by sentence-transformers)
    - Behavior model predictions
    - Intent classification
    
    Fuse them into a unified context representation
    """
    
    def __init__(
        self,
        rag_dim: int = 384,  # sentence-transformers output
        behavior_dim: int = 128,  # behavior model embeddings
        intent_dim: int = 6,  # number of intents (one-hot)
        hidden_size: int = 512,
        output_dim: int = 512,
        n_rag_docs: int = 3,  # number of RAG documents
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_rag_docs = n_rag_docs
        self.output_dim = output_dim
        
        # Project RAG embeddings
        self.rag_projection = nn.Sequential(
            nn.Linear(rag_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention over RAG documents
        self.rag_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False)
        )
        
        # Project behavior features
        self.behavior_projection = nn.Sequential(
            nn.Linear(behavior_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Project intent
        self.intent_projection = nn.Sequential(
            nn.Linear(intent_dim, hidden_size // 4),
            nn.ReLU()
        )
        
        # Fusion layer
        fusion_input_dim = hidden_size + hidden_size // 2 + hidden_size // 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )
        
    def forward(
        self,
        rag_embeddings: Optional[torch.Tensor] = None,
        behavior_features: Optional[torch.Tensor] = None,
        intent_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            rag_embeddings: (batch, n_docs, rag_dim)
            behavior_features: (batch, behavior_dim)
            intent_probs: (batch, num_intents)
            
        Returns:
            context: (batch, output_dim)
        """
        batch_size = None
        components = []
        
        # Process RAG embeddings
        if rag_embeddings is not None:
            batch_size = rag_embeddings.size(0)
            # Project each document
            rag_proj = self.rag_projection(rag_embeddings)  # (batch, n_docs, hidden)
            
            # Attention over documents
            scores = self.rag_attention(rag_proj).squeeze(-1)  # (batch, n_docs)
            weights = F.softmax(scores, dim=-1)
            
            # Weighted sum
            rag_context = torch.bmm(weights.unsqueeze(1), rag_proj).squeeze(1)  # (batch, hidden)
            components.append(rag_context)
        
        # Process behavior features
        if behavior_features is not None:
            if batch_size is None:
                batch_size = behavior_features.size(0)
            behavior_proj = self.behavior_projection(behavior_features)
            components.append(behavior_proj)
        
        # Process intent
        if intent_probs is not None:
            if batch_size is None:
                batch_size = intent_probs.size(0)
            intent_proj = self.intent_projection(intent_probs)
            components.append(intent_proj)
        
        # Handle missing components with zeros
        device = next(self.parameters()).device
        if batch_size is None:
            raise ValueError("At least one context source must be provided")
        
        if rag_embeddings is None:
            hidden_size = self.fusion[0].in_features
            rag_size = hidden_size - hidden_size // 2 - hidden_size // 4 + hidden_size // 4
            # This is messy, let's just compute properly
            rag_size = self.rag_projection[0].out_features
            components.insert(0, torch.zeros(batch_size, rag_size, device=device))
        
        if behavior_features is None:
            beh_size = self.behavior_projection[0].out_features
            components.insert(1 if len(components) > 0 else 0, torch.zeros(batch_size, beh_size, device=device))
        
        if intent_probs is None:
            int_size = self.intent_projection[0].out_features
            components.append(torch.zeros(batch_size, int_size, device=device))
        
        # Concatenate and fuse
        combined = torch.cat(components, dim=-1)
        context = self.fusion(combined)
        
        return context


class EncoderWithContext(nn.Module):
    """
    Combined encoder that processes query and context together
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        encoder_hidden: int = 256,
        context_hidden: int = 512,
        num_layers: int = 2,
        rag_dim: int = 384,
        behavior_dim: int = 128,
        num_intents: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Query encoder
        self.query_encoder = QueryEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=encoder_hidden,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Context encoder
        self.context_encoder = ContextEncoder(
            rag_dim=rag_dim,
            behavior_dim=behavior_dim,
            intent_dim=num_intents,
            hidden_size=context_hidden,
            output_dim=context_hidden,
            dropout=dropout
        )
        
        # Fusion: combine query encoding with context
        query_dim = encoder_hidden * 2  # bidirectional
        self.fusion = nn.Sequential(
            nn.Linear(query_dim + context_hidden, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = context_hidden
        
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        rag_embeddings: Optional[torch.Tensor] = None,
        behavior_features: Optional[torch.Tensor] = None,
        intent_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode query with context
        
        Returns:
            Dict with encoder_outputs, final_state, context
        """
        # Encode query
        query_out = self.query_encoder(input_ids, lengths)
        
        # Encode context
        context = self.context_encoder(
            rag_embeddings=rag_embeddings,
            behavior_features=behavior_features,
            intent_probs=intent_probs
        )
        
        # Fuse query final state with context
        combined = torch.cat([query_out['final_state'], context], dim=-1)
        fused_state = self.fusion(combined)
        
        return {
            'encoder_outputs': query_out['encoder_outputs'],
            'encoder_mask': query_out['mask'],
            'final_state': fused_state,
            'context': context,
            'attention_weights': query_out['attention_weights']
        }
