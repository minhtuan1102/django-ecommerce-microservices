"""
Deep Learning Model cho Behavior Analysis
Kiến trúc: Embedding + LSTM + Attention + Multi-task heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class AttentionLayer(nn.Module):
    """Self-Attention mechanism cho sequence data"""
    
    def __init__(self, hidden_size: int, attention_size: int = 64):
        super().__init__()
        self.attention_size = attention_size
        
        self.query = nn.Linear(hidden_size, attention_size)
        self.key = nn.Linear(hidden_size, attention_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.scale = math.sqrt(attention_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
            
        Returns:
            output: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.query(x)  # (batch, seq_len, attention_size)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, seq_len, seq_len)
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        
        # Handle NaN from all-padding sequences
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Weighted sum
        context = torch.bmm(attention_weights, V)  # (batch, seq_len, hidden_size)
        
        # Average pooling over sequence
        if mask is not None:
            mask_expanded = mask[:, 0, :].unsqueeze(-1)  # (batch, seq_len, 1)
            context = context * mask_expanded
            output = context.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            output = context.mean(dim=1)
        
        return output, attention_weights.mean(dim=1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    
    def __init__(self, hidden_size: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % n_heads == 0
        
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        attention = self.dropout(attention)
        
        # Output
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return self.output(context)


class EmbeddingLayer(nn.Module):
    """Embedding layer cho categorical features"""
    
    def __init__(self, vocab_sizes: Dict[str, int], embedding_dim: int = 32):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        
        for name, vocab_size in vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(
                num_embeddings=vocab_size + 1,  # +1 for padding/unknown
                embedding_dim=embedding_dim,
                padding_idx=0
            )
        
        self.output_dim = len(vocab_sizes) * embedding_dim
        
    def forward(self, categorical_features: torch.Tensor, feature_names: List[str]) -> torch.Tensor:
        """
        Args:
            categorical_features: (batch, n_categorical_features)
            feature_names: List of feature names
            
        Returns:
            Concatenated embeddings (batch, total_embedding_dim)
        """
        embeddings = []
        
        for i, name in enumerate(feature_names):
            if name in self.embeddings:
                emb = self.embeddings[name](categorical_features[:, i])
                embeddings.append(emb)
        
        return torch.cat(embeddings, dim=-1)


class SequenceEncoder(nn.Module):
    """LSTM encoder cho sequence data"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 n_layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
            lengths: (batch,) - actual sequence lengths
            
        Returns:
            outputs: (batch, seq_len, hidden_size * directions)
            final_state: (batch, hidden_size * directions)
        """
        # Pack sequences
        lengths = lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward
        outputs, (h_n, _) = self.lstm(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Concatenate final states from both directions
        if self.lstm.bidirectional:
            final_state = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final_state = h_n[-1]
        
        return outputs, final_state


class GNNEncoder(nn.Module):
    """
    Prototype GNN Encoder for Graph-based behavior analysis
    Uses Graph Convolutional Networks (GCN) to process user-product graph
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (n_nodes, input_dim)
            adj: Adjacency matrix (n_nodes, n_nodes)
        """
        # Simple GCN: Relu(AXW)
        # x = torch.matmul(adj, x)
        x = self.relu(self.conv1(torch.matmul(adj, x)))
        x = self.relu(self.conv2(torch.matmul(adj, x)))
        return x


class BehaviorAnalysisModel(nn.Module):
    """
    Main model architecture combining:
    - Embedding layers cho categorical features
    - LSTM cho sequence data
    - Attention mechanism
    - Multi-task output heads
    """
    
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        n_event_types: int = 10,
        n_categories: int = 10,
        embedding_dim: int = 32,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        attention_size: int = 64,
        numerical_features_dim: int = 6,
        n_segments: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Store config
        self.vocab_sizes = vocab_sizes
        self.n_segments = n_segments
        self.n_categories = n_categories
        
        # Embedding layers
        self.categorical_embedding = EmbeddingLayer(vocab_sizes, embedding_dim)
        self.event_embedding = nn.Embedding(n_event_types + 1, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(n_categories + 1, embedding_dim, padding_idx=0)
        
        # Sequence encoder
        sequence_input_dim = embedding_dim * 2 + 1  # event + category + amount
        self.sequence_encoder = SequenceEncoder(
            input_size=sequence_input_dim,
            hidden_size=lstm_hidden_size,
            n_layers=lstm_layers,
            dropout=dropout
        )
        
        # Attention
        self.attention = AttentionLayer(
            hidden_size=self.sequence_encoder.output_size,
            attention_size=attention_size
        )
        
        # Multi-head attention for richer representations
        self.mha = MultiHeadAttention(
            hidden_size=self.sequence_encoder.output_size,
            n_heads=4,
            dropout=dropout
        )
        
        # Feature fusion
        categorical_dim = self.categorical_embedding.output_dim
        sequence_dim = self.sequence_encoder.output_size
        
        fusion_input_dim = categorical_dim + numerical_features_dim + sequence_dim * 2
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        # 1. Segment classification (4 classes: VIP, Regular, New, Churned)
        self.segment_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_segments)
        )
        
        # 2. Next purchase category prediction
        self.category_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_categories)
        )
        
        # 3. Churn probability (binary)
        self.churn_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        categorical: torch.Tensor,
        numerical: torch.Tensor,
        event_seq: torch.Tensor,
        category_seq: torch.Tensor,
        amount_seq: torch.Tensor,
        seq_lengths: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            categorical: (batch, n_categorical) - encoded categorical features
            numerical: (batch, n_numerical) - normalized numerical features
            event_seq: (batch, seq_len) - event type sequence
            category_seq: (batch, seq_len) - category sequence
            amount_seq: (batch, seq_len) - amount sequence
            seq_lengths: (batch,) - actual sequence lengths
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions
        """
        batch_size = categorical.shape[0]
        
        # Embed categorical features
        feature_names = list(self.vocab_sizes.keys())
        categorical_emb = self.categorical_embedding(categorical, feature_names)
        
        # Embed sequences
        event_emb = self.event_embedding(event_seq)  # (batch, seq_len, emb_dim)
        category_emb = self.category_embedding(category_seq)
        amount_expanded = amount_seq.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Combine sequence features
        sequence_features = torch.cat([event_emb, category_emb, amount_expanded], dim=-1)
        
        # Encode sequence with LSTM
        lstm_output, lstm_final = self.sequence_encoder(sequence_features, seq_lengths)
        
        # Create mask for attention
        max_len = lstm_output.shape[1]
        mask = torch.arange(max_len, device=lstm_output.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        
        # Apply attention
        attention_output, attention_weights = self.attention(lstm_output, mask)
        
        # Apply multi-head attention
        mha_output = self.mha(lstm_output, mask)
        mha_pooled = mha_output.mean(dim=1)
        
        # Combine all features
        combined = torch.cat([
            categorical_emb,
            numerical,
            attention_output,
            mha_pooled
        ], dim=-1)
        
        # Feature fusion
        fused = self.feature_fusion(combined)
        
        # Output heads
        segment_logits = self.segment_head(fused)
        category_logits = self.category_head(fused)
        churn_prob = self.churn_head(fused).squeeze(-1)
        
        outputs = {
            'segment_logits': segment_logits,
            'segment_probs': F.softmax(segment_logits, dim=-1),
            'category_logits': category_logits,
            'category_probs': F.softmax(category_logits, dim=-1),
            'churn_prob': churn_prob,
            'embeddings': fused  # For downstream analysis
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def predict(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prediction mode (no gradient)"""
        self.eval()
        with torch.no_grad():
            return self.forward(
                categorical=features['categorical'],
                numerical=features['numerical'],
                event_seq=features['event_seq'],
                category_seq=features['category_seq'],
                amount_seq=features['amount_seq'],
                seq_lengths=features['seq_lengths'],
                return_attention=True
            )


class MultiTaskLoss(nn.Module):
    """Multi-task loss với learnable weights"""
    
    def __init__(self, n_tasks: int = 3, use_uncertainty: bool = True):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        
        if use_uncertainty:
            # Learnable log variance for uncertainty weighting
            self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        else:
            self.weights = nn.Parameter(torch.ones(n_tasks))
    
    def forward(
        self,
        segment_logits: torch.Tensor,
        segment_labels: torch.Tensor,
        category_logits: torch.Tensor,
        category_labels: torch.Tensor,
        churn_pred: torch.Tensor,
        churn_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        # Individual losses
        segment_loss = F.cross_entropy(segment_logits, segment_labels)
        category_loss = F.cross_entropy(category_logits, category_labels)
        churn_loss = F.binary_cross_entropy(churn_pred, churn_labels)
        
        losses = [segment_loss, category_loss, churn_loss]
        
        if self.use_uncertainty:
            # Uncertainty weighting: loss_i / (2 * exp(log_var_i)) + log_var_i / 2
            total_loss = 0
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * loss + self.log_vars[i]
        else:
            weights = F.softmax(self.weights, dim=0)
            total_loss = sum(w * l for w, l in zip(weights, losses))
        
        loss_dict = {
            'segment_loss': segment_loss,
            'category_loss': category_loss,
            'churn_loss': churn_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict


def create_model(
    vocab_sizes: Dict[str, int],
    n_event_types: int = 10,
    n_categories: int = 10,
    **kwargs
) -> BehaviorAnalysisModel:
    """Factory function để tạo model"""
    return BehaviorAnalysisModel(
        vocab_sizes=vocab_sizes,
        n_event_types=n_event_types,
        n_categories=n_categories,
        **kwargs
    )
