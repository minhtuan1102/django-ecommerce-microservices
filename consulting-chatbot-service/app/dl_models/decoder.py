"""
Decoder với Bahdanau Attention cho Seq2Seq Response Generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import random


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention mechanism
    
    score(h_t, h_s) = v^T * tanh(W_1 * h_t + W_2 * h_s)
    """
    
    def __init__(self, decoder_hidden: int, encoder_hidden: int, attention_dim: int):
        super().__init__()
        
        self.W_decoder = nn.Linear(decoder_hidden, attention_dim, bias=False)
        self.W_encoder = nn.Linear(encoder_hidden, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: (batch, decoder_hidden) - current decoder state
            encoder_outputs: (batch, src_len, encoder_hidden) - all encoder outputs
            mask: (batch, src_len) - 1 for valid, 0 for padding
            
        Returns:
            context: (batch, encoder_hidden) - attention-weighted context
            attention_weights: (batch, src_len)
        """
        src_len = encoder_outputs.size(1)
        
        # Expand decoder hidden for each source position
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Compute attention scores
        energy = self.v(torch.tanh(
            self.W_decoder(decoder_hidden) + self.W_encoder(encoder_outputs)
        )).squeeze(-1)  # (batch, src_len)
        
        # Apply mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(energy, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, src_len)
            encoder_outputs  # (batch, src_len, encoder_hidden)
        ).squeeze(1)  # (batch, encoder_hidden)
        
        return context, attention_weights


class AttentionDecoder(nn.Module):
    """
    LSTM Decoder với Bahdanau Attention
    
    At each step:
    1. Compute attention over encoder outputs
    2. Concatenate attention context with previous embedding
    3. LSTM step
    4. Predict next token
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        encoder_hidden: int = 512,
        attention_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Attention
        self.attention = BahdanauAttention(
            decoder_hidden=hidden_size,
            encoder_hidden=encoder_hidden,
            attention_dim=attention_dim
        )
        
        # LSTM input: embedding + attention context + context features
        lstm_input_size = embedding_dim + encoder_hidden
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size + encoder_hidden, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Single decoder step
        
        Args:
            input_token: (batch,) - previous token
            hidden: (h, c) tuple, each (num_layers, batch, hidden)
            encoder_outputs: (batch, src_len, encoder_hidden)
            encoder_mask: (batch, src_len)
            
        Returns:
            output: (batch, vocab_size) - token logits
            hidden: updated hidden state
            attention_weights: (batch, src_len)
        """
        batch_size = input_token.size(0)
        
        # Embed input token
        embedded = self.dropout(self.embedding(input_token))  # (batch, emb_dim)
        
        # Compute attention using top-layer hidden state
        h_top = hidden[0][-1]  # (batch, hidden)
        context, attention_weights = self.attention(h_top, encoder_outputs, encoder_mask)
        
        # LSTM input: [embedding; context]
        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)  # (batch, 1, input_size)
        
        # LSTM step
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        lstm_output = lstm_output.squeeze(1)  # (batch, hidden)
        
        # Output: [lstm_output; context]
        output = self.output_projection(torch.cat([lstm_output, context], dim=-1))
        
        return output, hidden, attention_weights
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
        initial_state: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        max_length: int = 50,
        teacher_forcing_ratio: float = 0.5,
        sos_token_id: int = 2,
        eos_token_id: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Full decoding pass
        
        Args:
            encoder_outputs: (batch, src_len, encoder_hidden)
            encoder_mask: (batch, src_len)
            initial_state: (batch, hidden) - from encoder
            target_ids: (batch, tgt_len) - target sequence for teacher forcing
            max_length: Maximum decoding length
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Dict with outputs, attention_weights
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize hidden state
        # Expand initial_state to match LSTM layers
        h_0 = initial_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)
        
        # Determine max decoding length
        if target_ids is not None:
            max_len = target_ids.size(1)
        else:
            max_len = max_length
        
        # Start with SOS token
        input_token = torch.full((batch_size,), sos_token_id, dtype=torch.long, device=device)
        
        # Collect outputs
        all_outputs = []
        all_attention = []
        
        for t in range(max_len):
            output, hidden, attention = self.forward_step(
                input_token, hidden, encoder_outputs, encoder_mask
            )
            
            all_outputs.append(output)
            all_attention.append(attention)
            
            # Next input: teacher forcing or predicted
            if target_ids is not None and random.random() < teacher_forcing_ratio:
                input_token = target_ids[:, t]
            else:
                input_token = output.argmax(dim=-1)
        
        # Stack outputs
        outputs = torch.stack(all_outputs, dim=1)  # (batch, tgt_len, vocab)
        attention_weights = torch.stack(all_attention, dim=1)  # (batch, tgt_len, src_len)
        
        return {
            'outputs': outputs,
            'attention_weights': attention_weights,
            'predictions': outputs.argmax(dim=-1)
        }
    
    def generate(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
        initial_state: torch.Tensor,
        max_length: int = 100,
        sos_token_id: int = 2,
        eos_token_id: int = 3,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9
    ) -> Dict[str, torch.Tensor]:
        """
        Inference generation with sampling
        
        Args:
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Keep only top_k tokens (0 = disabled)
            top_p: Nucleus sampling threshold
            
        Returns:
            Dict with generated sequences and attention
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize
        h_0 = initial_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)
        
        input_token = torch.full((batch_size,), sos_token_id, dtype=torch.long, device=device)
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        generated = []
        all_attention = []
        
        for _ in range(max_length):
            output, hidden, attention = self.forward_step(
                input_token, hidden, encoder_outputs, encoder_mask
            )
            
            all_attention.append(attention)
            
            # Apply temperature
            logits = output / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Keep EOS for finished sequences
            next_token = torch.where(finished, torch.tensor(eos_token_id, device=device), next_token)
            
            generated.append(next_token)
            
            # Update finished status
            finished = finished | (next_token == eos_token_id)
            
            # Stop if all finished
            if finished.all():
                break
            
            input_token = next_token
        
        return {
            'sequences': torch.stack(generated, dim=1),
            'attention_weights': torch.stack(all_attention, dim=1)
        }


class CopyMechanismDecoder(AttentionDecoder):
    """
    Decoder với Copy Mechanism (Pointer Network)
    Có thể copy tokens từ input (hữu ích cho tên sản phẩm, số đơn hàng, etc.)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Copy gate: decides whether to copy or generate
        self.copy_gate = nn.Sequential(
            nn.Linear(self.hidden_size + self.embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        source_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Decoder step with copy mechanism
        
        Args:
            source_ids: (batch, src_len) - source token IDs for copying
        """
        batch_size = input_token.size(0)
        
        # Regular decoding step
        embedded = self.dropout(self.embedding(input_token))
        h_top = hidden[0][-1]
        context, attention_weights = self.attention(h_top, encoder_outputs, encoder_mask)
        
        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        lstm_output = lstm_output.squeeze(1)
        
        # Generate distribution
        gen_output = self.output_projection(torch.cat([lstm_output, context], dim=-1))
        gen_probs = F.softmax(gen_output, dim=-1)
        
        if source_ids is not None:
            # Compute copy probability
            copy_gate = self.copy_gate(torch.cat([lstm_output, embedded], dim=-1))
            
            # Copy distribution based on attention
            copy_probs = torch.zeros_like(gen_probs)
            copy_probs.scatter_add_(1, source_ids, attention_weights)
            
            # Mix generate and copy
            output_probs = copy_gate * copy_probs + (1 - copy_gate) * gen_probs
            output = torch.log(output_probs + 1e-12)
        else:
            output = gen_output
        
        return output, hidden, attention_weights
