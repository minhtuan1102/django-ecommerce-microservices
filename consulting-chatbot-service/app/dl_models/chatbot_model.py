"""
Full Chatbot Model - Tích hợp Intent + Encoder + Decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .config import ModelConfig
from .intent_model import IntentClassifier
from .encoder import QueryEncoder, ContextEncoder, EncoderWithContext
from .decoder import AttentionDecoder


class ChatbotModel(nn.Module):
    """
    End-to-end Chatbot Model
    
    Architecture:
    1. Intent Classification (optional, can use external)
    2. Query Encoding (LSTM + Attention)
    3. Context Integration (RAG + Behavior + Intent)
    4. Response Decoding (LSTM + Bahdanau Attention)
    
    Flow:
    query -> Intent -> [RAG retrieval] -> Encoder -> Decoder -> response
                              |
                     Behavior Model features
    """
    
    def __init__(
        self,
        config: ModelConfig,
        include_intent_classifier: bool = True
    ):
        super().__init__()
        
        self.config = config
        self.include_intent_classifier = include_intent_classifier
        
        # Shared embedding (optional - can use separate)
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=0
        )
        
        # Intent Classifier
        if include_intent_classifier:
            self.intent_classifier = IntentClassifier(
                vocab_size=config.vocab_size,
                embedding_dim=config.embedding_dim,
                hidden_size=config.intent_hidden_size,
                num_layers=config.intent_num_layers,
                num_intents=config.num_intents,
                dropout=config.dropout
            )
            # Share embedding
            self.intent_classifier.embedding = self.embedding
        
        # Query Encoder
        self.query_encoder = QueryEncoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.encoder_hidden_size,
            num_layers=config.encoder_num_layers,
            bidirectional=config.encoder_bidirectional,
            dropout=config.dropout
        )
        # Share embedding
        self.query_encoder.embedding = self.embedding
        
        # Context Encoder
        self.context_encoder = ContextEncoder(
            rag_dim=config.rag_context_dim,
            behavior_dim=config.behavior_context_dim,
            intent_dim=config.num_intents,
            hidden_size=config.fusion_hidden_size,
            output_dim=config.fusion_hidden_size,
            dropout=config.dropout
        )
        
        # Bridge: combine query encoding + context for decoder init
        encoder_output_dim = config.encoder_hidden_size * 2  # bidirectional
        self.bridge = nn.Sequential(
            nn.Linear(encoder_output_dim + config.fusion_hidden_size, config.decoder_hidden_size),
            nn.Tanh()
        )
        
        # Decoder
        self.decoder = AttentionDecoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.decoder_hidden_size,
            encoder_hidden=encoder_output_dim,
            attention_dim=config.attention_dim,
            num_layers=config.decoder_num_layers,
            dropout=config.dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        self.embedding.weight.data[0].zero_()  # padding
    
    def encode(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        rag_embeddings: Optional[torch.Tensor] = None,
        behavior_features: Optional[torch.Tensor] = None,
        intent_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode query with context
        
        Args:
            input_ids: (batch, seq_len)
            lengths: (batch,)
            rag_embeddings: (batch, n_docs, rag_dim)
            behavior_features: (batch, behavior_dim)
            intent_probs: (batch, num_intents) - can be provided externally
            
        Returns:
            encoder_outputs, encoder_mask, decoder_initial_state
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Classify intent if not provided
        if intent_probs is None and self.include_intent_classifier:
            intent_out = self.intent_classifier(input_ids, lengths)
            intent_probs = intent_out['probs']
        elif intent_probs is None:
            intent_probs = torch.zeros(batch_size, self.config.num_intents, device=device)
        
        # Encode query
        query_out = self.query_encoder(input_ids, lengths)
        
        # Encode context
        context = self.context_encoder(
            rag_embeddings=rag_embeddings,
            behavior_features=behavior_features,
            intent_probs=intent_probs
        )
        
        # Bridge to decoder
        combined = torch.cat([query_out['final_state'], context], dim=-1)
        decoder_init = self.bridge(combined)
        
        return {
            'encoder_outputs': query_out['encoder_outputs'],
            'encoder_mask': query_out['mask'],
            'decoder_initial_state': decoder_init,
            'intent_probs': intent_probs,
            'context': context
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        rag_embeddings: Optional[torch.Tensor] = None,
        behavior_features: Optional[torch.Tensor] = None,
        intent_probs: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            input_ids: (batch, src_len) - query tokens
            target_ids: (batch, tgt_len) - response tokens (for training)
            
        Returns:
            Dict with outputs, intent_probs, attention
        """
        # Encode
        enc_out = self.encode(
            input_ids, lengths, rag_embeddings, behavior_features, intent_probs
        )
        
        # Decode
        dec_out = self.decoder(
            encoder_outputs=enc_out['encoder_outputs'],
            encoder_mask=enc_out['encoder_mask'],
            initial_state=enc_out['decoder_initial_state'],
            target_ids=target_ids,
            teacher_forcing_ratio=teacher_forcing_ratio,
            sos_token_id=2,  # <SOS>
            eos_token_id=3   # <EOS>
        )
        
        return {
            'outputs': dec_out['outputs'],
            'predictions': dec_out['predictions'],
            'attention_weights': dec_out['attention_weights'],
            'intent_probs': enc_out['intent_probs']
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        rag_embeddings: Optional[torch.Tensor] = None,
        behavior_features: Optional[torch.Tensor] = None,
        intent_probs: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Dict[str, torch.Tensor]:
        """
        Generate response (inference)
        
        Args:
            input_ids: (batch, src_len)
            max_length: Maximum response length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Dict with sequences, attention, intent
        """
        self.eval()
        with torch.no_grad():
            # Encode
            enc_out = self.encode(
                input_ids, lengths, rag_embeddings, behavior_features, intent_probs
            )
            
            # Generate
            gen_out = self.decoder.generate(
                encoder_outputs=enc_out['encoder_outputs'],
                encoder_mask=enc_out['encoder_mask'],
                initial_state=enc_out['decoder_initial_state'],
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sos_token_id=2,
                eos_token_id=3
            )
            
            return {
                'sequences': gen_out['sequences'],
                'attention_weights': gen_out['attention_weights'],
                'intent_probs': enc_out['intent_probs'],
                'intent': enc_out['intent_probs'].argmax(dim=-1)
            }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ChatbotModel':
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        config = ModelConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Model loaded from {path}")
        return model


class ChatbotLoss(nn.Module):
    """
    Combined loss for Chatbot training
    - Response generation loss (CrossEntropy)
    - Intent classification loss (optional)
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        label_smoothing: float = 0.1,
        intent_weight: float = 0.1
    ):
        super().__init__()
        
        self.response_loss = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing
        )
        
        self.intent_loss = nn.CrossEntropyLoss()
        self.intent_weight = intent_weight
        
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        intent_logits: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses
        
        Args:
            outputs: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
            intent_logits: (batch, num_intents)
            intent_labels: (batch,)
        """
        # Response loss
        batch_size, seq_len, vocab_size = outputs.shape
        response_loss = self.response_loss(
            outputs.view(-1, vocab_size),
            targets.view(-1)
        )
        
        total_loss = response_loss
        loss_dict = {'response_loss': response_loss}
        
        # Intent loss (if provided)
        if intent_logits is not None and intent_labels is not None:
            intent_loss = self.intent_loss(intent_logits, intent_labels)
            total_loss = total_loss + self.intent_weight * intent_loss
            loss_dict['intent_loss'] = intent_loss
        
        loss_dict['total_loss'] = total_loss
        return loss_dict


def create_chatbot_model(
    vocab_size: int,
    config_name: str = 'default'
) -> ChatbotModel:
    """
    Factory function to create chatbot model
    
    Args:
        vocab_size: Vocabulary size from tokenizer
        config_name: 'small', 'default', or 'large'
    """
    from .config import SMALL_CONFIG, DEFAULT_CONFIG, LARGE_CONFIG
    
    configs = {
        'small': SMALL_CONFIG,
        'default': DEFAULT_CONFIG,
        'large': LARGE_CONFIG
    }
    
    config = configs.get(config_name, DEFAULT_CONFIG)
    config.vocab_size = vocab_size
    
    return ChatbotModel(config)
