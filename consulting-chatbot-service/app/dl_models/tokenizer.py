"""
Vietnamese Tokenizer cho AI Chatbot
Sử dụng underthesea cho word segmentation + custom vocabulary
"""
import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False
    print("Warning: underthesea not installed. Using simple tokenizer.")


class VietnameseTokenizer:
    """
    Tokenizer cho tiếng Việt
    - Word segmentation với underthesea
    - Build vocabulary từ corpus
    - Encode/Decode text
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
    
    def __init__(
        self,
        vocab_size: int = 10000,
        max_length: int = 128,
        min_freq: int = 2
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_freq = min_freq
        
        # Initialize vocabulary with special tokens
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._init_special_tokens()
        
        # Statistics
        self.word_freq: Counter = Counter()
        self.is_fitted = False
        
    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    @property
    def pad_token_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def sos_token_id(self) -> int:
        return self.word2idx[self.SOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.word2idx[self.EOS_TOKEN]
    
    @property
    def vocab_len(self) -> int:
        return len(self.word2idx)
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text before tokenization"""
        # Lowercase
        text = text.lower().strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Keep Vietnamese characters, numbers, basic punctuation
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ.,!?]', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = self._preprocess(text)
        
        if not text:
            return []
        
        if HAS_UNDERTHESEA:
            # Use underthesea for Vietnamese word segmentation
            tokens = word_tokenize(text, format="text").split()
        else:
            # Simple whitespace tokenization
            tokens = text.split()
        
        # Split compound words by underscore (underthesea format)
        result = []
        for token in tokens:
            if '_' in token:
                # Keep compound word as is (e.g., "thể_loại" -> "thể_loại")
                result.append(token.replace('_', ' '))
            else:
                result.append(token)
        
        return result
    
    def fit(self, texts: List[str], verbose: bool = True):
        """
        Build vocabulary from corpus
        
        Args:
            texts: List of text documents
            verbose: Print progress
        """
        if verbose:
            print(f"Building vocabulary from {len(texts)} documents...")
        
        # Count word frequencies
        self.word_freq = Counter()
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            self.word_freq.update(tokens)
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(texts)} documents")
        
        # Filter by frequency and vocab size
        filtered_words = [
            word for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ]
        
        # Limit to vocab_size - special_tokens
        max_words = self.vocab_size - len(self.SPECIAL_TOKENS)
        filtered_words = filtered_words[:max_words]
        
        # Build vocabulary
        self._init_special_tokens()  # Reset
        for word in filtered_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.is_fitted = True
        
        if verbose:
            print(f"Vocabulary built: {len(self.word2idx)} words")
            print(f"  Total unique words: {len(self.word_freq)}")
            print(f"  Words meeting min_freq={self.min_freq}: {len(filtered_words)}")
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_length: bool = False
    ) -> Tuple[List[int], int] | List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Add SOS/EOS tokens
            max_length: Override default max_length
            padding: Pad to max_length
            return_length: Return actual length
            
        Returns:
            Token IDs (and length if return_length=True)
        """
        max_len = max_length or self.max_length
        tokens = self._tokenize(text)
        
        # Convert to IDs
        ids = [self.word2idx.get(t, self.unk_token_id) for t in tokens]
        
        # Add special tokens
        if add_special_tokens:
            ids = [self.sos_token_id] + ids + [self.eos_token_id]
        
        # Track actual length
        actual_length = len(ids)
        
        # Truncate if needed
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [self.eos_token_id]
            actual_length = max_len
        
        # Pad if needed
        if padding and len(ids) < max_len:
            ids = ids + [self.pad_token_id] * (max_len - len(ids))
        
        if return_length:
            return ids, actual_length
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text
        
        Args:
            ids: Token IDs
            skip_special_tokens: Skip PAD, SOS, EOS tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        for idx in ids:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    if token == self.EOS_TOKEN:
                        break  # Stop at EOS
                    continue
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        **kwargs
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Encode multiple texts
        
        Returns:
            (token_ids, lengths)
        """
        results = [self.encode(text, return_length=True, **kwargs) for text in texts]
        ids = [r[0] for r in results]
        lengths = [r[1] for r in results]
        return ids, lengths
    
    def batch_decode(self, batch_ids: List[List[int]], **kwargs) -> List[str]:
        """Decode multiple sequences"""
        return [self.decode(ids, **kwargs) for ids in batch_ids]
    
    def save(self, path: str):
        """Save tokenizer to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'min_freq': self.min_freq,
            'word2idx': self.word2idx,
            'word_freq': dict(self.word_freq),
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'VietnameseTokenizer':
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            max_length=data['max_length'],
            min_freq=data['min_freq']
        )
        
        tokenizer.word2idx = data['word2idx']
        tokenizer.idx2word = {v: k for k, v in data['word2idx'].items()}
        tokenizer.word_freq = Counter(data['word_freq'])
        tokenizer.is_fitted = data['is_fitted']
        
        print(f"Tokenizer loaded from {path}, vocab size: {len(tokenizer.word2idx)}")
        return tokenizer
    
    def get_statistics(self) -> Dict:
        """Get tokenizer statistics"""
        return {
            'vocab_size': len(self.word2idx),
            'max_length': self.max_length,
            'total_unique_words': len(self.word_freq),
            'most_common': self.word_freq.most_common(20),
            'is_fitted': self.is_fitted
        }


def create_tokenizer_from_knowledge_base(
    knowledge_base_path: str,
    vocab_size: int = 10000,
    max_length: int = 128
) -> VietnameseTokenizer:
    """
    Create and fit tokenizer from knowledge base documents
    
    Args:
        knowledge_base_path: Path to knowledge base documents
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        
    Returns:
        Fitted tokenizer
    """
    from pathlib import Path
    
    kb_path = Path(knowledge_base_path)
    texts = []
    
    # Read all markdown files
    for md_file in kb_path.rglob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by lines for more training data
            texts.extend([line.strip() for line in content.split('\n') if line.strip()])
    
    print(f"Found {len(texts)} text segments from knowledge base")
    
    # Add common Vietnamese e-commerce phrases
    ecommerce_phrases = [
        "xin chào", "cảm ơn bạn", "chào bạn",
        "sản phẩm", "đơn hàng", "giao hàng", "thanh toán",
        "giá", "khuyến mãi", "giảm giá", "voucher",
        "đổi trả", "hoàn tiền", "bảo hành",
        "sách", "tác giả", "thể loại", "xuất bản",
        "laptop", "điện thoại", "máy tính",
        "tôi muốn", "tôi cần", "cho tôi hỏi", "làm ơn",
        "được không", "có thể", "như thế nào",
        "đắt quá", "rẻ hơn", "chất lượng",
        "gợi ý", "đề xuất", "tư vấn",
        "tốt nhất", "phổ biến", "bán chạy"
    ]
    texts.extend(ecommerce_phrases)
    
    # Create and fit tokenizer
    tokenizer = VietnameseTokenizer(
        vocab_size=vocab_size,
        max_length=max_length
    )
    tokenizer.fit(texts)
    
    return tokenizer
