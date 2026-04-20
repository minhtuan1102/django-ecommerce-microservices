"""
Predictor class cho Chatbot Inference - Hybrid Approach
- Intent classification: từ trained model
- Response generation: template-based enhanced với model output
"""
import torch
import random
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np

from ..dl_models.chatbot_model import ChatbotModel
from ..dl_models.config import ModelConfig
from ..dl_models.tokenizer import VietnameseTokenizer


# Response templates cho từng intent - dùng khi model output không tốt
INTENT_TEMPLATES = {
    "greeting": [
        "Xin chào! Tôi là trợ lý AI của BookStore. Tôi có thể giúp bạn tìm sách, kiểm tra đơn hàng, hoặc giải đáp thắc mắc. Bạn cần gì ạ?",
        "Chào bạn! Rất vui được hỗ trợ. Hôm nay bạn muốn tìm kiếm sách gì, hay cần hỗ trợ về đơn hàng?",
        "Hello! BookStore xin chào. Tôi sẵn sàng giúp bạn tìm sách phù hợp hoặc giải đáp mọi thắc mắc.",
    ],
    "product_query": [
        "Để tìm sách phù hợp nhất cho bạn, bạn có thể cho tôi biết thêm về thể loại hoặc chủ đề bạn quan tâm không?",
        "Tôi có thể giúp bạn tìm kiếm sách. Bạn đang quan tâm đến thể loại nào: văn học, kinh doanh, khoa học, hay lĩnh vực khác?",
        "Đây là những gợi ý dựa trên yêu cầu của bạn. Bạn muốn tôi tìm thêm không?",
    ],
    "policy_query": [
        "Về chính sách của BookStore:\n• Đổi trả: 7 ngày với sách còn nguyên seal\n• Bảo hành: Đổi mới nếu sách lỗi in ấn\n• Thanh toán: COD, chuyển khoản, ví điện tử\n\nBạn cần thêm thông tin gì không?",
        "Chính sách mua hàng của chúng tôi: Đổi trả trong 7 ngày, miễn phí ship đơn từ 300k, thanh toán linh hoạt. Bạn cần hỗ trợ thêm không?",
    ],
    "order_support": [
        "Để kiểm tra đơn hàng, bạn có thể cung cấp mã đơn hàng không? Tôi sẽ tra cứu ngay cho bạn.",
        "Tôi sẽ giúp bạn kiểm tra thông tin đơn hàng. Vui lòng cho tôi biết mã đơn hoặc số điện thoại đặt hàng.",
        "Bạn cần hỗ trợ gì về đơn hàng? Kiểm tra tình trạng, thay đổi địa chỉ, hay hủy đơn?",
    ],
    "recommendation": [
        "Dựa trên sở thích của bạn, tôi gợi ý một số sách phù hợp. Bạn thích thể loại nào nhất?",
        "Để gợi ý sách tốt nhất, bạn cho tôi biết: Sách gần đây bạn đọc và thích nhất là gì?",
        "Tôi có thể gợi ý sách dựa trên thể loại yêu thích hoặc sách bestseller. Bạn muốn chọn cách nào?",
    ],
    "general_chat": [
        "Cảm ơn bạn! Nếu cần hỗ trợ thêm về sách hoặc đơn hàng, đừng ngại hỏi nhé!",
        "Rất vui được trò chuyện! Nếu bạn cần tìm sách hay hỗ trợ, tôi luôn sẵn sàng.",
        "Cảm ơn bạn đã ghé thăm BookStore. Chúc bạn ngày tốt lành!",
    ],
}


class ChatbotPredictor:
    """
    Inference wrapper cho Chatbot Model
    
    Tích hợp:
    - Tokenization
    - Model inference
    - Post-processing
    - RAG context integration
    - Behavior features integration
    """
    
    INTENT_NAMES = [
        "greeting",
        "product_query",
        "policy_query",
        "order_support",
        "recommendation",
        "general_chat"
    ]
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = VietnameseTokenizer.load(tokenizer_path)
        
        # Load model
        self.model = ChatbotModel.load(model_path, self.device)
        self.model.eval()
        
        print(f"Chatbot loaded on {self.device}")
        print(f"  Vocab size: {self.tokenizer.vocab_len}")
        print(f"  Model config: {self.model.config.to_dict()}")
    
    def predict(
        self,
        query: str,
        rag_embeddings: Optional[np.ndarray] = None,
        behavior_features: Optional[np.ndarray] = None,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Dict:
        """
        Generate response cho query
        
        Args:
            query: User query text
            rag_embeddings: (n_docs, rag_dim) - RAG document embeddings
            behavior_features: (behavior_dim,) - Customer behavior features
            max_length: Max response length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            Dict with response, intent, confidence
        """
        # Tokenize query
        input_ids, length = self.tokenizer.encode(
            query,
            max_length=64,
            return_length=True
        )
        
        # Convert to tensors
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([length], dtype=torch.long, device=self.device)
        
        # Process optional context
        rag_tensor = None
        if rag_embeddings is not None:
            rag_tensor = torch.tensor(
                rag_embeddings, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # Add batch dim
        
        behavior_tensor = None
        if behavior_features is not None:
            behavior_tensor = torch.tensor(
                behavior_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                lengths=lengths,
                rag_embeddings=rag_tensor,
                behavior_features=behavior_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode response
        sequences = output['sequences'][0].cpu().tolist()
        model_response = self.tokenizer.decode(sequences, skip_special_tokens=True)
        
        # Get ML model intent
        intent_probs = output['intent_probs'][0].cpu().numpy()
        intent_idx = int(output['intent'][0].cpu().item())
        ml_intent = self.INTENT_NAMES[intent_idx]
        ml_confidence = float(intent_probs[intent_idx])
        
        # Hybrid: use rule-based if confident, else ML
        rule_intent = self._rule_based_intent(query)
        if rule_intent:
            intent_name = rule_intent
            intent_confidence = 0.95  # High confidence for rule-based
        else:
            intent_name = ml_intent
            intent_confidence = ml_confidence
        
        # Hybrid response: use template if model output is not good
        response = self._get_best_response(model_response, intent_name, intent_confidence)
        
        return {
            'response': response,
            'model_response': model_response,
            'intent': intent_name,
            'intent_confidence': intent_confidence,
            'ml_intent': ml_intent,
            'rule_intent': rule_intent,
            'intent_probs': {
                name: float(prob) 
                for name, prob in zip(self.INTENT_NAMES, intent_probs)
            }
        }
    
    def _get_best_response(
        self,
        model_response: str,
        intent: str,
        confidence: float
    ) -> str:
        """
        Hybrid approach: dùng model response nếu tốt, fallback to template
        """
        # Response quá ngắn hoặc chứa pattern lặp = không tốt
        is_good_response = (
            len(model_response) > 30 and  # Đủ dài
            len(set(model_response.split())) > 10 and  # Đủ diverse
            model_response.count(model_response[:10]) < 3  # Không lặp
        )
        
        if is_good_response:
            # Polish the response
            return self._polish_response(model_response)
        
        # Use template with some variation
        templates = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["general_chat"])
        return random.choice(templates)
    
    def _polish_response(self, response: str) -> str:
        """Clean up and polish model response"""
        # Remove duplicate consecutive words
        words = response.split()
        polished = []
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                polished.append(word)
        response = ' '.join(polished)
        
        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:]
        
        # Ensure proper ending
        if response and response[-1] not in '.!?':
            response += '.'
            
        return response
    
    def _rule_based_intent(self, query: str) -> Optional[str]:
        """
        Rule-based intent detection as fallback/enhancement
        Returns None if no confident match
        """
        query_lower = query.lower()
        
        # Greeting patterns
        greeting_patterns = ['xin chào', 'chào', 'hello', 'hi', 'hey', 'xin chao', 'chao']
        if any(p in query_lower for p in greeting_patterns):
            return 'greeting'
        
        # Thank you / goodbye patterns  
        goodbye_patterns = ['cảm ơn', 'cam on', 'thanks', 'thank', 'tạm biệt', 'bye', 'goodbye']
        if any(p in query_lower for p in goodbye_patterns):
            return 'general_chat'
        
        # Policy patterns
        policy_patterns = ['chính sách', 'chinh sach', 'đổi trả', 'doi tra', 'bảo hành', 
                          'thanh toán', 'ship', 'giao hàng', 'phí', 'miễn phí']
        if any(p in query_lower for p in policy_patterns):
            return 'policy_query'
        
        # Order patterns
        order_patterns = ['đơn hàng', 'don hang', 'order', 'mã đơn', 'kiểm tra đơn', 
                         'tình trạng đơn', 'hủy đơn', 'theo dõi']
        if any(p in query_lower for p in order_patterns):
            return 'order_support'
        
        # Recommendation patterns
        recommend_patterns = ['gợi ý', 'goi y', 'đề xuất', 'recommend', 'nên đọc', 
                             'sách hay', 'phù hợp với tôi', 'tương tự']
        if any(p in query_lower for p in recommend_patterns):
            return 'recommendation'
        
        # Product query patterns
        product_patterns = ['tìm', 'tim', 'sách', 'sach', 'book', 'giá', 'gia', 
                           'bao nhiêu', 'còn không', 'có bán']
        if any(p in query_lower for p in product_patterns):
            return 'product_query'
        
        return None
    
    def classify_intent(self, query: str) -> Dict:
        """
        Classify intent only (faster than full prediction)
        """
        input_ids, length = self.tokenizer.encode(query, max_length=64, return_length=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([length], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if self.model.include_intent_classifier:
                intent_out = self.model.intent_classifier(input_ids, lengths)
                probs = intent_out['probs'][0].cpu().numpy()
                predicted = int(intent_out['predicted'][0].cpu().item())
            else:
                # Use full encode to get intent
                enc_out = self.model.encode(input_ids, lengths)
                probs = enc_out['intent_probs'][0].cpu().numpy()
                predicted = int(probs.argmax())
        
        return {
            'intent': self.INTENT_NAMES[predicted],
            'confidence': float(probs[predicted]),
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.INTENT_NAMES, probs)
            }
        }
    
    def batch_predict(
        self,
        queries: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Predict multiple queries (not batched for simplicity)
        """
        return [self.predict(q, **kwargs) for q in queries]


class ChatbotService:
    """
    High-level service integrating:
    - ChatbotPredictor
    - RAG retrieval
    - Behavior model
    """
    
    def __init__(
        self,
        predictor: ChatbotPredictor,
        rag_retriever=None,
        behavior_model=None
    ):
        self.predictor = predictor
        self.rag_retriever = rag_retriever
        self.behavior_model = behavior_model
    
    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        customer_id: Optional[str] = None
    ) -> Dict:
        """
        Full chat response with context
        """
        # Get RAG context
        rag_embeddings = None
        if self.rag_retriever:
            try:
                docs = self.rag_retriever.retrieve(query, top_k=3)
                rag_embeddings = np.array([d['embedding'] for d in docs])
            except Exception as e:
                print(f"RAG retrieval failed: {e}")
        
        # Get behavior features
        behavior_features = None
        if self.behavior_model and customer_id:
            try:
                behavior_out = self.behavior_model.predict(customer_id)
                behavior_features = behavior_out.get('embeddings')
            except Exception as e:
                print(f"Behavior prediction failed: {e}")
        
        # Generate response
        result = self.predictor.predict(
            query=query,
            rag_embeddings=rag_embeddings,
            behavior_features=behavior_features
        )
        
        return result


def load_predictor(
    model_dir: str,
    device: str = None
) -> ChatbotPredictor:
    """
    Load predictor from saved directory
    
    Expected structure:
    model_dir/
        best_model.pt
        tokenizer.pkl
    """
    model_dir = Path(model_dir)
    
    model_path = model_dir / "best_model.pt"
    tokenizer_path = model_dir / "tokenizer.pkl"
    
    return ChatbotPredictor(
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        device=device
    )
