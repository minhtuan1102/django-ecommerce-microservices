"""
RAG Generator - Response generation với Local AI Model hoặc Gemini/OpenAI API
"""
import os
import logging
import random
from typing import List, Optional, Dict, Any
from pathlib import Path

from django.conf import settings

from .retriever import Document
from .gemini_client import GeminiClient, GEMINI_AVAILABLE
from .prompts import (
    select_prompt_template,
    get_customer_info_section,
    PERSONALIZED_PROMPT,
    FALLBACK_PROMPT,
    SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

# Try to import local model
LOCAL_MODEL_AVAILABLE = False
try:
    from ..inference.predictor import ChatbotPredictor, load_predictor
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    logger.warning("Local chatbot model not available. Will use API or mock.")


class ResponseGenerator:
    """
    Response Generator với 3 modes:
    1. Local AI Model (trained Seq2Seq) - preferred
    2. Gemini / OpenAI / Ollama API - fallback
    3. Mock responses - last resort
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        use_local_model: bool = True
    ):
        """
        Khởi tạo generator
        """
        # Gemini Settings
        self.gemini_api_key = getattr(settings, 'GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))
        self.gemini_model = getattr(settings, 'GEMINI_MODEL', os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash'))
        
        # OpenAI/Ollama Settings
        self.openai_api_key = getattr(settings, 'OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY', 'ollama'))
        self.openai_model = model_name or getattr(settings, 'LLM_MODEL', 'llama3')
        self.base_url = getattr(settings, 'LLM_BASE_URL', os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1'))
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client = None
        self._client_type = None # 'gemini' or 'openai'
        self._local_predictor = None
        self._use_mock = False
        self._use_local_model = use_local_model
        
        # Try to load local model
        if use_local_model and LOCAL_MODEL_AVAILABLE:
            self._initialize_local_model()
    
    def _initialize_local_model(self):
        """Initialize local trained chatbot model"""
        try:
            model_dir = Path(__file__).parent.parent.parent / "saved_models"
            model_path = model_dir / "best_model.pt"
            tokenizer_path = model_dir / "tokenizer.pkl"
            
            if model_path.exists() and tokenizer_path.exists():
                self._local_predictor = ChatbotPredictor(
                    model_path=str(model_path),
                    tokenizer_path=str(tokenizer_path)
                )
                logger.info("Local AI model loaded successfully!")
            else:
                logger.info(f"Local model not found at {model_dir}. Will use API or mock.")
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            self._local_predictor = None
    
    def _initialize_client(self):
        """Lazy initialization của LLM client (Gemini preferred, then OpenAI)"""
        if self._client is not None:
            return
            
        # 1. Try Gemini
        if self.gemini_api_key and GEMINI_AVAILABLE:
            try:
                self._client = GeminiClient(
                    api_key=self.gemini_api_key,
                    model_name=self.gemini_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                self._client_type = 'gemini'
                logger.info(f"Initialized Gemini client with model: {self.gemini_model}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

        # 2. Fallback to OpenAI/Ollama
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.base_url
            )
            self._client_type = 'openai'
            logger.info(f"Initialized OpenAI client at {self.base_url} with model: {self.openai_model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self._use_mock = True
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents thành context string"""
        if not documents:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.content if hasattr(doc, 'content') else str(doc)
            category = doc.metadata.get('category', 'general') if hasattr(doc, 'metadata') else 'general'
            context_parts.append(f"{i}. [{category}] {content}")
        
        return '\n\n'.join(context_parts)
    
    def generate(
        self,
        query: str,
        context: List[Document],
        customer_info: Dict[str, Any] = None
    ) -> str:
        """
        Generate response dựa trên query và context
        """
        # Try local model first
        if self._local_predictor is not None:
            try:
                rag_embeddings = None
                if context:
                    embeddings = []
                    for doc in context[:3]:
                        if hasattr(doc, 'embedding') and doc.embedding is not None:
                            embeddings.append(doc.embedding)
                    if embeddings:
                        import numpy as np
                        rag_embeddings = np.array(embeddings)
                
                result = self._local_predictor.predict(
                    query=query,
                    rag_embeddings=rag_embeddings,
                    temperature=self.temperature
                )
                
                response = result['response']
                intent = result['intent']
                
                if intent == 'product_query' and context:
                    response = self._enhance_with_products(response, context)
                elif intent == 'policy_query' and context:
                    response = self._enhance_with_context(response, context)
                
                return response
            except Exception as e:
                logger.warning(f"Local model failed: {e}, falling back to API")
        
        # Fallback to API
        self._initialize_client()
        
        if self._use_mock:
            return self._generate_mock_response(query, context, customer_info)
            
        # Prepare prompts
        context_str = self._format_context(context)
        context_category = context[0].metadata.get('category') if context and hasattr(context[0], 'metadata') else None
        prompt_template = select_prompt_template(query, context_category)
        customer_info_section = get_customer_info_section(customer_info) if customer_info else ""
        
        prompt = prompt_template.format(
            query=query,
            context=context_str,
            customer_info_section=customer_info_section
        )
        
        try:
            if self._client_type == 'gemini':
                return self._client.chat_simple(prompt, system_prompt=SYSTEM_PROMPT)
            else:
                response = self._client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API generation error ({self._client_type}): {e}")
            return self._generate_mock_response(query, context, customer_info)

    def generate_with_behavior(
        self,
        query: str,
        context: List[Document],
        behavior_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate personalized response dựa trên behavior analysis
        """
        # Try local model first
        if self._local_predictor is not None:
            try:
                rag_embeddings = None
                if context:
                    embeddings = []
                    for doc in context[:3]:
                        if hasattr(doc, 'embedding') and doc.embedding is not None:
                            embeddings.append(doc.embedding)
                    if embeddings:
                        import numpy as np
                        rag_embeddings = np.array(embeddings)
                
                behavior_features = None
                if 'embeddings' in behavior_analysis:
                    import numpy as np
                    behavior_features = np.array(behavior_analysis['embeddings'])
                
                result = self._local_predictor.predict(
                    query=query,
                    rag_embeddings=rag_embeddings,
                    behavior_features=behavior_features,
                    temperature=self.temperature
                )
                
                response = result['response']
                segment = behavior_analysis.get('segment', '')
                recommendations = behavior_analysis.get('recommendations', [])
                
                if segment:
                    response = self._add_personalized_intro(response, segment)
                if recommendations:
                    response = self._add_recommendations(response, recommendations)
                
                return response
            except Exception as e:
                logger.warning(f"Local model failed for personalized: {e}, falling back")
        
        # Fallback to API
        self._initialize_client()
        
        if self._use_mock:
            return self._generate_mock_personalized_response(query, context, behavior_analysis)
            
        # Format prompts
        context_str = self._format_context(context)
        customer_segment = behavior_analysis.get('segment', 'Khách hàng mới')
        favorite_categories = ', '.join(behavior_analysis.get('favorite_categories', ['Chưa xác định']))
        purchase_frequency = behavior_analysis.get('purchase_frequency', 'Chưa có dữ liệu')
        avg_order_value = behavior_analysis.get('avg_order_value', 'Chưa có dữ liệu')
        recommendations = self._format_recommendations(behavior_analysis.get('recommendations', []))
        
        prompt = PERSONALIZED_PROMPT.format(
            query=query,
            context=context_str,
            customer_segment=customer_segment,
            favorite_categories=favorite_categories,
            purchase_frequency=purchase_frequency,
            avg_order_value=avg_order_value,
            recommendations=recommendations
        )
        
        try:
            system_p = SYSTEM_PROMPT + "\n\nBạn đang ở chế độ tư vấn CÁ NHÂN HÓA - hãy thể hiện sự hiểu biết về sở thích khách hàng."
            if self._client_type == 'gemini':
                return self._client.chat_simple(prompt, system_prompt=system_p)
            else:
                response = self._client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_p},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API personalized error ({self._client_type}): {e}")
            return self._generate_mock_personalized_response(query, context, behavior_analysis)

    def _enhance_with_products(self, response: str, context: List[Document]) -> str:
        """Enhance response with product details from RAG context"""
        if not context:
            return response
        
        products = []
        for doc in context[:3]:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            product_name = metadata.get('name', metadata.get('Product', metadata.get('title')))
            price = metadata.get('price', '')
            category = metadata.get('category', '')
            
            if not product_name:
                lines = content.split('\n')
                for line in lines:
                    if 'Product:' in line:
                        product_name = line.split('Product:')[1].strip()
                    elif 'price:' in line:
                        price = line.split('price:')[1].strip()
                    elif 'category:' in line:
                        category = line.split('category:')[1].strip()
            
            if product_name and product_name not in ['Sản phẩm', 'Unknown']:
                products.append({'name': product_name, 'price': price, 'category': category})
        
        if not products:
            return response
        
        product_section = "\n\n📚 **Gợi ý sản phẩm:**\n"
        for i, prod in enumerate(products, 1):
            product_section += f"\n{i}. **{prod['name']}**"
            if prod['price']:
                product_section += f" - {prod['price']}đ"
            if prod['category']:
                product_section += f" ({prod['category']})"
            product_section += "\n"
        
        product_section += "\n💡 Bạn có thể tìm kiếm sách trên trang chủ để xem chi tiết và đặt hàng nhé!"
        return response + product_section

    def _enhance_with_context(self, response: str, context: List[Document]) -> str:
        """Enhance response with policy/FAQ context"""
        if not context:
            return response
        
        doc = context[0]
        content = doc.content if hasattr(doc, 'content') else str(doc)
        key_points = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') and len(line) > 3:
                point = line[1:].strip()
                if not any(tech in point.lower() for tech in ['api', 'endpoint', '/', 'http']):
                    if len(point) < 100:
                        key_points.append(point)
        
        if not key_points:
            return response
        
        context_section = "\n\n📋 **Thông tin chi tiết:**\n"
        for point in key_points[:4]:
            context_section += f"• {point}\n"
        
        return response + context_section

    def _add_personalized_intro(self, response: str, segment: str) -> str:
        """Add personalized greeting based on segment"""
        intros = {
            'VIP': "🌟 Chào khách hàng VIP! ",
            'Premium': "💎 Chào khách hàng Premium! ",
            'Regular': "👋 Chào bạn! ",
            'New': "🎉 Chào mừng khách hàng mới! ",
        }
        return intros.get(segment, "") + response

    def _add_recommendations(self, response: str, recommendations: List[Dict]) -> str:
        """Add product recommendations to response"""
        if not recommendations:
            return response
        
        rec_section = "\n\n🎯 **Gợi ý dành riêng cho bạn:**\n"
        for i, rec in enumerate(recommendations[:3], 1):
            name = rec.get('name', rec.get('title', 'Sản phẩm'))
            price = rec.get('price', '')
            item_id = rec.get('id', '')
            rec_section += f"\n{i}. **{name}**"
            if price:
                rec_section += f" - {price}"
            if item_id:
                rec_section += f"\n   👉 [Xem chi tiết](/store/book/{item_id}/)"
            rec_section += "\n"
        return response + rec_section

    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        """Format recommendations list"""
        if not recommendations:
            return "Chưa có đề xuất cụ thể"

        rec_parts = []
        for i, rec in enumerate(recommendations[:5], 1):
            name = rec.get('name', rec.get('title', 'Sản phẩm'))
            price = rec.get('price', 'Liên hệ')
            item_id = rec.get('id', 'unknown')
            item_type = rec.get('type', rec.get('item_type', 'book'))
            author = rec.get('author', rec.get('brand', ''))
            info = f"{i}. {name} - Giá: {price}"
            if author:
                info += f" - Tác giả/Thương hiệu: {author}"
            info += f" [ID: {item_id}, Type: {item_type}]"
            rec_parts.append(info)
        return '\n'.join(rec_parts)

    def _generate_mock_response(self, query: str, context: List[Document], customer_info: Dict[str, Any] = None) -> str:
        """Generate mock response"""
        return "Xin chào! Tôi là chatbot tư vấn. Hiện tại hệ thống AI đang bận, vui lòng thử lại sau hoặc liên hệ Hotline: 1900-1234."

    def _generate_mock_personalized_response(self, query: str, context: List[Document], behavior_analysis: Dict[str, Any]) -> str:
        """Generate mock personalized response"""
        segment = behavior_analysis.get('segment', 'Khách hàng')
        return f"Chào {segment}! Tôi là trợ lý ảo cá nhân của bạn. Hiện tại tôi chưa thể kết nối với bộ não AI Gemini, nhưng tôi vẫn luôn ở đây hỗ trợ bạn!"
