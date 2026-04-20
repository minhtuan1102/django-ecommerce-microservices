"""
RAG Chain - Kết hợp Retrieval và Generation
"""
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from .retriever import KnowledgeRetriever, Document
from .generator import ResponseGenerator

logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Response object cho chat API"""
    response: str
    session_id: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    customer_id: Optional[str] = None
    personalized: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'response': self.response,
            'session_id': self.session_id,
            'sources': self.sources,
            'customer_id': self.customer_id,
            'personalized': self.personalized,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class ConsultingChain:
    """
    Consulting Chain kết hợp RAG retrieval và generation
    với tích hợp behavior analysis cho personalization
    """
    
    def __init__(
        self,
        retriever: KnowledgeRetriever = None,
        generator: ResponseGenerator = None,
        behavior_client = None
    ):
        """
        Khởi tạo consulting chain
        
        Args:
            retriever: KnowledgeRetriever instance
            generator: ResponseGenerator instance
            behavior_client: BehaviorClient instance (optional)
        """
        self.retriever = retriever or KnowledgeRetriever()
        self.generator = generator or ResponseGenerator()
        self.behavior_client = behavior_client
        
        logger.info("ConsultingChain initialized")
    
    def process(
        self,
        query: str,
        customer_id: str = None,
        session_id: str = None,
        category: str = None,
        k: int = 5
    ) -> ChatResponse:
        """
        Xử lý query và generate response
        
        Args:
            query: Câu hỏi của khách hàng
            customer_id: ID khách hàng (optional)
            session_id: ID session chat (optional)
            category: Category để filter search (optional)
            k: Số documents để retrieve
            
        Returns:
            ChatResponse object
        """
        import uuid
        
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            # Step 1: Lấy phân tích hành vi từ Deep Learning Model (Local)
            behavior_analysis = {}
            customer_context = ""
            customer_info = None
            
            if self.behavior_client and customer_id:
                try:
                    # Gọi Deep Learning model để lấy "Hồ sơ tâm lý" khách hàng
                    behavior_analysis = self.behavior_client.get_customer_analysis(customer_id)
                    
                    # Lấy ngữ cảnh đồ thị (GraphRAG)
                    graph_context = self.behavior_client.get_graph_context(customer_id)
                    
                    segment = behavior_analysis.get('segment', 'Khách hàng mới')
                    predicted_cats = behavior_analysis.get('predicted_categories', [])
                    churn_risk = behavior_analysis.get('churn_risk', 0.0)
                    
                    # Tạo ngữ cảnh tư vấn dựa trên Model Deep Learning + Graph
                    customer_context = f"\n- Phân khúc khách hàng: {segment}"
                    
                    # Thêm thông tin từ Graph (recent interactions)
                    recent = graph_context.get('recent_interactions', [])
                    if recent:
                        recent_products = ", ".join([str(r.get('product_id')) for r in recent[:3]])
                        customer_context += f"\n- Sản phẩm tương tác gần đây (từ Graph): {recent_products}"
                    
                    if predicted_cats:
                        cats = ", ".join([c.get('category', c) if isinstance(c, dict) else str(c) for c in predicted_cats[:2]])
                        customer_context += f"\n- Danh mục quan tâm nhất (Dự đoán từ AI): {cats}"
                    
                    if churn_risk > 0.6:
                        customer_context += "\n- Trạng thái: Khách hàng có nguy cơ rời bỏ, cần ưu đãi đặc biệt để giữ chân."
                    
                    # Build customer_info for generator
                    customer_info = {
                        **behavior_analysis,
                        'graph_context': graph_context,
                        'ai_context': customer_context
                    }
                        
                except Exception as e:
                    logger.warning(f"Could not get behavior analysis from DL model: {e}")
            
            # Step 2: Tìm kiếm sản phẩm phù hợp nhất (RAG)
            # Nếu có category filter, sử dụng nó
            if category:
                documents = self.retriever.search_by_category(query, category, k)
            else:
                # Nếu model DL dự đoán khách thích category nào, ưu tiên tìm sản phẩm đó
                search_query = query
                if behavior_analysis.get('predicted_categories'):
                    top_cats = behavior_analysis['predicted_categories'][:2]
                    cat_names = [c.get('category', c) if isinstance(c, dict) else str(c) for c in top_cats]
                    search_query = f"{query} {' '.join(cat_names)}"
                
                documents = self.retriever.retrieve(search_query, k)
            
            # Step 3: Tạo câu trả lời cá nhân hóa
            response_text = self.generator.generate(
                query=query,
                context=documents,
                customer_info=customer_info
            )
            
            # Step 4: Format sources
            sources = self._format_sources(documents)
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                sources=sources,
                customer_id=customer_id,
                personalized=customer_info is not None,
                metadata={
                    'documents_retrieved': len(documents),
                    'category_filter': category,
                    'model': self.generator.model_name,
                    'behavior_segment': behavior_analysis.get('segment') if behavior_analysis else None
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return ChatResponse(
                response=self._get_error_response(),
                session_id=session_id,
                sources=[],
                customer_id=customer_id,
                metadata={'error': str(e)}
            )
    
    def get_personalized_response(
        self,
        query: str,
        customer_id: str,
        session_id: str = None,
        k: int = 5
    ) -> ChatResponse:
        """
        Generate personalized response dựa trên behavior analysis
        
        Args:
            query: Câu hỏi của khách hàng
            customer_id: ID khách hàng (required)
            session_id: ID session chat (optional)
            k: Số documents để retrieve
            
        Returns:
            ChatResponse object với personalized response
        """
        import uuid
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            # Step 1: Retrieve relevant documents
            documents = self.retriever.retrieve(query, k)
            
            # Step 2: Get behavior analysis
            behavior_analysis = {}
            if self.behavior_client:
                try:
                    # Get full customer analysis
                    behavior_analysis = self.behavior_client.get_customer_analysis(customer_id)
                    
                    # Get segment
                    segment = self.behavior_client.get_customer_segment(customer_id)
                    behavior_analysis['segment'] = segment
                    
                    # Get recommendations
                    recommendations = self.behavior_client.get_recommendations(customer_id)
                    
                    # Lấy sản phẩm thật từ RAG dựa trên hành vi phân tích
                    predicted_categories = behavior_analysis.get('predicted_categories', [])
                    favorite_categories = behavior_analysis.get('favorite_categories', [])
                    
                    search_cats = []
                    if predicted_categories:
                        search_cats = [c['category'] for c in predicted_categories]
                    elif favorite_categories:
                        search_cats = favorite_categories
                        
                    if search_cats:
                        search_query = " ".join(search_cats[:2])
                        product_docs = self.search_products(search_query, k=3)
                        
                        actual_recommendations = []
                        for doc in product_docs:
                            metadata = doc.metadata or {}
                            product_name = metadata.get('Product', metadata.get('name'))
                            item_id = metadata.get('id', metadata.get('item_id', 'unknown'))
                            item_type = metadata.get('item_type', 'book')
                            price = metadata.get('price', 'Liên hệ')
                            
                            # Cố gắng lấy thêm từ content nếu metadata thiếu
                            lines = doc.content.split('\n')
                            for line in lines:
                                if not product_name and 'Product:' in line:
                                    product_name = line.split('Product:')[1].strip()
                                elif 'item_type:' in line:
                                    item_type = line.split('item_type:')[1].strip()
                                elif 'price:' in line:
                                    price = line.split('price:')[1].strip()
                            
                            if not product_name:
                                product_name = 'Sản phẩm'
                            
                            # Gán ID giả nếu trong metadata chưa có
                            if item_id == 'unknown':
                                import hashlib
                                item_id = str(int(hashlib.md5(product_name.encode()).hexdigest()[:8], 16) % 100)
                                
                            actual_recommendations.append({
                                'id': item_id,
                                'type': item_type,
                                'name': product_name,
                                'price': price,
                                'author': metadata.get('author', metadata.get('brand', '')),
                                'category': metadata.get('category', '')
                            })
                            
                        if actual_recommendations:
                            recommendations = actual_recommendations

                    behavior_analysis['recommendations'] = recommendations
                    
                except Exception as e:
                    logger.warning(f"Could not get behavior analysis: {e}")
                    behavior_analysis = self._get_mock_behavior_analysis(customer_id)
            else:
                behavior_analysis = self._get_mock_behavior_analysis(customer_id)
            
            # Step 3: Generate personalized response
            response_text = self.generator.generate_with_behavior(
                query=query,
                context=documents,
                behavior_analysis=behavior_analysis
            )
            
            # Step 4: Format sources
            sources = self._format_sources(documents)
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                sources=sources,
                customer_id=customer_id,
                personalized=True,
                metadata={
                    'documents_retrieved': len(documents),
                    'segment': behavior_analysis.get('segment'),
                    'recommendations_count': len(behavior_analysis.get('recommendations', [])),
                    'model': self.generator.model_name
                }
            )
            
        except Exception as e:
            logger.error(f"Error in personalized response: {e}")
            # Fallback to regular processing
            return self.process(query, customer_id, session_id)
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format documents thành sources list"""
        sources = []
        for doc in documents:
            source = {
                'content': doc.content[:200] + '...' if len(doc.content) > 200 else doc.content,
                'score': round(doc.score, 3) if doc.score else None,
                'metadata': doc.metadata
            }
            sources.append(source)
        return sources
    
    def _get_mock_behavior_analysis(self, customer_id: str) -> Dict[str, Any]:
        """Get mock behavior analysis khi không có behavior service"""
        return {
            'customer_id': customer_id,
            'segment': 'Regular',
            'favorite_categories': ['Văn học', 'Self-help'],
            'purchase_frequency': 'Trung bình',
            'avg_order_value': '200.000đ',
            'recommendations': [
                {'id': '1', 'type': 'book', 'name': 'Đắc Nhân Tâm', 'price': '89.000đ'},
                {'id': '2', 'type': 'book', 'name': 'Nhà Giả Kim', 'price': '79.000đ'},
                {'id': '3', 'type': 'book', 'name': 'Tuổi Trẻ Đáng Giá Bao Nhiêu', 'price': '75.000đ'}
            ]
        }
    
    def _get_error_response(self) -> str:
        """Get error response message"""
        return """Xin lỗi, tôi đang gặp sự cố kỹ thuật và không thể xử lý yêu cầu của bạn.

Vui lòng thử lại sau hoặc liên hệ:
📞 Hotline: 1900-1234
📧 Email: support@nhasachonline.vn

Xin cảm ơn bạn đã thông cảm!"""
    
    def search_products(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Tìm kiếm sản phẩm
        
        Args:
            query: Từ khóa tìm kiếm
            k: Số kết quả
            
        Returns:
            List các Document về sản phẩm
        """
        return self.retriever.search_by_category(query, 'products', k)
    
    def search_policies(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Tìm kiếm chính sách
        
        Args:
            query: Từ khóa tìm kiếm
            k: Số kết quả
            
        Returns:
            List các Document về chính sách
        """
        return self.retriever.search_by_category(query, 'policies', k)
    
    def search_faqs(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Tìm kiếm FAQs
        
        Args:
            query: Từ khóa tìm kiếm
            k: Số kết quả
            
        Returns:
            List các Document về FAQs
        """
        return self.retriever.search_by_category(query, 'faqs', k)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status của chain"""
        status = {
            'status': 'healthy',
            'components': {}
        }
        
        # Check retriever
        try:
            retriever_stats = self.retriever.get_collection_stats()
            status['components']['retriever'] = {
                'status': 'healthy' if 'error' not in retriever_stats else 'degraded',
                'stats': retriever_stats
            }
        except Exception as e:
            status['components']['retriever'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            status['status'] = 'degraded'
        
        # Check generator
        status['components']['generator'] = {
            'status': 'healthy',
            'model': self.generator.model_name,
            'using_mock': self.generator._use_mock
        }
        
        # Check behavior client
        if self.behavior_client:
            try:
                health = self.behavior_client.health_check()
                status['components']['behavior_client'] = {
                    'status': 'healthy' if health else 'unhealthy'
                }
            except Exception as e:
                status['components']['behavior_client'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            status['components']['behavior_client'] = {
                'status': 'not_configured'
            }
        
        return status
