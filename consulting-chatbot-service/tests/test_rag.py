"""
Tests for RAG components
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.rag.retriever import KnowledgeRetriever, Document
from app.rag.generator import ResponseGenerator
from app.rag.chain import ConsultingChain, ChatResponse
from app.rag.prompts import select_prompt_template, get_customer_info_section


class TestDocument:
    """Tests for Document dataclass"""
    
    def test_document_creation(self):
        doc = Document(
            content="Test content",
            metadata={'category': 'products'},
            score=0.95
        )
        assert doc.content == "Test content"
        assert doc.metadata['category'] == 'products'
        assert doc.score == 0.95
    
    def test_document_page_content_alias(self):
        doc = Document(content="Test", metadata={})
        assert doc.page_content == "Test"
    
    def test_document_to_dict(self):
        doc = Document(
            content="Test",
            metadata={'key': 'value'},
            score=0.8
        )
        result = doc.to_dict()
        assert result['content'] == "Test"
        assert result['metadata'] == {'key': 'value'}
        assert result['score'] == 0.8


class TestKnowledgeRetriever:
    """Tests for KnowledgeRetriever"""
    
    def test_initialization(self):
        retriever = KnowledgeRetriever(
            chroma_path="/test/path",
            embedding_model="test-model"
        )
        assert retriever.chroma_path == "/test/path"
        assert retriever.embedding_model_name == "test-model"
    
    def test_mock_documents_products(self):
        retriever = KnowledgeRetriever()
        docs = retriever._get_mock_documents("tìm sách", k=3, category='products')
        assert len(docs) <= 3
        assert all(isinstance(d, Document) for d in docs)
    
    def test_mock_documents_policies(self):
        retriever = KnowledgeRetriever()
        docs = retriever._get_mock_documents("chính sách đổi trả", k=3, category='policies')
        assert len(docs) <= 3


class TestResponseGenerator:
    """Tests for ResponseGenerator"""
    
    def test_initialization_without_api_key(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            generator = ResponseGenerator()
            assert generator._use_mock == True
    
    def test_format_context(self):
        generator = ResponseGenerator()
        docs = [
            Document(content="Content 1", metadata={'category': 'products'}),
            Document(content="Content 2", metadata={'category': 'policies'})
        ]
        context = generator._format_context(docs)
        assert "Content 1" in context
        assert "Content 2" in context
    
    def test_mock_response_greeting(self):
        generator = ResponseGenerator()
        generator._use_mock = True
        response = generator._generate_mock_response(
            "Xin chào",
            [],
            {'name': 'Nguyễn Văn A'}
        )
        assert "chào" in response.lower() or "Nguyễn Văn A" in response
    
    def test_mock_response_product(self):
        generator = ResponseGenerator()
        generator._use_mock = True
        docs = [Document(content="Sách Đắc Nhân Tâm", metadata={})]
        response = generator._generate_mock_response("tìm sách", docs, None)
        assert len(response) > 0


class TestPromptSelection:
    """Tests for prompt template selection"""
    
    def test_greeting_prompt(self):
        from app.rag.prompts import GREETING_PROMPT
        template = select_prompt_template("Xin chào")
        assert template == GREETING_PROMPT
    
    def test_policy_prompt(self):
        from app.rag.prompts import POLICY_RESPONSE_PROMPT
        template = select_prompt_template("chính sách đổi trả")
        assert template == POLICY_RESPONSE_PROMPT
    
    def test_product_prompt(self):
        from app.rag.prompts import PRODUCT_CONSULTATION_PROMPT
        template = select_prompt_template("tìm sách về kinh tế")
        assert template == PRODUCT_CONSULTATION_PROMPT
    
    def test_order_prompt(self):
        from app.rag.prompts import ORDER_SUPPORT_PROMPT
        template = select_prompt_template("kiểm tra đơn hàng của tôi")
        assert template == ORDER_SUPPORT_PROMPT
    
    def test_customer_info_section(self):
        info = {
            'name': 'Test User',
            'segment': 'VIP',
            'total_orders': 10
        }
        section = get_customer_info_section(info)
        assert 'Test User' in section
        assert 'VIP' in section


class TestConsultingChain:
    """Tests for ConsultingChain"""
    
    def test_initialization(self):
        chain = ConsultingChain()
        assert chain.retriever is not None
        assert chain.generator is not None
    
    def test_chat_response_creation(self):
        response = ChatResponse(
            response="Test response",
            session_id="test-session",
            sources=[],
            customer_id="cust-1"
        )
        assert response.response == "Test response"
        assert response.session_id == "test-session"
    
    def test_chat_response_to_dict(self):
        response = ChatResponse(
            response="Test",
            session_id="123",
            sources=[{'content': 'source'}]
        )
        data = response.to_dict()
        assert data['response'] == "Test"
        assert data['session_id'] == "123"
        assert len(data['sources']) == 1
    
    @patch.object(KnowledgeRetriever, 'retrieve')
    def test_process_query(self, mock_retrieve):
        mock_retrieve.return_value = [
            Document(content="Test content", metadata={}, score=0.9)
        ]
        
        chain = ConsultingChain()
        chain.generator._use_mock = True
        
        response = chain.process("test query")
        
        assert response.response is not None
        assert response.session_id is not None
    
    def test_format_sources(self):
        chain = ConsultingChain()
        docs = [
            Document(content="A" * 300, metadata={'cat': 'test'}, score=0.95)
        ]
        sources = chain._format_sources(docs)
        
        assert len(sources) == 1
        assert sources[0]['score'] == 0.95
        assert len(sources[0]['content']) <= 203  # 200 + '...'
    
    def test_mock_behavior_analysis(self):
        chain = ConsultingChain()
        analysis = chain._get_mock_behavior_analysis("customer-123")
        
        assert 'segment' in analysis
        assert 'favorite_categories' in analysis
        assert 'recommendations' in analysis


class TestChatResponse:
    """Tests for ChatResponse dataclass"""
    
    def test_default_values(self):
        response = ChatResponse(
            response="Test",
            session_id="123"
        )
        assert response.sources == []
        assert response.customer_id is None
        assert response.personalized == False
        assert response.metadata == {}
    
    def test_timestamp_auto_generated(self):
        response = ChatResponse(
            response="Test",
            session_id="123"
        )
        assert response.timestamp is not None
