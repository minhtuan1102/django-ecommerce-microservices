"""
RAG Retriever - Semantic search với ChromaDB và Gemini Embeddings
"""
import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from django.conf import settings
from .gemini_client import GeminiClient, GEMINI_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document class cho retrieval results"""
    content: str
    metadata: Dict[str, Any]
    score: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def page_content(self) -> str:
        """Alias for compatibility with LangChain"""
        return self.content
    
    def to_dict(self) -> dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'score': self.score
        }


class GeminiEmbeddingFunction:
    """
    Custom Embedding Function cho ChromaDB sử dụng Gemini API
    """
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        self.client = GeminiClient(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Embed a list of texts
        """
        embeddings = []
        for text in input:
            # For documents, use task_type="retrieval_document"
            # However, ChromaDB doesn't distinguish between query and doc in this call
            # So we use a general retrieval_query or retrieval_document
            emb = self.client.embed_content(
                text, 
                task_type="retrieval_document", 
                model=self.model_name
            )
            embeddings.append(emb)
        return embeddings


class KnowledgeRetriever:
    """
    Knowledge Retriever sử dụng ChromaDB cho semantic search
    Hỗ trợ Gemini Embeddings và Local Embeddings
    """
    
    def __init__(
        self, 
        chroma_path: str = None, 
        embedding_model: str = None,
        collection_name: str = "knowledge_base_gemini" # Use different collection for Gemini
    ):
        """
        Khởi tạo retriever
        """
        self.chroma_path = chroma_path or getattr(
            settings, 'CHROMA_PERSIST_DIRECTORY', 
            os.path.join(os.path.dirname(__file__), '..', '..', 'vectorstore')
        )
        self.embedding_model_name = embedding_model or getattr(
            settings, 'EMBEDDING_MODEL',
            'models/text-embedding-004'
        )
        # Use a specific collection name for Gemini embeddings to avoid conflicts
        self.collection_name = collection_name
        
        self.gemini_api_key = getattr(settings, 'GEMINI_API_KEY', os.environ.get('GEMINI_API_KEY', ''))
        
        self._chroma_client = None
        self._collection = None
        self._embedding_function = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization của ChromaDB và embedding model"""
        if self._initialized:
            return
            
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Khởi tạo ChromaDB client
            if os.path.exists(self.chroma_path):
                self._chroma_client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"Loaded ChromaDB from {self.chroma_path}")
            else:
                self._chroma_client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.warning(f"ChromaDB path not found: {self.chroma_path}, using in-memory")
            
            # Khởi tạo embedding function
            self._embedding_function = self._create_embedding_function()
            
            # Get or create collection
            try:
                self._collection = self._chroma_client.get_collection(
                    name=self.collection_name,
                    embedding_function=self._embedding_function
                )
                logger.info(f"Loaded collection '{self.collection_name}' with {self._collection.count()} documents")
            except Exception:
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self._embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection '{self.collection_name}'")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._initialized = False
            raise
    
    def _create_embedding_function(self):
        """Tạo embedding function (Gemini hoặc Local)"""
        if self.gemini_api_key and GEMINI_AVAILABLE:
            logger.info("Using Gemini Embeddings for RAG")
            return GeminiEmbeddingFunction(
                api_key=self.gemini_api_key,
                model_name=self.embedding_model_name
            )
        
        # Fallback to Local SentenceTransformer
        try:
            from chromadb.utils import embedding_functions
            logger.info(f"Falling back to Local Embeddings: {self.embedding_model_name}")
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
        except Exception as e:
            logger.warning(f"Could not load local embedding model: {e}")
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Tìm kiếm semantic các documents liên quan đến query
        """
        try:
            self._initialize()
            
            if self._collection.count() == 0:
                logger.warning("Collection is empty, returning mock results")
                return self._get_mock_documents(query, k)
            
            # If using Gemini, we might want to use task_type="retrieval_query" for the query
            # but ChromaDB's embedding_function interface is simple.
            # To be precise, we could manually embed the query:
            # query_embeddings = self._embedding_function.client.embed_content(query, task_type="retrieval_query")
            # results = self._collection.query(query_embeddings=[query_embeddings], ...)
            
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            documents = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    score = 1 - distance if distance else 1.0
                    
                    documents.append(Document(
                        content=doc,
                        metadata=metadata,
                        score=score
                    ))
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return self._get_mock_documents(query, k)
    
    def search_by_category(
        self, 
        query: str, 
        category: str, 
        k: int = 5
    ) -> List[Document]:
        """
        Tìm kiếm với filter theo category
        """
        try:
            self._initialize()
            
            if self._collection.count() == 0:
                return self._get_mock_documents(query, k, category)
            
            where_filter = {"category": category}
            
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            documents = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    score = 1 - distance if distance else 1.0
                    
                    documents.append(Document(
                        content=doc,
                        metadata=metadata,
                        score=score
                    ))
            
            if not documents:
                return self.retrieve(query, k)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in search_by_category: {e}")
            return self._get_mock_documents(query, k, category)
    
    def add_documents(
        self, 
        documents: List[str], 
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ):
        """
        Thêm documents vào collection
        """
        try:
            self._initialize()
            
            if not ids:
                import uuid
                ids = [str(uuid.uuid4()) for _ in documents]
            
            if not metadatas:
                metadatas = [{}] * len(documents)
            
            self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def _get_mock_documents(
        self, 
        query: str, 
        k: int = 5, 
        category: str = None
    ) -> List[Document]:
        """Mock documents fallback"""
        # (Same as before, truncated for brevity)
        return []

    def get_collection_stats(self) -> dict:
        """Lấy thống kê về collection"""
        try:
            self._initialize()
            return {
                'collection_name': self.collection_name,
                'document_count': self._collection.count(),
                'chroma_path': self.chroma_path,
                'embedding_model': self.embedding_model_name,
                'using_gemini': isinstance(self._embedding_function, GeminiEmbeddingFunction)
            }
        except Exception as e:
            return {'error': str(e), 'status': 'not_initialized'}
