"""
Enhanced Response Generator with Gemini API support
"""
import os
import logging
from typing import List, Optional, Dict, Any
from django.conf import settings

from .retriever import Document
from .prompts import (
    select_prompt_template,
    get_customer_info_section,
    SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)


class ResponseGeneratorV2:
    """
    Response Generator with support for multiple LLM backends:
    1. Gemini API (Google)
    2. OpenAI API (OpenAI)
    3. Local models
    4. Mock responses
    """
    
    def __init__(
        self,
        llm_type: str = None,
        model_name: str = None,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize generator with LLM backend
        
        Args:
            llm_type: "gemini", "openai", "local", "mock" (auto-detect if None)
            model_name: Model name/ID
            api_key: API key (for gemini/openai)
            temperature: Generation temperature
            max_tokens: Max tokens in response
        """
        self.llm_type = llm_type or self._detect_llm_type()
        self.model_name = model_name or self._get_default_model(self.llm_type)
        self.api_key = api_key or self._get_api_key(self.llm_type)
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client = None
        self._use_mock = False
        
        # Initialize client
        self._initialize_client()
        
        logger.info(f"ResponseGeneratorV2 initialized with {self.llm_type} ({self.model_name})")
    
    def _detect_llm_type(self) -> str:
        """Auto-detect LLM type from environment"""
        # Check environment variables in order of preference
        if os.environ.get('GEMINI_API_KEY') or getattr(settings, 'GEMINI_API_KEY', None):
            return "gemini"
        elif os.environ.get('OPENAI_API_KEY') or getattr(settings, 'OPENAI_API_KEY', None):
            return "openai"
        else:
            return "mock"
    
    def _get_default_model(self, llm_type: str) -> str:
        """Get default model for LLM type"""
        models = {
            "gemini": "gemini-1.5-flash",
            "openai": "gpt-3.5-turbo",
            "local": "local-model",
            "mock": "mock"
        }
        return models.get(llm_type, "mock")
    
    def _get_api_key(self, llm_type: str) -> Optional[str]:
        """Get API key from environment or settings"""
        if llm_type == "gemini":
            return os.environ.get('GEMINI_API_KEY') or getattr(settings, 'GEMINI_API_KEY', None)
        elif llm_type == "openai":
            return os.environ.get('OPENAI_API_KEY') or getattr(settings, 'OPENAI_API_KEY', None)
        return None
    
    def _initialize_client(self):
        """Initialize LLM client"""
        try:
            if self.llm_type == "gemini":
                from .gemini_client import GeminiClient
                if not self.api_key:
                    raise ValueError("GEMINI_API_KEY not provided")
                self._client = GeminiClient(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                logger.info(f"Gemini client initialized")
                
            elif self.llm_type == "openai":
                from openai import OpenAI
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY not provided")
                self._client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI client initialized")
                
            elif self.llm_type == "local":
                logger.info("Using local model (mock)")
                self._use_mock = True
                
            else:  # mock
                logger.info("Using mock responses")
                self._use_mock = True
                
        except Exception as e:
            logger.warning(f"Failed to initialize {self.llm_type} client: {e}")
            self._use_mock = True
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string"""
        if not documents:
            return "No relevant information found in knowledge base."
        
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
        Generate response with selected LLM backend
        
        Args:
            query: User query
            context: Retrieved context documents
            customer_info: Optional customer information for personalization
            
        Returns:
            Generated response string
        """
        if self._use_mock:
            return self._generate_mock_response(query, context, customer_info)
        
        try:
            # Format context
            context_str = self._format_context(context)
            
            # Get context category
            context_category = None
            if context and hasattr(context[0], 'metadata'):
                context_category = context[0].metadata.get('category')
            
            # Select prompt template
            prompt_template = select_prompt_template(query, context_category)
            
            # Build customer info section
            customer_info_section = get_customer_info_section(customer_info) if customer_info else ""
            
            # Format final prompt
            prompt = prompt_template.format(
                query=query,
                context=context_str,
                customer_info_section=customer_info_section
            )
            
            # Generate based on LLM type
            if self.llm_type == "gemini":
                return self._generate_with_gemini(prompt)
            elif self.llm_type == "openai":
                return self._generate_with_openai(prompt)
            else:
                return self._generate_mock_response(query, context, customer_info)
                
        except Exception as e:
            logger.error(f"Generation error ({self.llm_type}): {e}")
            return self._generate_mock_response(query, context, customer_info)
    
    def _generate_with_gemini(self, prompt: str) -> str:
        """Generate using Gemini API"""
        try:
            response = self._client.chat_simple(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT
            )
            logger.info("Response generated by Gemini")
            return response
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate using OpenAI API"""
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            logger.info("Response generated by OpenAI")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def _generate_mock_response(self, query: str, context: List[Document], customer_info: Dict = None) -> str:
        """Generate mock response (fallback)"""
        mock_responses = [
            f"Thank you for your question about '{query}'. Based on our knowledge base, here's what I found:\n\nOur support team is here to help. For more specific assistance, please contact our customer support.",
            f"I understand you're asking about: {query}\n\nThis is important information that might help you. Please feel free to ask if you need more details or clarification.",
            "Thank you for reaching out! I'm here to help with your inquiry. Our team ensures the best service to all our customers."
        ]
        
        import random
        response = random.choice(mock_responses)
        
        # Add customer-specific greeting if available
        if customer_info and customer_info.get('segment'):
            segment = customer_info.get('segment', 'Valued')
            response = f"Hello {segment} customer!\n\n{response}"
        
        return response
    
    def get_info(self) -> Dict[str, Any]:
        """Get generator information"""
        return {
            'llm_type': self.llm_type,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'using_mock': self._use_mock,
            'client_initialized': self._client is not None
        }
