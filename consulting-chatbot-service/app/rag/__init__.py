"""
RAG (Retrieval Augmented Generation) module for Consulting Chatbot
"""
from .retriever import KnowledgeRetriever, Document
from .generator import ResponseGenerator
from .chain import ConsultingChain, ChatResponse
from .prompts import (
    select_prompt_template,
    get_customer_info_section,
    PRODUCT_CONSULTATION_PROMPT,
    POLICY_RESPONSE_PROMPT,
    ORDER_SUPPORT_PROMPT,
    FALLBACK_PROMPT,
    PERSONALIZED_PROMPT,
    DEFAULT_PROMPT,
)

__all__ = [
    'KnowledgeRetriever',
    'Document',
    'ResponseGenerator',
    'ConsultingChain',
    'ChatResponse',
    'select_prompt_template',
    'get_customer_info_section',
    'PRODUCT_CONSULTATION_PROMPT',
    'POLICY_RESPONSE_PROMPT',
    'ORDER_SUPPORT_PROMPT',
    'FALLBACK_PROMPT',
    'PERSONALIZED_PROMPT',
    'DEFAULT_PROMPT',
]
