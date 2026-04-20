"""
Support for Google Gemini API in addition to OpenAI
Install: pip install google-generativeai
"""
import os
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Install with: pip install google-generativeai")


class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google API Key
            model_name: Model name (gemini-1.5-flash, gemini-1.5-pro, etc.)
            temperature: Temperature (0-1)
            max_tokens: Max output tokens
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name=model_name)
        
        logger.info(f"Gemini client initialized with model: {model_name}")
    
    def embed_content(
        self,
        text: str,
        task_type: str = "retrieval_query",
        model: str = "models/text-embedding-004"
    ) -> List[float]:
        """
        Generate embedding for a piece of text
        
        Args:
            text: Text to embed
            task_type: Type of task (retrieval_query, retrieval_document, etc.)
            model: Embedding model name
            
        Returns:
            List of floats (embedding vector)
        """
        try:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Gemini Embedding error: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None
    ) -> str:
        """
        Send message to Gemini
        
        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."} dicts
            system_prompt: Optional system prompt
            
        Returns:
            Model response text
        """
        try:
            # Build conversation
            conversation = []
            
            # Add system prompt as first user message if provided
            if system_prompt:
                conversation.append({"role": "user", "parts": system_prompt})
                conversation.append({"role": "model", "parts": "Understood. I'll follow those instructions."})
            
            # Add user messages
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Convert roles
                if role == "system":
                    continue  # Already added as part of system prompt
                elif role == "user":
                    conversation.append({"role": "user", "parts": content})
                elif role == "assistant":
                    conversation.append({"role": "model", "parts": content})
            
            # Generate response
            response = self.client.generate_content(
                contents=conversation,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def chat_simple(self, prompt: str, system_prompt: str = None) -> str:
        """
        Simple chat interface
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            response = self.client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise


def create_llm_client(llm_type: str, api_key: str, model_name: str, **kwargs):
    """
    Factory function to create LLM client
    
    Args:
        llm_type: "openai", "gemini", "ollama", "anthropic", etc.
        api_key: API key for the service
        model_name: Model name/ID
        **kwargs: Additional arguments
        
    Returns:
        LLM client object
    """
    if llm_type.lower() == "gemini":
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
        return GeminiClient(
            api_key=api_key,
            model_name=model_name,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000)
        )
    elif llm_type.lower() == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
