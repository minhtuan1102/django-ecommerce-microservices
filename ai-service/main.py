from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import os
import re

# Hỗ trợ import linh hoạt cho các phiên bản LangChain khác nhau
try:
    from langchain_community.graphs import Neo4jGraph
except ImportError:
    try:
        from langchain.graphs import Neo4jGraph
    except ImportError:
        Neo4jGraph = None

try:
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
except ImportError:
    try:
        from langchain.chains import GraphCypherQAChain
    except ImportError:
        GraphCypherQAChain = None

app = FastAPI(title="AI Service", description="AI Service for E-Commerce Recommendation and Chatbot")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Neo4j setup reading from environment variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

def build_llm():
    llm_type = os.getenv("LLM_TYPE", "gemini").lower()
    
    if llm_type == "ollama":
        from langchain_ollama import ChatOllama
        model_name = os.getenv("LLM_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0,
            timeout=60, # Set timeout to 60 seconds for local Llama3
        )
        return llm, f"Ollama ({model_name})"

    if llm_type == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment")
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=gemini_api_key,
        )
        return llm, f"Gemini ({model_name})"

    if llm_type == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=openai_api_key,
        )
        return llm, f"OpenAI ({model_name})"

    raise RuntimeError(f"Unsupported LLM_TYPE: {llm_type}. Set LLM_TYPE to 'ollama', 'gemini', or 'openai'.")

# Caching graph and chain instances
_graph = None
_chain = None

def get_graph_qa_chain():
    global _graph, _chain
    if _chain is not None:
        return _chain

    if Neo4jGraph is None or GraphCypherQAChain is None:
        raise RuntimeError("Không tìm thấy thư viện LangChain Neo4j. Vui lòng cài đặt: langchain-community, langchain-google-genai (cho Gemini), hoặc langchain-ollama (cho Ollama).")

    print(f"Connecting to Neo4j at {NEO4J_URI}")
    _graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    llm, llm_name = build_llm()
    print(f"Using LLM: {llm_name}")
    
    try:
        _chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=_graph,
            verbose=True,
            allow_dangerous_requests=True,
        )
    except TypeError:
        _chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=_graph,
            verbose=True,
        )
    return _chain

# --- Schemas ---
class ChatbotRequest(BaseModel):
    user_id: int | str = None
    message: str

class BehaviorTrackRequest(BaseModel):
    user_id: int
    product_id: int
    action: str

# --- Endpoints ---

@app.options("/recommend")
def recommend_products_options():
    return {"message": "OK"}

@app.get("/recommend")
def recommend_products(user_id: int):
    """
    Recommendation List API
    Returns a list of recommended product IDs for the given user.
    """
    # Mocking hybrid recommendation logic (LSTM + Graph + RAG)
    # In a real scenario, this would load model_best.keras or query Neo4j.
    recommendations = [101, 102, 205, random.randint(200, 300), random.randint(300, 400)]
    return recommendations[:3] # Returning 3 recommendations

@app.options("/chatbot")
def chatbot_options():
    return {"message": "OK"}

@app.post("/chatbot")
def chatbot(request: ChatbotRequest):
    """
    Chatbot API
    Returns an AI-generated response based on user input, querying Neo4j Knowledge Graph.
    """
    try:
        chain = get_graph_qa_chain()
        message = request.message
        
        # Format a prompt for the user query
        prompt = f"Bạn là trợ lý AI của nhà sách trực tuyến. Người dùng hỏi: {message}"
        
        try:
            result = chain.invoke({"query": prompt})
        except Exception:
            result = chain.invoke(prompt)

        if isinstance(result, dict):
            response_text = result.get("result") or result.get("answer") or str(result)
        else:
            response_text = str(result)

    except Exception as e:
        print(f"Error calling RAG chain: {e}")
        response_text = f"Xin lỗi, hiện tại tính năng tư vấn qua Graph RAG gặp lỗi kết nối: {str(e)}"
        
    return {"response": response_text}

@app.post("/track")
def track_behavior(request: BehaviorTrackRequest):
    """
    Behavior Analysis API
    Saves user actions (view, click, add_to_cart) for later analysis.
    """
    # In a real scenario, save to database or send to a message queue.
    return {"status": "success", "message": f"Action '{request.action}' tracked for user {request.user_id} on product {request.product_id}."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
