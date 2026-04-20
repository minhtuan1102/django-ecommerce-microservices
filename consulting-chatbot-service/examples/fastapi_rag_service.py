"""FastAPI deployment sample for RAG ecommerce assistant.

Install:
  pip install fastapi uvicorn langchain langchain-community langchain-openai chromadb sentence-transformers

Run:
  uvicorn examples.fastapi_rag_service:app --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="Ecommerce RAG Service", version="1.0.0")

DOC_ROOT = Path(os.environ.get("KB_DOCUMENTS_ROOT", "knowledge-base/documents"))
VEC_DIR = Path(os.environ.get("CHROMA_PERSIST_DIRECTORY", "knowledge-base/vectorstore"))


class ChatRequest(BaseModel):
    message: str
    top_k: int = 4


class ChatResponse(BaseModel):
    response: str
    sources: List[str]


def _load_docs() -> List[Document]:
    docs = []
    for file_path in DOC_ROOT.rglob("*.md"):
        docs.append(
            Document(
                page_content=file_path.read_text(encoding="utf-8"),
                metadata={"source": file_path.as_posix(), "category": file_path.parent.name},
            )
        )
    return docs


def _embeddings():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def _llm():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.3)


def _store() -> Chroma:
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name="knowledge_base",
        persist_directory=str(VEC_DIR),
        embedding_function=_embeddings(),
    )

    if store._collection.count() == 0:
        docs = _load_docs()
        if docs:
            store.add_documents(docs)
            if hasattr(store, "persist"):
                store.persist()

    return store


@app.get("/health")
def health():
    return {"status": "ok", "service": "fastapi-rag"}


@app.post("/index/rebuild")
def rebuild_index():
    docs = _load_docs()
    store = Chroma(
        collection_name="knowledge_base",
        persist_directory=str(VEC_DIR),
        embedding_function=_embeddings(),
    )
    if store._collection.count() > 0:
        ids = store._collection.get().get("ids", [])
        if ids:
            store.delete(ids=ids)
    if docs:
        store.add_documents(docs)
    if hasattr(store, "persist"):
        store.persist()

    return {"status": "ok", "documents": len(docs)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    store = _store()
    hits = store.similarity_search_with_relevance_scores(req.message, k=req.top_k)

    context_chunks = []
    sources = []
    for doc, score in hits:
        sources.append(doc.metadata.get("source", "unknown"))
        context_chunks.append(f"[score={score:.3f}] {doc.page_content[:500]}")

    context = "\n\n".join(context_chunks) if context_chunks else "No context"
    llm = _llm()
    if llm is None:
        return ChatResponse(
            response="OPENAI_API_KEY not set. Context retrieved successfully.",
            sources=sources,
        )

    out = llm.invoke(
        [
            SystemMessage(content="You are an ecommerce consultant assistant. Reply in Vietnamese."),
            HumanMessage(content=f"Question: {req.message}\n\nContext:\n{context}"),
        ]
    )
    return ChatResponse(response=out.content, sources=sources)
