"""LangChain + Chroma RAG sample for ecommerce consultation.

Run:
  python examples/langchain_rag_app.py --question "Toi muon mua ao thun duoi 300k"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


def load_docs(documents_root: Path) -> List[Document]:
    docs: List[Document] = []
    for file_path in documents_root.rglob("*.md"):
        content = file_path.read_text(encoding="utf-8")
        category = file_path.parent.name
        docs.append(
            Document(
                page_content=content,
                metadata={"source": file_path.as_posix(), "category": category},
            )
        )
    return docs


def get_embeddings():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def get_llm():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.3)


def build_or_load_vectorstore(documents_root: Path, persist_dir: Path) -> Chroma:
    docs = load_docs(documents_root)
    embeddings = get_embeddings()

    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name="knowledge_base",
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    if store._collection.count() == 0 and docs:
        store.add_documents(docs)
        if hasattr(store, "persist"):
            store.persist()

    return store


def answer_question(question: str, store: Chroma) -> str:
    results = store.similarity_search_with_relevance_scores(question, k=4)
    context_chunks = []
    for doc, score in results:
        context_chunks.append(
            f"[score={score:.3f}] ({doc.metadata.get('category')}) {doc.page_content[:500]}"
        )

    context = "\n\n".join(context_chunks) if context_chunks else "No context available"

    llm = get_llm()
    if llm is None:
        return (
            "OPENAI_API_KEY is not set.\n\n"
            "Retrieved context:\n"
            f"{context}\n\n"
            "Set OPENAI_API_KEY to enable generated answers."
        )

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an ecommerce consultant assistant. "
                    "Answer in Vietnamese, grounded in the provided context."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n\n"
                    f"Context:\n{context}\n\n"
                    "If context is insufficient, say what is missing."
                )
            ),
        ]
    )
    return response.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--documents-root", default="knowledge-base/documents")
    parser.add_argument("--persist-dir", default="knowledge-base/vectorstore")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = build_or_load_vectorstore(Path(args.documents_root), Path(args.persist_dir))
    print(answer_question(args.question, store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
