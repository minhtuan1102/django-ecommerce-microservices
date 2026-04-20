"""
Script để build Vector Store từ knowledge base documents
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_service.settings')

# Configure Django settings if not already configured
try:
    import django
    from django.conf import settings
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            DATABASES={},
            INSTALLED_APPS=['app'],
            CHROMA_PERSIST_DIRECTORY=str(PROJECT_ROOT / 'vectorstore'),
            EMBEDDING_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        )
    django.setup()
except Exception as e:
    print(f"Django setup: {e}")

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Paths
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent.parent.parent / 'knowledge-base' / 'documents'
VECTORSTORE_PATH = Path(__file__).resolve().parent.parent.parent / 'knowledge-base' / 'vectorstore'


def read_markdown_files(directory: Path) -> list:
    """Read all markdown files from a directory"""
    documents = []
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return documents
    
    for md_file in directory.glob("**/*.md"):
        try:
            content = md_file.read_text(encoding='utf-8')
            # Get relative path for metadata
            rel_path = md_file.relative_to(KNOWLEDGE_BASE_PATH)
            category = rel_path.parts[0] if rel_path.parts else 'general'
            
            documents.append({
                'content': content,
                'metadata': {
                    'source': str(rel_path),
                    'category': category,
                    'filename': md_file.name,
                    'title': md_file.stem.replace('_', ' ').title()
                },
                'id': f"{category}_{md_file.stem}"
            })
            print(f"  Read: {rel_path}")
        except Exception as e:
            print(f"  Error reading {md_file}: {e}")
    
    return documents


def chunk_document(doc: dict, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split a document into chunks"""
    content = doc['content']
    chunks = []
    
    # Split by headers first
    sections = []
    current_section = ""
    
    for line in content.split('\n'):
        if line.startswith('#'):
            if current_section.strip():
                sections.append(current_section)
            current_section = line + "\n"
        else:
            current_section += line + "\n"
    
    if current_section.strip():
        sections.append(current_section)
    
    # Now chunk each section if too large
    chunk_id = 0
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append({
                'content': section.strip(),
                'metadata': {**doc['metadata'], 'chunk_id': chunk_id},
                'id': f"{doc['id']}_chunk_{chunk_id}"
            })
            chunk_id += 1
        else:
            # Further split large sections
            words = section.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'content': chunk_text.strip(),
                        'metadata': {**doc['metadata'], 'chunk_id': chunk_id},
                        'id': f"{doc['id']}_chunk_{chunk_id}"
                    })
                    chunk_id += 1
                    # Start new chunk with overlap
                    overlap_words = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else []
                    current_chunk = overlap_words + [word]
                    current_length = sum(len(w) + 1 for w in current_chunk)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text.strip(),
                    'metadata': {**doc['metadata'], 'chunk_id': chunk_id},
                    'id': f"{doc['id']}_chunk_{chunk_id}"
                })
    
    return chunks


def build_vectorstore():
    """Build ChromaDB vectorstore from knowledge base documents"""
    print("=" * 60)
    print("BUILDING VECTORSTORE FROM KNOWLEDGE BASE")
    print("=" * 60)
    
    # Create vectorstore directory
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\nVectorstore path: {VECTORSTORE_PATH}")
    
    # Initialize ChromaDB
    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Initialize embedding function
    print("Loading embedding model (this may take a while on first run)...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    
    # Delete existing collection if exists
    try:
        client.delete_collection("knowledge_base")
        print("Deleted existing collection")
    except:
        pass
    
    # Create collection
    collection = client.get_or_create_collection(
        name="knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created collection: knowledge_base")
    
    # Read all documents
    print("\nReading documents...")
    all_documents = []
    
    for subdir in ['faqs', 'policies', 'products']:
        subdir_path = KNOWLEDGE_BASE_PATH / subdir
        docs = read_markdown_files(subdir_path)
        all_documents.extend(docs)
    
    print(f"\nTotal documents read: {len(all_documents)}")
    
    # Chunk documents
    print("\nChunking documents...")
    all_chunks = []
    for doc in all_documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Add to collection
    if all_chunks:
        print("\nAdding to vectorstore...")
        
        # Batch add for efficiency
        batch_size = 50
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            collection.add(
                documents=[c['content'] for c in batch],
                metadatas=[c['metadata'] for c in batch],
                ids=[c['id'] for c in batch]
            )
            print(f"  Added batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
    
    # Print stats
    print("\n" + "=" * 60)
    print("VECTORSTORE BUILD COMPLETE")
    print("=" * 60)
    print(f"Collection: knowledge_base")
    print(f"Total documents: {collection.count()}")
    print(f"Storage path: {VECTORSTORE_PATH}")
    
    # Test retrieval
    print("\n" + "=" * 60)
    print("TESTING RETRIEVAL")
    print("=" * 60)
    
    test_queries = [
        "How do I return a book?",
        "What are the shipping options?",
        "Do you have self-help books?",
        "Membership benefits"
    ]
    
    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=2,
            include=['documents', 'metadatas', 'distances']
        )
        print(f"\nQuery: '{query}'")
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0][:2]):
                distance = results['distances'][0][i] if results['distances'] else 0
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                print(f"  [{i+1}] Score: {1-distance:.3f}, Source: {metadata.get('source', 'N/A')}")
                print(f"      {doc[:100]}...")
    
    return collection.count()


if __name__ == "__main__":
    count = build_vectorstore()
    print(f"\n✓ Vectorstore built with {count} documents")
