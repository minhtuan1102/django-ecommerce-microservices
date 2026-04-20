# AI Tu Van E-commerce (RAG + Knowledge Base)

Tai lieu nay tra loi day du 3 yeu cau:
- (4) Thiet ke Knowledge Base cho e-commerce nhieu mat hang
- (5) Code RAG application bang Python + LangChain + ChromaDB + LLM
- (6) Deployment thanh API service va tich hop vao backend hien tai

## 1. Tong quan implementation da hoan thanh

Da bo sung cac thanh phan trong workspace:

- Multi-item catalog endpoint:
  - `GET /api/ecommerce/products/` (API Gateway)
  - Gom du lieu tu `book-service`, `clothe-service`, va mo rong `laptop/mobile` qua env URL.
- AI endpoints qua gateway:
  - `POST /api/ai/chat/`
  - `GET /api/ai/behavior/`
  - `GET /api/ai/chat/health/`
  - `GET /api/ai/chat/index-status/` (admin)
  - `POST /api/ai/chat/rebuild-index/` (admin)
- KB data pipeline:
  - `knowledge-base/scripts/crawl_products.py`
  - `knowledge-base/scripts/build_kb_documents.py`
- LangChain indexing pipeline:
  - `consulting-chatbot-service/app/rag/indexer.py`
  - `python manage.py build_kb_index --force`

## 2. Mo rong bookstore thanh ecommerce nhieu mat hang

### 2.1 Unified catalog API

Gateway da co endpoint:

```http
GET /api/ecommerce/products/?q=<keyword>&type=<all|book|clothe|laptop|mobile>&stock=<all|in_stock|out_of_stock>
```

Response schema:

```json
{
  "total": 123,
  "item_types": ["book", "clothe", "mobile"],
  "products": [
    {
      "id": 1,
      "sku": "book-1",
      "title": "Clean Code",
      "item_type": "book",
      "category": "programming",
      "price": 189000,
      "stock": 24,
      "in_stock": true,
      "source_service": "book-service"
    }
  ]
}
```

### 2.2 Nguon du lieu mo rong

- Mac dinh: `book-service`, `clothe-service`
- Mo rong optional:
  - `LAPTOP_SERVICE_URL=http://laptop-service:8000`
  - `MOBILE_SERVICE_URL=http://mobile-service:8000`

## 4. Thiet ke Knowledge Base (KB)

### 4.1 Cau truc KB

```text
knowledge-base/
  documents/
    products/
    policies/
    faqs/
  scripts/
    crawl_products.py
    build_kb_documents.py
  vectorstore/
```

### 4.2 Chien luoc crawl du lieu san pham

Crawler su dung API noi bo cua backend (khong scrape HTML ngau nhien), dam bao du lieu chuan va on dinh:

```bash
cd knowledge-base/scripts
python crawl_products.py --gateway-url http://localhost:8000/api/ecommerce/products/
```

Crawler output:
- `documents/products/products_raw.jsonl` (du lieu tho)
- `documents/products/products_catalog.md` (doc markdown de RAG dung)

### 4.3 Chuan hoa va format du lieu KB

```bash
cd knowledge-base/scripts
python build_kb_documents.py
```

Script se sinh:
- `documents/products/products_<item_type>.md`
- `documents/products/category_<category>.md`
- `documents/faqs/ecommerce_faq.md`
- `documents/policies/ecommerce_policies.md`

Metadata khuyen nghi tren tung chunk:
- `category`: products | faqs | policies
- `item_type`: book | clothe | laptop | mobile | general
- `source`: path file
- `source_service`: ten microservice nguon

## 5. Code RAG Application (LangChain + ChromaDB + LLM)

### 5.1 Ma nguon da trien khai

- Indexer: `consulting-chatbot-service/app/rag/indexer.py`
  - Load markdown docs
  - Split chunk (RecursiveCharacterTextSplitter)
  - Embedding:
    - Co API key: OpenAIEmbeddings
    - Khong co key: HuggingFaceEmbeddings
  - Index vao Chroma
- Generator: `consulting-chatbot-service/app/rag/generator.py`
  - Dung LangChain `ChatOpenAI`
  - Fallback mock neu thieu API key

### 5.2 Lenh build/rebuild index

```bash
cd consulting-chatbot-service
python manage.py build_kb_index --force
```

Hoac goi API:

```bash
curl -X POST http://localhost:8015/api/chat/index/rebuild/ \
  -H "Content-Type: application/json" \
  -d '{"force": true, "include_product_sync": true}'
```

### 5.3 Ma mau Python (yeu cau de bai)

Da cung cap ma mau doc lap:
- `consulting-chatbot-service/examples/langchain_rag_app.py`
- `consulting-chatbot-service/examples/fastapi_rag_service.py`

Mau nay minh hoa:
- Load docs
- Tao/su dung Chroma vectorstore
- Retrieval top-k
- Generate response voi LangChain LLM

## 6. Deployment AI module thanh API Service + tich hop backend

### 6.1 Dong goi AI module (service hien tai)

AI module da dong goi san trong `consulting-chatbot-service` va duoc chay qua Docker Compose.

Cau hinh quan trong (`docker-compose.yml`):
- Mount KB docs:
  - `./knowledge-base/documents:/app/knowledge-base/documents`
- Mount vectorstore:
  - `./knowledge-base/vectorstore:/app/vectorstore`
- Startup command:
  - migrate + build index + runserver

### 6.2 Bien moi truong can thiet

```env
OPENAI_API_KEY=sk-...
CHROMA_PERSIST_DIRECTORY=/app/vectorstore
KB_DOCUMENTS_ROOT=/app/knowledge-base/documents
PRODUCT_AGGREGATOR_URL=http://api-gateway:8000/api/ecommerce/products/
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=gpt-3.5-turbo
```

### 6.3 Trien khai

```bash
cd c:\Users\NITRO\bookstore-microservice

# (optional) build KB from live data
python knowledge-base/scripts/crawl_products.py --gateway-url http://localhost:8000/api/ecommerce/products/
python knowledge-base/scripts/build_kb_documents.py

# build and run services
docker-compose build
docker-compose up -d
```

### 6.4 Tich hop call API vao backend hien tai

Da tich hop qua API Gateway:

- Chat:

```http
POST /api/ai/chat/
Content-Type: application/json

{
  "message": "Toi can goi y san pham cho hoc lap trinh",
  "category": "products",
  "personalized": true
}
```

- Lay phan tich hanh vi:

```http
GET /api/ai/behavior/
```

- Quan tri index:

```http
GET /api/ai/chat/index-status/
POST /api/ai/chat/rebuild-index/
```

### 6.5 FastAPI/Flask deployment option

Neu can tach rieng thanh AI gateway doc lap (ngoai Django), dung mau:
- `consulting-chatbot-service/examples/fastapi_rag_service.py`

Run:

```bash
uvicorn examples.fastapi_rag_service:app --host 0.0.0.0 --port 8100
```

Sau do backend hien tai goi:

```http
POST http://<ai-host>:8100/chat
```

## 7. Checklist nghiem thu

- [ ] `GET /api/ecommerce/products/` tra ve du lieu nhieu item type
- [ ] `crawl_products.py` tao `products_raw.jsonl`
- [ ] `build_kb_documents.py` tao file markdown products/faqs/policies
- [ ] `python manage.py build_kb_index --force` thanh cong
- [ ] `GET /api/chat/index/status/` co document_count > 0
- [ ] `POST /api/ai/chat/` tra ve response + sources

## 8. Ghi chu quan trong

- Neu `OPENAI_API_KEY` chua set, service van chay fallback mode (mock response).
- Index rebuild co the chay thu cong qua API admin de cap nhat ngay khi catalog thay doi.
- Pipeline duoc thiet ke de khong crash neu backend tam thoi chua bat.
