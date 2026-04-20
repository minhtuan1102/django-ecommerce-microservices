# Consulting Chatbot Service

Dịch vụ chatbot tư vấn sử dụng RAG (Retrieval Augmented Generation) cho hệ thống thương mại điện tử.

## Tính năng

- **RAG Retriever**: Semantic search với ChromaDB
- **RAG Generator**: Response generation với OpenAI GPT (có fallback mock)
- **Personalization**: Tích hợp behavior analysis service
- **Multi-turn Conversations**: Lưu trữ lịch sử chat
- **Vietnamese Support**: Prompt templates tiếng Việt

## Cài đặt

### Yêu cầu
- Python 3.10+
- PostgreSQL
- ChromaDB vectorstore (từ knowledge-base service)

### Local Development

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Tạo migrations
python manage.py makemigrations
python manage.py migrate

# Chạy server
python manage.py runserver
```

### Docker

```bash
# Build và chạy với docker-compose
docker-compose up --build
```

## API Endpoints

### Chat

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| POST | `/api/chat/message/` | Gửi tin nhắn, nhận phản hồi |
| GET | `/api/chat/history/{session_id}/` | Lấy lịch sử chat |
| POST | `/api/chat/feedback/` | Gửi feedback |
| GET | `/api/chat/health/` | Health check |

### Session Management

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/chat/sessions/{customer_id}/` | Danh sách sessions của khách hàng |
| DELETE | `/api/chat/session/{session_id}/` | Xóa session |
| POST | `/api/chat/session/{session_id}/clear/` | Xóa messages trong session |

### Search

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/chat/search/?q={query}` | Tìm kiếm trong knowledge base |

## Request/Response Examples

### Gửi tin nhắn

**Request:**
```json
{
    "message": "Tôi muốn tìm sách về lập trình",
    "customer_id": "customer-123",
    "session_id": "optional-session-id",
    "personalized": true
}
```

**Response:**
```json
{
    "response": "Xin chào! Dựa trên sở thích của bạn, tôi xin giới thiệu...",
    "session_id": "uuid-session-id",
    "sources": [
        {
            "content": "Sách Clean Code...",
            "metadata": {"category": "products"},
            "score": 0.95
        }
    ],
    "personalized": true,
    "timestamp": "2024-01-15T10:30:00"
}
```

### Gửi feedback

**Request:**
```json
{
    "message_id": "uuid-message-id",
    "rating": 5,
    "comment": "Câu trả lời rất hữu ích!",
    "is_helpful": true
}
```

## Environment Variables

```env
# Django
SECRET_KEY=your-secret-key
DEBUG=True

# Database
DB_ENGINE=django.db.backends.postgresql
DB_NAME=chatbot_db
DB_USER=bookstore_user
DB_PASSWORD=bookstore_pass
DB_HOST=localhost
DB_PORT=5432

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Microservices
BEHAVIOR_SERVICE_URL=http://behavior-analysis-service:8000

# ChromaDB
CHROMA_PERSIST_DIRECTORY=/app/vectorstore
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=gpt-3.5-turbo
```

## Testing

```bash
# Chạy tests
pytest

# Với coverage
pytest --cov=app --cov-report=html
```

## Cấu trúc thư mục

```
consulting-chatbot-service/
├── app/
│   ├── api/
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   └── views.py
│   ├── rag/
│   │   ├── chain.py
│   │   ├── generator.py
│   │   ├── prompts.py
│   │   └── retriever.py
│   ├── utils/
│   │   └── behavior_client.py
│   ├── admin.py
│   ├── apps.py
│   └── models.py
├── chatbot_service/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── tests/
│   ├── test_api.py
│   ├── test_behavior_client.py
│   ├── test_models.py
│   └── test_rag.py
├── Dockerfile
├── docker-compose.yml
├── manage.py
├── pytest.ini
├── README.md
└── requirements.txt
```

## Fallback Mode

Khi không có OpenAI API key, service tự động sử dụng mock responses:
- Chào hỏi thân thiện
- Tư vấn sản phẩm cơ bản
- Trả lời về chính sách
- Hỗ trợ đơn hàng

## License

MIT
