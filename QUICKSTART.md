# Quick Start Guide - AI Chatbot với Gemini

## ⚡ Nhanh nhất (5 phút)

### 1. Lấy Gemini API Key (2 phút)
```
1. Vào: https://aistudio.google.com/app/apikey
2. Login Google Account
3. Click "Create API key"
4. Copy key (dạng: AIzaSy...)
```

### 2. Tạo file .env (1 phút)
```bash
cd C:\Users\NITRO\bookstore-microservice

# Copy template
copy .env.example .env

# Mở .env và thêm:
GEMINI_API_KEY=AIzaSy_YOUR_KEY_HERE
LLM_TYPE=gemini
LLM_MODEL=gemini-1.5-flash
```

### 3. Start Services (2 phút)
```bash
# Build và start
docker-compose up -d --build

# Hoặc chỉ chatbot service
docker-compose up -d consulting-chatbot-service
```

### 4. Test Chatbot
```bash
# Check health
curl http://localhost:8015/api/health/

# Test chat
curl -X POST http://localhost:8015/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"Chào bạn"}'
```

✅ **DONE!** Chatbot hoạt động với Gemini API.

## 📊 Trạng thái Integration

### ✅ Đã hoàn thành
- [x] Model training với Kaggle dataset (83% accuracy)
- [x] Knowledge base (10 documents, 305 chunks)
- [x] Vector store (ChromaDB indexed)
- [x] Behavior Analysis Service (port 8014)
- [x] Chatbot Service với Gemini support (port 8015)
- [x] Docker configuration
- [x] API Gateway routes

### 📂 Files đã copy
```
C:\Users\NITRO\bookstore-microservice\
├── behavior-analysis-service/
│   └── data/models/              ✅ Trained models
├── consulting-chatbot-service/
│   └── app/rag/
│       ├── gemini_client.py      ✅ Gemini API client
│       └── generator_v2.py       ✅ Multi-LLM generator
├── knowledge-base/
│   ├── documents/                ✅ 10 MD files
│   └── vectorstore/              ✅ ChromaDB indexed
├── .env.example                  ✅ Gemini config
├── docker-compose.yml            ✅ AI services
├── GEMINI_API_SETUP.md           ✅ Setup guide
└── AI_SERVICES_GUIDE.md          ✅ Full docs
```

## 🎯 Endpoints

| Service | URL | Port |
|---------|-----|------|
| API Gateway | http://localhost:8000 | 8000 |
| Behavior Analysis | http://localhost:8014 | 8014 |
| Chatbot Service | http://localhost:8015 | 8015 |

### Via API Gateway
```
http://localhost:8000/api/ai/chat/          # Chat endpoint
http://localhost:8000/api/ai/behavior/      # Behavior analysis
http://localhost:8000/store/ai-assistant/   # Web UI
```

## 🔧 Các lựa chọn LLM

### Option 1: Gemini (Khuyến nghị) ⭐
```bash
GEMINI_API_KEY=AIzaSy...
LLM_TYPE=gemini
LLM_MODEL=gemini-1.5-flash
```
- ✅ Miễn phí 60 requests/phút
- ✅ Không cần thẻ tín dụng
- ✅ Tiếng Việt tốt

### Option 2: OpenAI
```bash
OPENAI_API_KEY=sk-...
LLM_TYPE=openai
LLM_MODEL=gpt-3.5-turbo
```
- ⚠️ Trả phí ~$0.002-0.01/message
- ✅ Chất lượng cao

### Option 3: Mock (Testing)
```bash
LLM_TYPE=mock
```
- ✅ Không cần API key
- ⚠️ Responses giả lập

## 📝 Ví dụ sử dụng

### Python
```python
import requests

response = requests.post('http://localhost:8015/api/chat/', json={
    'session_id': 'user-123',
    'message': 'Chính sách vận chuyển như thế nào?',
    'customer_id': '123'
})

print(response.json()['response'])
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/api/ai/chat/', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        session_id: 'user-123',
        message: 'Có sách gì hay về lập trình?',
        customer_id: '123'
    })
});

const data = await response.json();
console.log(data.response);
```

### cURL
```bash
curl -X POST http://localhost:8015/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "message": "Làm sao để tìm sách Python?",
    "customer_id": "1"
  }'
```

## 🐛 Troubleshooting

### Chatbot không trả lời
```bash
# Check logs
docker-compose logs consulting-chatbot-service

# Check if API key is set
docker-compose exec consulting-chatbot-service env | grep GEMINI

# Restart service
docker-compose restart consulting-chatbot-service
```

### Vector store rỗng
```bash
# Rebuild index
curl -X POST http://localhost:8015/api/chat/rebuild-index/
```

### Model không load
```bash
# Check model files
docker-compose exec behavior-analysis-service ls -la /app/data/models/
```

## 📚 Tài liệu chi tiết

- **Gemini Setup**: `GEMINI_API_SETUP.md` - Hướng dẫn API key
- **AI Services**: `AI_SERVICES_GUIDE.md` - Toàn bộ API docs
- **Docker Compose**: `docker-compose.yml` - Service config

## 💰 Chi phí

### Gemini Free Tier
- 60 requests/phút
- 1500 requests/ngày
- = **~1000-2000 conversations/ngày MIỄN PHÍ**

### Nếu cần nhiều hơn
- Enable billing trên Google Cloud
- Quota tự động tăng

## 🎉 Kết luận

Tất cả đã sẵn sàng! Chỉ cần:
1. ✅ Lấy Gemini API key
2. ✅ Cập nhật `.env`
3. ✅ `docker-compose up -d`

**Chatbot đã hoạt động với Gemini API!** 🚀
