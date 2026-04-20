# Hướng Dẫn Sử Dụng Google Gemini API

## Tại sao chọn Gemini?

✅ **Miễn phí**: Gemini có free tier rộng rãi (60 requests/phút)
✅ **Hiệu suất tốt**: Gemini 1.5 Flash nhanh và chính xác
✅ **Tiếng Việt**: Hỗ trợ tiếng Việt tốt
✅ **Không cần thẻ tín dụng**: Đăng ký dễ dàng

## Bước 1: Lấy Gemini API Key

### 1.1 Truy cập Google AI Studio
Vào: https://aistudio.google.com/app/apikey

### 1.2 Đăng nhập
Đăng nhập bằng Google Account của bạn

### 1.3 Tạo API Key
1. Click "Get API key" hoặc "Create API key"
2. Chọn project hoặc tạo mới
3. Copy API key (dạng: `AIzaSy...`)

### 1.4 Lưu API Key
```bash
# API key sẽ có dạng:
AIzaSyDXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

⚠️ **LƯU Ý**: Không share API key công khai!

## Bước 2: Cấu hình trong Project

### 2.1 Tạo file .env
```bash
cd C:\Users\NITRO\bookstore-microservice
copy .env.example .env
```

### 2.2 Chỉnh sửa .env
Mở file `.env` và thêm:

```bash
# Gemini API Configuration
GEMINI_API_KEY=AIzaSy_YOUR_ACTUAL_KEY_HERE
LLM_TYPE=gemini
LLM_MODEL=gemini-1.5-flash

# Optional: Temperature & Max Tokens
TEMPERATURE=0.7
MAX_TOKENS=1000
```

### 2.3 Lưu file

## Bước 3: Cài đặt Dependencies

```bash
# Install google-generativeai
pip install google-generativeai
```

Hoặc nếu dùng Docker (đã có trong requirements.txt):
```bash
docker-compose build consulting-chatbot-service
```

## Bước 4: Khởi động Services

```bash
# Start all services
docker-compose up -d

# Or start only chatbot
docker-compose up -d consulting-chatbot-service
```

## Bước 5: Test Chatbot

### 5.1 Kiểm tra Health
```bash
curl http://localhost:8015/api/health/
```

Expected response:
```json
{
  "status": "healthy",
  "llm_type": "gemini",
  "model": "gemini-1.5-flash"
}
```

### 5.2 Test Chat
```bash
curl -X POST http://localhost:8015/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-123",
    "message": "Chính sách đổi trả như thế nào?",
    "customer_id": "1"
  }'
```

## Các Model Gemini Có Sẵn

| Model | Mô tả | Tốc độ | Giá (Free Tier) |
|-------|-------|--------|-----------------|
| `gemini-1.5-flash` | Nhanh, hiệu quả | ⚡⚡⚡ | ✅ 15 RPM |
| `gemini-1.5-pro` | Chất lượng cao | ⚡⚡ | ✅ 2 RPM |
| `gemini-1.0-pro` | Cũ hơn nhưng ổn định | ⚡⚡ | ✅ 60 RPM |

**Khuyến nghị**: Dùng `gemini-1.5-flash` cho production.

## So Sánh Gemini vs OpenAI

| Feature | Gemini (Free) | OpenAI (Paid) |
|---------|---------------|---------------|
| Giá | ✅ Miễn phí (có limit) | ❌ $0.002-0.01/msg |
| Tiếng Việt | ✅ Tốt | ✅ Tốt |
| Tốc độ | ⚡⚡⚡ | ⚡⚡ |
| Limit | 60 requests/phút | Tùy plan |
| Đăng ký | ✅ Dễ | ⚠️ Cần thẻ |

## Troubleshooting

### ❌ Lỗi: "API key not valid"
```
Giải pháp:
1. Kiểm tra API key trong .env
2. Đảm bảo không có khoảng trắng thừa
3. Thử tạo API key mới
```

### ❌ Lỗi: "google-generativeai not installed"
```bash
# Fix:
pip install google-generativeai

# Hoặc với Docker:
docker-compose build consulting-chatbot-service
```

### ❌ Lỗi: "Rate limit exceeded"
```
Giải pháp:
1. Chờ 1 phút và thử lại
2. Nâng cấp lên paid tier
3. Giảm số requests
```

### ❌ Chatbot trả lời bằng tiếng Anh
```
Giải pháp:
1. Thêm "Trả lời bằng tiếng Việt" vào query
2. Kiểm tra system prompt
3. Tăng TEMPERATURE lên 0.8-0.9
```

## Chuyển từ OpenAI sang Gemini

Nếu đã dùng OpenAI, chỉ cần:

```bash
# 1. Comment OpenAI config
# OPENAI_API_KEY=sk-...
# LLM_TYPE=openai

# 2. Thêm Gemini config
GEMINI_API_KEY=AIzaSy...
LLM_TYPE=gemini
LLM_MODEL=gemini-1.5-flash
```

Restart service:
```bash
docker-compose restart consulting-chatbot-service
```

## Giới Hạn Free Tier

- **Requests**: 60 requests/phút (gemini-1.0-pro), 15 RPM (gemini-1.5-flash)
- **Tokens**: 32,768 tokens/request
- **Quota**: 1500 requests/ngày

Đủ cho ~**1000-2000 conversations/ngày**!

## Nâng cấp lên Paid Tier

Nếu cần nhiều hơn:
1. Vào Google Cloud Console
2. Enable billing
3. Quota sẽ tự động tăng

## Liên Hệ

Nếu gặp vấn đề:
- 📧 Email: support@bookstore.com
- 📞 Hotline: 1900-xxxx
- 📚 Docs: https://ai.google.dev/docs

---

**Last Updated**: April 2026
