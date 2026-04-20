# AI_ecommerce Integration Guide - Bookstore Microservice

**Integration Date:** April 6, 2026  
**Status:** ✅ Ready for Deployment

---

## 1. Overview

The AI_ecommerce project has been successfully integrated into the bookstore-microservice architecture. Two new AI services are now part of the ecosystem:

- **Behavior Analysis Service (Port 8014)** - Analyzes customer behavior, segmentation, and churn prediction
- **Consulting Chatbot Service (Port 8015)** - RAG-based chatbot with knowledge base retrieval

Both services run on shared PostgreSQL infrastructure and integrate through the API Gateway.

---

## 2. Architecture Changes

### 2.1 New Services Added

```yaml
Services Added to docker-compose.yml:
├── behavior-analysis-service:8014
│   ├── Database: PostgreSQL (behavior_db)
│   ├── Dependencies: postgres-db, docker network
│   └── Environment: Upstream service URLs for data collection
│
└── consulting-chatbot-service:8015
    ├── Database: PostgreSQL (chatbot_db)
    ├── Dependencies: postgres-db, behavior-analysis-service
    ├── Volumes: knowledge-base/vectorstore for ChromaDB persistence
    └── Environment: OpenAI API key, embedding configuration
```

### 2.2 API Gateway Enhanced

**New Routes:**
```
POST /api/ai/chat/          - Send chat message to chatbot
GET  /api/ai/behavior/      - Get customer behavior analysis
GET  /api/ai/chat/health/   - Chatbot health check
```

**Access Control:**
- All `/api/ai/` endpoints require customer authentication via session
- JWT token forwarding to AI services for future auth integration
- Health check endpoint is public for monitoring

### 2.3 Database Schema

**New Databases Created:**
- `behavior_db` - Customer segmentation, RFM scores, engagement data
- `chatbot_db` - Chat sessions, messages, feedback, embeddings metadata

**Auto-initialized on service startup** via Django migrations

---

## 3. Quick Start Guide

### 3.1 Prerequisites

```bash
# Required
✓ Docker with docker-compose installed
✓ OpenAI API key (for chatbot only)
✓ 8+ GB free disk space (for model embeddings)
✓ 2+ CPU cores available

# Optional
- Redis for caching (future enhancement)
- Elasticsearch for advanced search (future)
```

### 3.2 Environment Setup

Create or update `.env` file in project root:

```bash
# OpenAI Configuration (REQUIRED for chatbot)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Configure embedding model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Optional: Configure LLM model
LLM_MODEL=gpt-3.5-turbo  # or gpt-4 for better quality
```

### 3.3 Deployment

```bash
# 1. Navigate to project root
cd c:\Users\NITRO\bookstore-microservice

# 2. Build all services (including new AI services)
docker-compose build

# 3. Start all containers
docker-compose up -d

# 4. Verify services are healthy
docker-compose ps
# Look for health status: "healthy" or "running"

# 5. Check logs for any issues
docker-compose logs -f behavior-analysis-service
docker-compose logs -f consulting-chatbot-service
```

### 3.4 Verification

```bash
# Check API Gateway is accessible
curl http://localhost:8000/store/

# Check Chatbot Health
curl http://localhost:8000/api/ai/chat/health/

# Check Behavior Service (via proxy)
# Login first, then:
curl -X GET http://localhost:8000/api/ai/behavior/ \
  -H "Cookie: sessionid=<your-session-id>"
```

---

## 4. Configuration Details

### 4.1 Behavior Analysis Service (8014)

**Django Settings:**
```python
# Database
DB_ENGINE = 'django.db.backends.postgresql'
DB_NAME = 'behavior_db'
DB_USER = 'bookstore_user'
DB_PASSWORD = 'bookstore_pass'
DB_HOST = 'postgres-db'
DB_PORT = 5432

# Upstream Services (for data collection)
CUSTOMER_SERVICE_URL = 'http://customer-service:8001'
ORDER_SERVICE_URL = 'http://order-service:8004'
COMMENT_RATE_SERVICE_URL = 'http://comment-rate-service:8010'

# Analysis Configuration
SEGMENT_TYPES = ['VIP', 'Regular', 'New', 'Churned']
RFM_LOOKBACK_DAYS = 90
CHURN_THRESHOLD = 0.7
ENGAGEMENT_SCALE = 100
```

**API Endpoints:**
```
GET  /api/behavior/health/
GET  /api/behavior/customer/<customer_id>/
POST /api/behavior/analyze-customer/
POST /api/behavior/batch-analyze/
GET  /api/behavior/events/
POST /api/behavior/events/track/
```

**Data Models:**
- `CustomerBehavior` - Cached analysis results with segment, churn_risk, RFM metrics
- `BehaviorEvent` - Individual event tracking (login, purchase, review, wishlist, etc.)
- `AnalysisResult` - Historical analysis records with timestamps

### 4.2 Consulting Chatbot Service (8015)

**Django Settings:**
```python
# Database
DB_ENGINE = 'django.db.backends.postgresql'
DB_NAME = 'chatbot_db'
DB_USER = 'bookstore_user'
DB_PASSWORD = 'bookstore_pass'
DB_HOST = 'postgres-db'
DB_PORT = 5432

# External Services
BEHAVIOR_SERVICE_URL = 'http://behavior-analysis-service:8014'
OPENAI_API_KEY = '<required>'  # from environment

# RAG Configuration
CHROMA_PERSIST_DIRECTORY = '/app/vectorstore'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL = 'gpt-3.5-turbo'
CONTEXT_WINDOW = 10  # messages in context

# Document Paths
KNOWLEDGE_BASE_PATH = '/knowledge-base/documents'
FAQS_PATH = '/knowledge-base/documents/faqs'
PRODUCTS_PATH = '/knowledge-base/documents/products'
POLICIES_PATH = '/knowledge-base/documents/policies'
```

**API Endpoints:**
```
GET  /api/chat/health/
POST /api/chat/message/
GET  /api/chat/sessions/
GET  /api/chat/sessions/<session_id>/
POST /api/chat/search/
POST /api/chat/feedback/
```

**Data Models:**
- `ChatSession` - User conversation session metadata
- `ChatMessage` - Individual messages with role, content, sources
- `ChatFeedback` - User feedback on responses
- `CustomerBehavior` - Cached customer context for personalization

### 4.3 Knowledge Base Population

**Current Status:** ⚠️ Empty - Needs population

**How to Populate:**

```bash
# 1. Create documents in knowledge base
mkdir -p knowledge-base/documents/{faqs,products,policies}

# 2. Add markdown files with content
knowledge-base/documents/
├── faqs/
│   ├── general_questions.md
│   ├── shipping_faqs.md
│   └── payment_faqs.md
├── products/
│   ├── book_categories.md
│   ├── product_guides.md
│   └── bestsellers.md
└── policies/
    ├── shipping_policy.md
    ├── refund_policy.md
    ├── terms_and_conditions.md
    └── privacy_policy.md

# 3. Files are automatically loaded on service startup
# ChromaDB creates embeddings and stores them in vectorstore/
```

**Document Format (Markdown):**
```markdown
# Frequently Asked Questions

## How long does shipping take?
Standard shipping takes 3-5 business days.
Express shipping takes 1-2 business days.

## What is your refund policy?
We offer 30-day returns on all items...
```

---

## 5. Integration Points with Existing Services

### 5.1 Customer Service Integration

**Behavior Analysis** fetches customer data:
```
GET http://customer-service:8001/customers/<customer_id>/
└─ Retrieves: name, email, phone, address, job info
```

### 5.2 Order Service Integration

**Behavior Analysis** collects transaction history:
```
GET http://order-service:8004/orders/?customer_id=<id>
└─ Extracts: RFM metrics, purchase patterns, spending behavior
```

### 5.3 Comment-Rate Service Integration

**Behavior Analysis** gathers engagement data:
```
GET http://comment-rate-service:8010/reviews/?customer_id=<id>
└─ Analyzes: review frequency, rating patterns, engagement level
```

### 5.4 API Gateway Routing

**Customer Session Flow:**
```
1. Customer logs in at /store/login/
2. Session ID stored: request.session['customer_id']
3. Subsequent requests validated by API Gateway middleware
4. AI services receive authenticated customer context
5. Behavior analysis personalized to customer
6. Chatbot responses tailored to customer segment
```

---

## 6. Usage Examples

### 6.1 Get Customer Behavior Analysis

**Request:**
```bash
curl -X GET http://localhost:8000/api/ai/behavior/ \
  -H "Cookie: sessionid=<your-session-id>"
```

**Response:**
```json
{
  "customer_id": 42,
  "segment": "VIP",
  "churn_risk": 0.15,
  "engagement_score": 85,
  "rfm": {
    "recency_days": 3,
    "frequency": 12,
    "monetary_value": 4500.00
  },
  "predicted_categories": ["Fiction", "Biography", "Science"],
  "last_updated": "2026-04-06T10:30:00Z"
}
```

### 6.2 Chat with AI Chatbot

**Request:**
```bash
curl -X POST http://localhost:8000/api/ai/chat/ \
  -H "Content-Type: application/json" \
  -H "Cookie: sessionid=<your-session-id>" \
  -d '{
    "message": "What is your refund policy?",
    "session_id": "<optional-session-id>"
  }'
```

**Response:**
```json
{
  "response": "We offer a 30-day refund policy on all items purchased...",
  "sources": [
    "policies/refund_policy.md",
    "faqs/general_questions.md"
  ],
  "confidence": 0.95,
  "session_id": "uuid-string"
}
```

---

## 7. Monitoring & Troubleshooting

### 7.1 Health Checks

```bash
# Check all service health
docker-compose ps

# Check specific service logs
docker-compose logs chatbot-service -f
docker-compose logs behavior-analysis-service -f

# Manual health endpoint tests
curl http://localhost:8014/api/behavior/health/
curl http://localhost:8015/api/chat/health/
curl http://localhost:8000/api/ai/chat/health/  # via gateway
```

### 7.2 Common Issues

| Issue | Symptom | Solution |
|---|---|---|
| **OpenAI Key Missing** | Chatbot returns 500 errors | Set `OPENAI_API_KEY` env var before starting |
| **Empty Knowledge Base** | Vague chatbot responses | Populate `/knowledge-base/documents/` subdirs |
| **Database Connection Failed** | Services crash on startup | Ensure PostgreSQL running on port 5433 |
| **Port Conflicts** | Service won't start | Check ports 8014, 8015 not already in use |
| **Slow Startup** | 5-10 min first run | Normal - downloads embedding models (~500MB) |

### 7.3 Logs

```bash
# All AI service logs
docker compose logs behavior-analysis-service consulting-chatbot-service -f

# Search for errors
docker-compose logs | grep ERROR

# Check migrations
docker-compose logs consulting-chatbot-service | grep migrate
```

---

## 8. Performance Optimization

### 8.1 Caching Strategy

**Current:** No caching (reads from DB)

**Recommended (Future):**
```python
# Add Redis for:
- ChatMessage cache (30 min TTL)
- Behavior analysis results (1 hour TTL)
- Knowledge base embeddings (persistent)
- Session analysis (1 day TTL)
```

### 8.2 Scalability Considerations

**Current Architecture:**
- Single instance per service
- Shared PostgreSQL database
- No load balancing

**For Production (Future):**
- Replicate services with load balancer (NGINX)
- Read replicas for PostgreSQL
- Async task queue for model training (Celery)
- CDN for knowledge base documents

---

## 9. Security Checklist

- [ ] Change `SECRET_KEY` in django settings (don't use default)
- [ ] Set `DEBUG=False` before production deployment
- [ ] Validate OpenAI API key has appropriate rate limits
- [ ] Restrict knowledge base documents (don't expose sensitive data)
- [ ] Enable HTTPS for AI endpoints
- [ ] Implement API rate limiting per customer
- [ ] Add request signing between services (optional)
- [ ] Audit customer data access logs

---

## 10. Next Steps & Roadmap

### Immediate (Week 1)
- [ ] Populate knowledge base with FAQs and policies
- [ ] Test chatbot responses with sample queries
- [ ] Monitor OpenAI API costs
- [ ] Set up logging and alerts

### Short-term (Month 1)
- [ ] Collect baseline customer data
- [ ] Validate behavior analysis accuracy
- [ ] Fine-tune LLM response parameters
- [ ] Implement feedback collection UI

### Medium-term (Month 2-3)
- [ ] Train custom behavior models on historical data
- [ ] Add multi-language support for chatbot
- [ ] Implement Redis caching layer
- [ ] Performance testing and optimization

### Long-term (Month 4+)
- [ ] A/B testing for chatbot responses
- [ ] Advanced personalization engine
- [ ] Recommendation system integration
- [ ] Knowledge base auto-update pipeline

---

## 11. Docker Compose Services

```yaml
# 14 services total (original 12 + 2 new AI services)

Backend Services (13):
- api-gateway:8000 ← Routes AI requests
- customer-service:8001
- book-service:8002
- cart-service:8003
- order-service:8004
- staff-service:8005
- manager-service:8006
- product-service:8007
- pay-service:8008
- ship-service:8009
- comment-rate-service:8010
- recommender-ai-service:8011
- auth-service:8012
- clothe-service:8013
- behavior-analysis-service:8014 ← NEW
- consulting-chatbot-service:8015 ← NEW

Databases:
- MySQL:8307 (customer_db, bookstore_mysql)
- PostgreSQL:5433 (shared: bookstore_postgres, auth_db, order_db, etc.)

Supporting:
- RabbitMQ:5672 (message queue for async events)
- phpMyAdmin:8080
- pgAdmin:8081
```

---

## 12. Files Changed/Created

**New Directories:**
```
c:\Users\NITRO\bookstore-microservice\
├── behavior-analysis-service/ ← COPIED
├── consulting-chatbot-service/ ← COPIED
└── knowledge-base/ ← COPIED
```

**Modified Files:**
```
c:\Users\NITRO\bookstore-microservice\
├── docker-compose.yml (added AI services)
├── init-scripts/init-ai-databases.sh (NEW)
├── api-gateway/app/views.py (added proxy views)
└── api-gateway/app/urls.py (added AI routes)
```

**Documentation:**
```
d:\AI_ecommerce\COMPLETENESS_ASSESSMENT.md ← Detailed assessment
c:\Users\NITRO\bookstore-microservice\AI_INTEGRATION_GUIDE.md ← This file
```

---

## 13. Support & Additional Resources

**Error Reporting:**
Report integration issues with:
- Docker logs: `docker-compose logs <service>`
- Database: `psql -U bookstore_user -d chatbot_db`
- OpenAI: Check API status and costs

**Configuration Help:**
- AI Service Settings: Review django settings in each service
- Knowledge Base: Check  `/knowledge-base/` directory structure
- API Gateway: Review middleware in `api-gateway/app/middleware.py`

---

**Integration Completed:** ✅ April 6, 2026


