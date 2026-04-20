# AI Services Deployment & Integration Guide

## Overview

This guide describes how to deploy and integrate the AI services (Behavior Analysis & Consulting Chatbot) into the bookstore-microservice system.

## Prerequisites

- Docker & Docker Compose installed
- Python 3.10+ (for local development)
- OpenAI API key (for chatbot service)

## Services Summary

| Service | Port | Purpose |
|---------|------|---------|
| behavior-analysis-service | 8014 | Customer segmentation, churn prediction, behavior tracking |
| consulting-chatbot-service | 8015 | RAG-based AI chatbot for customer support |

## Model Training Results

The behavior analysis model was trained on **Kaggle Ecommerce Customer Behavior Dataset** (50,000 records):

| Metric | Value |
|--------|-------|
| Segment Accuracy | **83%** |
| Category Prediction Accuracy | **92.95%** |
| Churn Prediction AUC | **0.937** |

Model files location: `behavior-analysis-service/data/models/`

## Quick Start

### 1. Setup Environment

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 2. Start All Services

```bash
# Build and start all services (including AI)
docker-compose up -d --build

# Or start only AI services
docker-compose up -d behavior-analysis-service consulting-chatbot-service
```

### 3. Verify Deployment

```bash
# Check service health
curl http://localhost:8014/api/health/          # Behavior service
curl http://localhost:8015/api/health/          # Chatbot service

# Check knowledge base status
curl http://localhost:8015/api/chat/index-status/
```

## API Endpoints

### Behavior Analysis Service (Port 8014)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health/` | GET | Health check |
| `/api/analyze-customer/` | POST | Analyze single customer |
| `/api/batch-analyze/` | POST | Analyze multiple customers |
| `/api/track-event/` | POST | Track customer behavior event |
| `/api/customer/{id}/` | GET | Get customer analysis |

#### Analyze Customer Example

```bash
curl -X POST http://localhost:8014/api/analyze-customer/ \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 123,
    "include_recommendations": true
  }'
```

Response:
```json
{
  "customer_id": 123,
  "segment": "VIP",
  "segment_probability": 0.85,
  "churn_risk": 0.12,
  "engagement_score": 78.5,
  "rfm_scores": {"recency": 5, "frequency": 4, "monetary": 5},
  "next_purchase_category": "Technology",
  "recommendations": [...]
}
```

### Consulting Chatbot Service (Port 8015)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health/` | GET | Health check |
| `/api/chat/` | POST | Send chat message |
| `/api/chat/sessions/` | GET | List chat sessions |
| `/api/chat/search/` | POST | Search knowledge base |
| `/api/chat/feedback/` | POST | Submit feedback |
| `/api/chat/index-status/` | GET | Check KB index status |

#### Chat Example

```bash
curl -X POST http://localhost:8015/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123-session",
    "message": "Chính sách đổi trả như thế nào?",
    "customer_id": 123
  }'
```

Response:
```json
{
  "session_id": "user-123-session",
  "response": "Chính sách đổi trả của chúng tôi...",
  "sources": ["policies/refund_policy.md"],
  "confidence": 0.92
}
```

## Via API Gateway (Port 8000)

All AI services are also accessible through the API Gateway:

| Gateway Endpoint | Proxies To |
|------------------|------------|
| `/api/ai/behavior/` | behavior-analysis-service |
| `/api/ai/behavior/track/` | Track behavior events |
| `/api/ai/chat/` | consulting-chatbot-service |
| `/api/ai/chat/health/` | Chatbot health check |
| `/api/ai/recommendations/` | Get AI recommendations |

## Knowledge Base

### Structure

```
knowledge-base/
├── documents/
│   ├── faqs/           # FAQ documents
│   │   ├── general_faqs.md
│   │   └── shipping_faqs.md
│   ├── policies/       # Store policies
│   │   ├── shipping_policy.md
│   │   ├── refund_policy.md
│   │   ├── privacy_policy.md
│   │   └── terms_of_service.md
│   └── products/       # Product information
│       ├── book_categories.md
│       ├── recommendation_guide.md
│       └── promotions_membership.md
└── vectorstore/        # ChromaDB vector database
```

### Rebuild Index

If you update knowledge base documents:

```bash
curl -X POST http://localhost:8015/api/chat/rebuild-index/
```

## Integration with Frontend

### AI Assistant Page

Access the AI assistant UI at: `http://localhost:8000/store/ai-assistant/`

### JavaScript Integration

```javascript
// Send chat message
async function sendChatMessage(message, sessionId) {
    const response = await fetch('/api/ai/chat/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            session_id: sessionId,
            message: message,
            customer_id: getCurrentCustomerId()
        })
    });
    return response.json();
}

// Track customer behavior
async function trackBehavior(eventType, data) {
    await fetch('/api/ai/behavior/track/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            customer_id: getCurrentCustomerId(),
            event_type: eventType,  // 'page_view', 'add_to_cart', 'purchase', etc.
            ...data
        })
    });
}
```

## Troubleshooting

### Chatbot not responding

1. Check if OpenAI API key is set:
   ```bash
   docker-compose exec consulting-chatbot-service env | grep OPENAI
   ```

2. Check service logs:
   ```bash
   docker-compose logs consulting-chatbot-service
   ```

### Empty knowledge base responses

1. Check index status:
   ```bash
   curl http://localhost:8015/api/chat/index-status/
   ```

2. Rebuild index:
   ```bash
   curl -X POST http://localhost:8015/api/chat/rebuild-index/
   ```

### Model not loading

1. Verify model files exist:
   ```bash
   docker-compose exec behavior-analysis-service ls -la /app/data/models/
   ```

2. Check service logs:
   ```bash
   docker-compose logs behavior-analysis-service
   ```

## Performance Notes

- First chatbot request may be slow (loading embedding model ~500MB)
- Behavior analysis predictions are cached for 1 hour
- Knowledge base uses ChromaDB with cosine similarity
- Recommended: Add Redis cache for production

## Security Considerations

1. Keep `OPENAI_API_KEY` secret
2. Enable authentication for AI endpoints in production
3. Rate limit chatbot requests (recommended: 10 req/min per user)
4. Sanitize user inputs before processing

## Costs

- **OpenAI API**: ~$0.002-0.01 per chat message (GPT-3.5-turbo)
- **Embedding model**: Free (runs locally)
- **Behavior analysis**: Free (runs locally)
