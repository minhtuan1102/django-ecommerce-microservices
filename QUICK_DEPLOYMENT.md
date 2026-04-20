# 🚀 Quick Deployment Checklist

**Status:** ✅ Integration Complete - Ready to Deploy

---

## Pre-Deployment Checklist

### ✅ Code Integration (DONE)
- [x] AI services copied to project
- [x] docker-compose.yml updated
- [x] Database init script created
- [x] API Gateway configured
- [x] Routes added (/api/ai/*)

### ⚠️ Configuration (ACTION REQUIRED)

#### 1. OpenAI API Key (REQUIRED)
```bash
# Get key from: https://platform.openai.com/account/api-keys

# Option A: Create .env file
echo OPENAI_API_KEY=sk-your-key-here >> .env

# Option B: Set environment variable
export OPENAI_API_KEY=sk-your-key-here

# Option C: Add to docker-compose before starting
# (See AI_INTEGRATION_GUIDE.md section 3.2)
```

#### 2. Knowledge Base (RECOMMENDED)
```bash
# Navigate to knowledge base directory
cd knowledge-base/documents/

# Create required subdirectories
mkdir -p faqs products policies

# Add sample markdown files
# Example: faqs/general_questions.md
echo "# FAQs

## How long does shipping take?
Standard: 3-5 days | Express: 1-2 days

## What's your refund policy?
30-day returns on all items." > faqs/general_questions.md

# Add more documents as needed
```

#### 3. System Prerequisites
- [ ] Docker installed and running
- [ ] docker-compose installed
- [ ] 8+ GB RAM available
- [ ] 10+ GB free disk space
- [ ] Ports 8014, 8015 available (AI services)

---

## Deployment Steps

### Step 1: Prepare Environment
```bash
cd c:\Users\NITRO\bookstore-microservice
```

### Step 2: Set OpenAI Key
```bash
# Create .env file if not exists
# Add: OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Build Services
```bash
docker-compose build
# Takes 3-5 minutes. Downloads base images and dependencies.
```

### Step 4: Start Services
```bash
docker-compose up -d

# Wait 2-3 minutes for initialization
# Services download embedding models on first run (~500MB)
```

### Step 5: Verify Deployment
```bash
# Check all services running
docker-compose ps
# Look for: "healthy" or "running" status for AI services

# Test API endpoints
curl http://localhost:8000/api/ai/chat/health/
# Expected: {"status": "healthy", "service": "consulting-chatbot"}

# Access frontend
open http://localhost:8000/store/
```

### Step 6: Test Services
```bash
# 1. Login to storefront
# 2. Request: GET http://localhost:8000/api/ai/behavior/
# 3. Response: Customer segment, churn risk, engagement score

# 4. Request: POST http://localhost:8000/api/ai/chat/
# Body: {"message": "What is your refund policy?"}
# Response: Chatbot answer with citations
```

---

## Verification Checklist

After deployment, verify:

- [ ] All 15 services running: `docker-compose ps`
- [ ] Both AI services show "running": 
  - behavior-analysis-service:8014
  - consulting-chatbot-service:8015
- [ ] Chatbot health check responds: `curl .../api/ai/chat/health/`
- [ ] PostgreSQL has new databases: behavior_db, chatbot_db
- [ ] Knowledge base documents accessible: check vectorstore
- [ ] Customer data flowing from upstream services

---

## Troubleshooting

### Issue: "OpenAI API key not found"
```bash
# Solution: Add OPENAI_API_KEY to .env and restart
export OPENAI_API_KEY=sk-xxx
docker-compose down
docker-compose up -d
```

### Issue: "Chatbot service won't start"
```bash
# Check logs:
docker-compose logs consulting-chatbot-service

# Common cause: Missing OpenAI key or DB connection
# Solution: Set key and ensure postgresql is healthy
```

### Issue: "Embedding model downloading (slow startup)"
```bash
# Normal behavior on first run
# Services download sentence-transformers (~500MB)
# Wait 5-10 minutes, then restart

# Monitor:
docker-compose logs consulting-chatbot-service | grep embed
```

### Issue: Port 8014/8015 already in use
```bash
# Find what's using the port:
netstat -ano | findstr :8014

# Solution: Change ports in docker-compose.yml or stop conflicting service
```

---

## After Deployment

### Monitor Performance
```bash
# Watch logs in real-time
docker-compose logs -f consulting-chatbot-service

# Check OpenAI costs
# Dashboard: https://platform.openai.com/account/usage/overview
```

### Populate Knowledge Base (Ongoing)
```bash
# Add more documents over time
# Changes automatically picked up (requires service restart)
docker-compose restart consulting-chatbot-service
```

### Scaling (Future)
```bash
# For production, consider:
# 1. Redis caching layer
# 2. Load balancer for multiple instances
# 3. Separate databases per service
# 4. Monitoring and alerting setup
```

---

## Quick Reference

| Component | Port | Status | Action |
|-----------|------|--------|--------|
| API Gateway | 8000 | ✅ Running | Access http://localhost:8000 |
| Behavior Service | 8014 | ✅ Running | GET /api/ai/behavior/ |
| Chatbot Service | 8015 | ✅ Running | POST /api/ai/chat/ |
| PostgreSQL | 5433 | ✅ Running | pgAdmin: http://localhost:8081 |
| MySQL | 3307 | ✅ Running | phpMyAdmin: http://localhost:8080 |

---

## Documentation Links

- **Quick Start:** AI_INTEGRATION_GUIDE.md (section 3)
- **Full Guide:** AI_INTEGRATION_GUIDE.md
- **Technical Details:** COMPLETENESS_ASSESSMENT.md
- **Overview:** INTEGRATION_SUMMARY.md

---

## Support

**Can't find something?**
1. Check logs: `docker-compose logs <service-name>`
2. Review guides in project root
3. Check memory files for previous integration notes

**Need to restart services?**
```bash
docker-compose down
docker-compose up -d
```

**Need to reset databases?**
```bash
docker-compose down -v  # -v removes volumes/data
docker-compose up -d    # recreates databases
```

---

**Last Updated:** April 6, 2026  
**Status:** ✅ Ready to Deploy

