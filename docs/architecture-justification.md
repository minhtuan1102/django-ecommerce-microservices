# Architecture Justification (Assignment 06)

## 1. JWT Authentication Service
A dedicated `auth-service` issues and validates JWT tokens. API Gateway validates token claims through auth-service and enforces RBAC for protected paths.

## 2. Saga Pattern
Order flow is event-driven with compensation:
1. Create order (`pending`)
2. Payment worker reserves payment
3. Shipping worker reserves shipping
4. Order worker confirms order
5. On failure events, order is cancelled and stock is compensated

## 3. Event Bus
RabbitMQ decouples services and avoids synchronous REST dependency chains for critical distributed transactions.

## 4. API Gateway
Gateway responsibilities include:
- routing
- token validation hook
- role checks (RBAC)
- request logging
- rate limiting

## 5. Observability
Health and metrics endpoints are exposed for core services. Metrics follow Prometheus text format for easy scraping.

## 6. Fault Simulation
Environment flags in workers support forced payment/shipping failure to validate compensation paths.
