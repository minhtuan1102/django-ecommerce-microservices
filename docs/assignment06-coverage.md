# Assignment 06 Coverage

## Implemented
- Central auth-service with JWT issue/validate
- RabbitMQ-based event bus
- Saga flow with compensation
- API Gateway middleware: auth validation, path-based JWT protection, RBAC checks, rate limiting, request logging
- Health and metrics endpoints for core services
- Fault simulation flags for worker services
- Load test script + sample results
- Architecture justification report

## Remaining hardening suggestions
- Replace local in-memory counters with Redis/Prometheus exporters
- Add distributed tracing (OpenTelemetry)
- Add contract tests for saga messages
