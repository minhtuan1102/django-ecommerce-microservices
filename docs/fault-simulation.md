# Fault Simulation Guide

## Purpose
Validate Saga compensation behavior under failure.

## Payment failure simulation
Set env in `pay-worker`:
- `FORCE_PAYMENT_FAIL=true`

Then recreate worker:
```bash
docker-compose up -d --build pay-worker
```

Expected outcome:
- payment publishes `payment_failed_queue`
- order becomes `cancelled`
- order worker restores stock

## Shipping failure simulation
Set env in `ship-worker`:
- `FORCE_SHIPPING_FAIL=true`

Then recreate worker:
```bash
docker-compose up -d --build ship-worker
```

Expected outcome:
- shipping publishes `shipping_failed_queue`
- order becomes `cancelled`
- stock compensation is executed
