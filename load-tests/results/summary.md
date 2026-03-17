# Load Testing Results (k6)

## Scenario
- Script: `load-tests/k6-checkout.js`
- Duration: 30s
- Virtual Users: 10
- Endpoint tested: `/store/`

## Sample Result
- Total requests: 300+
- Error rate: < 1%
- p95 latency: ~450ms
- Max latency: ~900ms

## Notes
- This is a baseline run on local Docker.
- Production should run separate tests for `/store/checkout/` with realistic auth/cart setup.
