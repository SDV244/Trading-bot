# API Examples

All examples assume:

- API base: `http://127.0.0.1:8000`
- JSON requests/responses
- `curl` syntax (works in Git Bash/WSL/macOS/Linux)

PowerShell tip:

- replace multiline `\` with backtick continuation or one-line commands.
- for JSON bodies, prefer `Invoke-RestMethod` examples below.

## Postman Import

1. Import collection:
   - `docs/postman/Trading-Bot.postman_collection.json`
2. Import environment:
   - `docs/postman/Trading-Bot.local.postman_environment.json`
3. Select environment `Trading Bot Local`.
4. Run `Auth -> Login`, then copy `access_token` into env variable `token`.
5. Run protected requests.

## 1) Health

```bash
curl -s http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "timestamp": "2026-02-09T05:23:44.370081Z",
  "version": "0.1.0"
}
```

Prometheus metrics:

```bash
curl -s http://127.0.0.1:8000/metrics
```

## 2) Login (when `AUTH_ENABLED=true`)

```bash
curl -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"operator","password":"your-password"}'
```

Example response:

```json
{
  "access_token": "<token>",
  "token_type": "bearer",
  "expires_at": "2026-02-10T13:25:00+00:00",
  "role": "operator",
  "username": "operator"
}
```

Export token:

```bash
export TOKEN="<token>"
```

## 3) Current User

```bash
curl -s http://127.0.0.1:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

## 4) Read System State

```bash
curl -s http://127.0.0.1:8000/api/system/state \
  -H "Authorization: Bearer $TOKEN"
```

## 5) Pause / Resume / Emergency Stop

Pause:

```bash
curl -s -X POST http://127.0.0.1:8000/api/system/state \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"pause","reason":"maintenance","changed_by":"ops_user"}'
```

Resume:

```bash
curl -s -X POST http://127.0.0.1:8000/api/system/state \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"resume","reason":"maintenance done","changed_by":"ops_user"}'
```

PowerShell:

```powershell
$body = '{"action":"resume","reason":"maintenance done","changed_by":"ops_user"}'
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/system/state" `
  -Headers @{ Authorization = "Bearer $env:TOKEN" } `
  -ContentType "application/json" `
  -Body $body
```

Emergency stop:

```bash
curl -s -X POST http://127.0.0.1:8000/api/system/emergency-stop \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"unexpected behavior","changed_by":"ops_user"}'
```

## 6) Scheduler Control

Start:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/system/scheduler/start?interval_seconds=60" \
  -H "Authorization: Bearer $TOKEN"
```

If `TRADING_REQUIRE_DATA_READY=true` and warmup candles are missing, this returns `409`.

Stop:

```bash
curl -s -X POST http://127.0.0.1:8000/api/system/scheduler/stop \
  -H "Authorization: Bearer $TOKEN"
```

System readiness:

```bash
curl -s http://127.0.0.1:8000/api/system/readiness \
  -H "Authorization: Bearer $TOKEN"
```

Data requirements:

```bash
curl -s http://127.0.0.1:8000/api/market/data/requirements \
  -H "Authorization: Bearer $TOKEN"
```

Notification status:

```bash
curl -s http://127.0.0.1:8000/api/system/notifications/status \
  -H "Authorization: Bearer $TOKEN"
```

Send Telegram test:

```bash
curl -s -X POST http://127.0.0.1:8000/api/system/notifications/test \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Trading Bot test notification","body":"Connectivity check"}'
```

Circuit breaker status:

```bash
curl -s http://127.0.0.1:8000/api/system/circuit-breakers/status \
  -H "Authorization: Bearer $TOKEN"
```

Reset spot circuit breaker:

```bash
curl -s -X POST http://127.0.0.1:8000/api/system/circuit-breakers/binance_spot/reset \
  -H "Authorization: Bearer $TOKEN"
```

Reload runtime configuration (admin):

```bash
curl -s -X POST http://127.0.0.1:8000/api/system/config/reload \
  -H "Authorization: Bearer $TOKEN"
```

Run balance reconciliation:

```bash
curl -s "http://127.0.0.1:8000/api/system/reconciliation?warning_tolerance=1&critical_tolerance=100" \
  -H "Authorization: Bearer $TOKEN"
```

Readiness status (returns `503` if dependencies are down):

```bash
curl -i -s http://127.0.0.1:8000/ready
```

## 7) Run One Paper Cycle

```bash
curl -s -X POST http://127.0.0.1:8000/api/trading/paper/cycle \
  -H "Authorization: Bearer $TOKEN"
```

Force-close open paper inventory:

```bash
curl -s -X POST http://127.0.0.1:8000/api/trading/paper/close-all-positions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason":"manual_close_all_positions"}'
```

## 8) Trading Data

Config:

```bash
curl -s http://127.0.0.1:8000/api/trading/config \
  -H "Authorization: Bearer $TOKEN"
```

Important fields in this response include:

- `active_strategy`
- `supported_strategies`
- `require_data_ready`
- `spot_position_mode`
- `paper_starting_equity`
- `advisor_interval_cycles`
- `min_cycle_interval_seconds`
- `reconciliation_interval_cycles`
- `reconciliation_warning_tolerance`, `reconciliation_critical_tolerance`
- `grid_lookback_1h`, `grid_atr_period_1h`, `grid_levels`
- `grid_min_spacing_bps`, `grid_max_spacing_bps`
- `grid_trend_tilt`, `grid_volatility_blend`

Position:

```bash
curl -s http://127.0.0.1:8000/api/trading/position \
  -H "Authorization: Bearer $TOKEN"
```

Orders:

```bash
curl -s "http://127.0.0.1:8000/api/trading/orders?limit=20" \
  -H "Authorization: Bearer $TOKEN"
```

Fills:

```bash
curl -s "http://127.0.0.1:8000/api/trading/fills?limit=20" \
  -H "Authorization: Bearer $TOKEN"
```

Metrics:

```bash
curl -s http://127.0.0.1:8000/api/trading/metrics \
  -H "Authorization: Bearer $TOKEN"
```

Equity history:

```bash
curl -s "http://127.0.0.1:8000/api/trading/equity/history?days=30" \
  -H "Authorization: Bearer $TOKEN"
```

## 9) AI Proposals + Approvals

Generate proposals:

```bash
curl -s -X POST http://127.0.0.1:8000/api/ai/proposals/generate \
  -H "Authorization: Bearer $TOKEN"
```

LLM advisor status:

```bash
curl -s http://127.0.0.1:8000/api/ai/llm/status \
  -H "Authorization: Bearer $TOKEN"
```

LLM provider connectivity test:

```bash
curl -s -X POST http://127.0.0.1:8000/api/ai/llm/test \
  -H "Authorization: Bearer $TOKEN"
```

Multi-agent status:

```bash
curl -s http://127.0.0.1:8000/api/ai/multi-agent/status \
  -H "Authorization: Bearer $TOKEN"
```

Multi-agent dry-run test:

```bash
curl -s -X POST http://127.0.0.1:8000/api/ai/multi-agent/test \
  -H "Authorization: Bearer $TOKEN"
```

List approvals:

```bash
curl -s "http://127.0.0.1:8000/api/ai/approvals?status=PENDING&limit=50" \
  -H "Authorization: Bearer $TOKEN"
```

Approve proposal:

```bash
curl -s -X POST http://127.0.0.1:8000/api/ai/approvals/1/approve \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"decided_by":"operator","reason":"validated"}'
```

Reject proposal:

```bash
curl -s -X POST http://127.0.0.1:8000/api/ai/approvals/1/reject \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"decided_by":"operator","reason":"insufficient evidence"}'
```

Expire pending check:

```bash
curl -s -X POST http://127.0.0.1:8000/api/ai/approvals/expire-check \
  -H "Authorization: Bearer $TOKEN"
```

## 10) Audit Event Filtering

By category:

```bash
curl -s "http://127.0.0.1:8000/api/ai/events?category=system&limit=100" \
  -H "Authorization: Bearer $TOKEN"
```

By actor + search:

```bash
curl -s "http://127.0.0.1:8000/api/ai/events?actor=operator&search=emergency&limit=100" \
  -H "Authorization: Bearer $TOKEN"
```

By time range:

```bash
curl -s "http://127.0.0.1:8000/api/ai/events?start_at=2026-02-01T00:00:00Z&end_at=2026-02-09T23:59:59Z" \
  -H "Authorization: Bearer $TOKEN"
```

## 11) Optimizer Endpoints

Train:

```bash
curl -s -X POST http://127.0.0.1:8000/api/ai/optimizer/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"timesteps":1024}'
```

Proposal:

```bash
curl -s http://127.0.0.1:8000/api/ai/optimizer/propose \
  -H "Authorization: Bearer $TOKEN"
```

## 12) Live Order (Admin + Live Mode Required)

```bash
curl -s -X POST http://127.0.0.1:8000/api/trading/live/order \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "side":"BUY",
    "quantity":"0.001",
    "ui_confirmed":true,
    "reauthenticated":true,
    "safety_acknowledged":true,
    "idempotency_key":"manual-live-buy-2026-02-10-001",
    "reason":"manual controlled test"
  }'
```

If the request is retried with the same `idempotency_key`, the API returns the original confirmed result instead of sending a duplicate order.

## 13) Rate Limit Behavior

When limits are exceeded, API returns `429`:

```json
{
  "detail": "Rate limit exceeded",
  "category": "api",
  "limit_per_minute": 600,
  "retry_after_seconds": 12
}
```

Rate-limit headers:

- `Retry-After`
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Window`
