# API + RBAC Reference

Base URL:

- `http://127.0.0.1:8000`

Swagger UI:

- `/docs`

## Authentication Endpoints

- `POST /api/auth/login`
  - input: `username`, `password`
  - output: bearer token + role + expiry
- `GET /api/auth/me`
  - returns current auth principal

## Role Levels

- `viewer`
  - read-only operations
- `operator`
  - can execute operational controls and approval decisions
- `admin`
  - includes operator + admin-only controls (for example live order endpoint)

## Endpoint Groups

## Health

- `GET /health`
- `GET /ready` (returns `503` when DB/Binance dependencies are not ready)
- `GET /metrics` (Prometheus format, when `OBS_METRICS_ENABLED=true`)

Auth requirement:

- no token required

## System

- `GET /api/system/state` (`viewer+`)
- `GET /api/system/readiness` (`viewer+`)
- `POST /api/system/state` (`operator+`)
- `POST /api/system/emergency-stop` (`operator+`)
- `GET /api/system/state/history` (`viewer+`)
- `GET /api/system/scheduler` (`viewer+`)
- `POST /api/system/scheduler/start` (`operator+`, returns `409` if warmup data is missing and `TRADING_REQUIRE_DATA_READY=true`)
- `POST /api/system/scheduler/stop` (`operator+`)
- `GET /api/system/circuit-breakers/status` (`viewer+`)
- `POST /api/system/circuit-breakers/{name}/reset` (`operator+`)
- `POST /api/system/config/reload` (`admin+`, safeguards runtime `live_mode` toggle)
- `GET /api/system/reconciliation` (`operator+`, can trigger emergency-stop on critical mismatch)

## Market

- `GET /api/market/price` (`viewer+`)
- `GET /api/market/candles` (`viewer+`)
- `GET /api/market/data/status` (`viewer+`)
- `GET /api/market/data/requirements` (`viewer+`)
- `POST /api/market/data/fetch` (`operator+`)

## Trading

- `GET /api/trading/config` (`viewer+`)
- `GET /api/trading/position` (`viewer+`)
- `GET /api/trading/orders` (`viewer+`)
- `GET /api/trading/fills` (`viewer+`)
- `GET /api/trading/metrics` (`viewer+`)
- `GET /api/trading/equity/history` (`viewer+`)
- `POST /api/trading/paper/cycle` (`operator+`)
- `POST /api/trading/paper/close-all-positions` (`operator+`, force-closes open paper inventory)
- `POST /api/trading/paper/reset` (`operator+`, requires system `PAUSED`)
- `POST /api/trading/live/order` (`admin+`, and `TRADING_LIVE_MODE=true`)
  - supports optional `idempotency_key` to deduplicate retries safely

## Rate Limiting

- Middleware-based per-client limits are enabled by default.
- General limit: `API_RATE_LIMIT_REQUESTS_PER_MINUTE`
- Auth endpoints limit: `API_RATE_LIMIT_AUTH_REQUESTS_PER_MINUTE`
- Exempt prefixes: `API_RATE_LIMIT_EXEMPT_PATHS`
- On limit breach, API returns `429` with:
  - `Retry-After`
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Window`

## AI + Approvals

- `POST /api/ai/proposals/generate` (`operator+`)
- `GET /api/ai/llm/status` (`viewer+`)
- `POST /api/ai/llm/test` (`operator+`)
- `GET /api/ai/multi-agent/status` (`viewer+`)
- `POST /api/ai/multi-agent/test` (`operator+`)
- `GET /api/ai/approvals` (`viewer+`)
- `POST /api/ai/approvals/{id}/approve` (`operator+`)
- `POST /api/ai/approvals/{id}/reject` (`operator+`)
- `POST /api/ai/approvals/expire-check` (`operator+`)
- `GET /api/ai/events` (`viewer+`)
- `POST /api/ai/optimizer/train` (`operator+`)
- `GET /api/ai/optimizer/propose` (`operator+`)

## Event Filtering Query Parameters

`GET /api/ai/events` supports:

- `category`
- `event_type`
- `actor`
- `search`
- `start_at`
- `end_at`
- `limit`

## Security Headers

API responses include hardened defaults:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy`
- `Referrer-Policy`
- `Permissions-Policy`
