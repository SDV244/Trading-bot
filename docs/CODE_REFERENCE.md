# Code Reference

This is a practical map of where logic lives and what to edit for each change.

## Repository Layout

- `apps/api`
  - FastAPI entrypoint, middleware, route handlers, auth dependencies
- `apps/web`
  - React dashboard, routing, auth/session handling, page UIs
- `packages/core`
  - business logic (state, scheduler, strategy, risk, execution, metrics, AI, DB access)
- `packages/adapters`
  - Binance and Telegram integrations
- `packages/research`
  - walk-forward evaluation tooling
- `tests/unit`
  - unit/integration-style API/core tests

## Backend: API Layer

- `apps/api/main.py`
  - app creation
  - middleware registration
  - router registration
- `apps/api/routers/health.py`
  - health/readiness checks
- `apps/api/routers/auth.py`
  - login and current-user endpoints
- `apps/api/routers/system.py`
  - state machine + scheduler control + system readiness endpoint
- `apps/api/routers/market.py`
  - market/candle/data-fetch + data-requirements endpoints
- `apps/api/routers/trading.py`
  - config/position/orders/fills/metrics/paper cycle/live order endpoints
- `apps/api/routers/ai.py`
  - proposal/approval/events/optimizer endpoints
- `apps/api/security/auth.py`
  - token signing/verification
  - credential auth
  - RBAC dependency guards
- `apps/api/middleware/request_id.py`
  - request correlation IDs
- `apps/api/middleware/security_headers.py`
  - hardened response headers

## Backend: Core Domain

- `packages/core/config.py`
  - all environment-driven settings (including strategy selection, data readiness gate, paper starting equity)
  - file-based secrets via `APP_SECRETS_DIR`
- `packages/core/state.py`
  - system state machine and transition rules
- `packages/core/scheduler.py`
  - periodic cycle execution and status tracking
- `packages/core/trading_cycle.py`
  - orchestration of strategy -> risk -> execution -> persistence
  - active strategy instantiation from registry
  - strategy candle requirements + readiness checks
- `packages/core/data_fetcher.py`
  - batch candle fetching and storage
- `packages/core/audit.py`
  - audit event append helpers
- `packages/core/logging_setup.py`
  - structured/local logging configuration

### Strategies

- `packages/core/strategies/base.py` (signal contract + registry)
- `packages/core/strategies/trend.py`
- `packages/core/strategies/mean_reversion.py`
- `packages/core/strategies/breakout.py`
- `packages/core/strategies/smart_grid.py` (adaptive grid with regime tilt + volatility spacing)

### Risk + Execution + Metrics

- `packages/core/risk/engine.py`
- `packages/core/execution/paper_engine.py`
- `packages/core/execution/live_engine.py`
- `packages/core/metrics/calculator.py`

### AI Workflow

- `packages/core/ai/advisor.py`
- `packages/core/ai/approval_gate.py`
- `packages/core/ai/drl_optimizer.py`

### Data Access

- `packages/core/database/models.py`
- `packages/core/database/repositories.py`
- `packages/core/database/session.py`
- `packages/core/database/migrations.py`

## Frontend: Web Dashboard

- `apps/web/src/App.tsx`
  - route table and access wrapping
- `apps/web/src/auth.tsx`
  - login state, token lifecycle, role helpers
- `apps/web/src/dashboard.tsx`
  - periodic shared data loader for dashboard pages
- `apps/web/src/api.ts`
  - typed API client + auth header injection
- `apps/web/src/pages/*`
  - page-level UIs
- `apps/web/src/components/*`
  - shared layout and visual components

## Testing

- `tests/unit/test_api.py`
  - API endpoint behavior and security headers
- `tests/unit/test_auth_api.py`
  - auth + RBAC behavior
- `tests/unit/test_*`
  - strategy/risk/execution/ai/data layer coverage

## Where To Change Common Features

- Add new API endpoint:
  - route file in `apps/api/routers`
  - add auth dependency in endpoint signature
  - add tests under `tests/unit`
- Add new strategy:
  - implement in `packages/core/strategies`
  - register in strategy registry
  - add unit tests
- Add dashboard section:
  - page component in `apps/web/src/pages`
  - route in `apps/web/src/App.tsx`
  - API methods in `apps/web/src/api.ts`
