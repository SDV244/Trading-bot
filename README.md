# Trading Bot (Spot BTCUSDT)

Production-oriented local trading platform with strict safety controls:

- Spot only (no futures, no margin)
- Single symbol (`BTCUSDT`) for v1
- AI can only propose changes; humans approve/reject
- Approval timeout triggers `EMERGENCY_STOP`
- Full audit trail in SQLite
- Local web dashboard for ops/control

## Architecture

### Backend (`apps/api`)

- FastAPI service with:
  - system state APIs (`RUNNING`, `PAUSED`, `EMERGENCY_STOP`)
  - scheduler control APIs
  - market data APIs
  - trading APIs (position, orders, fills, metrics, equity, paper cycle)
  - AI approval APIs (generate proposals, queue, approve/reject, events)

### Core (`packages/core`)

- `execution/paper_engine.py`: realistic paper fill simulation with fees/slippage/guards
- `strategies/`: plugin-style strategy framework + EMA trend strategy
- `risk/engine.py`: risk gate and sizing
- `trading_cycle.py`: signal -> risk -> execution -> persistence -> metrics/audit
- `scheduler.py`: periodic background cycles + approval expiry checks + advisor triggers
- `ai/`:
  - `advisor.py`: generates proposal candidates from metrics/slippage anomalies
  - `approval_gate.py`: pending queue, approvals, config versioning, timeout safety
- `metrics/calculator.py`: full performance metrics and composite score

### Adapters (`packages/adapters`)

- `binance_spot.py`: async market data adapter with retry/rate-limit controls
- `telegram_bot.py`: optional Telegram notifications for critical/info alerts

### Frontend (`apps/web`)

- React + TypeScript + Vite + Tailwind
- Routed dashboard pages:
  - Home, Trading, Config, AI Approvals, Logs, Controls
- Dashboard includes:
  - system status, scheduler status, equity trend
  - position, metrics, recent orders/fills
  - controls (pause/resume/emergency, manual paper cycle, scheduler start/stop)
  - AI approvals queue with approve/reject actions
  - audit event feed with server-side filtering
  - optional login + role-based access when `AUTH_ENABLED=true`

## Safety Rules

- Spot only
- Manual resume required after emergency stop
- AI proposals are never auto-applied without approval
- Approval expiry triggers emergency stop
- Every significant action writes an audit event

## Quick Start

## 1) Install backend

```bash
poetry install
```

## 2) Install frontend

```bash
cd apps/web
npm install
cd ../..
```

## 3) Configure environment

```bash
copy .env.example .env
```

Set required values in `.env`:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_TESTNET=true` (recommended for local)
- optional Telegram:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID`
- runtime profile:
  - `APP_ENV=dev|staging|prod`
  - optional `APP_SECRETS_DIR=/path/to/secrets` (file-per-secret)
- auth/rbac (recommended for production):
  - `AUTH_ENABLED=true`
  - `AUTH_SECRET_KEY=<long-random-secret>`
  - `AUTH_ADMIN_PASSWORD`, `AUTH_OPERATOR_PASSWORD`, `AUTH_VIEWER_PASSWORD`

## 4) Run API

```bash
poetry run uvicorn apps.api.main:app --reload
```

## 5) Run Web

```bash
cd apps/web
npm run dev
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- Web: `http://127.0.0.1:5173`

## Validation Commands

Backend:

```bash
poetry run pytest tests/unit -v --tb=short
poetry run mypy packages/core packages/research apps/api
poetry run ruff check .
```

Frontend:

```bash
cd apps/web
npm run lint
npm run typecheck
npm run build
```

Database migrations:

```bash
poetry run alembic upgrade head
```

## Operational APIs

System:

- `GET /api/system/state`
- `POST /api/system/state`
- `POST /api/system/emergency-stop`
- `GET /api/system/scheduler`
- `POST /api/system/scheduler/start`
- `POST /api/system/scheduler/stop`

Trading:

- `POST /api/trading/paper/cycle`
- `POST /api/trading/live/order` (requires `TRADING_LIVE_MODE=true` and checklist)
- `GET /api/trading/position`
- `GET /api/trading/orders`
- `GET /api/trading/fills`
- `GET /api/trading/metrics`
- `GET /api/trading/equity/history`

AI approvals:

- `POST /api/ai/proposals/generate`
- `GET /api/ai/approvals`
- `POST /api/ai/approvals/{id}/approve`
- `POST /api/ai/approvals/{id}/reject`
- `POST /api/ai/approvals/expire-check`
- `GET /api/ai/events`
- `POST /api/ai/optimizer/train`
- `GET /api/ai/optimizer/propose`

Auth:

- `POST /api/auth/login`
- `GET /api/auth/me`

## Docker Deployment

Build and run all services locally:

```bash
docker compose up --build -d
```

Services:

- API: `http://127.0.0.1:8000`
- Web: `http://127.0.0.1:5173`

Production images are built/published by `.github/workflows/cd.yml` on `main`.

## Notes

- SQLite file path defaults to `./data/trading.db`; parent directory is auto-created.
- Alembic migrations run at startup when `DB_AUTO_MIGRATE=true`.
- Scheduler executes in-process with API.
- Current release is paper-trading only; live engine is intentionally gated for future release.
