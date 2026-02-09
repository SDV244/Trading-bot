# User Guide

This guide explains how to install, run, and operate the Trading Bot platform.

## 1) What You Are Running

You are running two services:

- API (`FastAPI`) on `http://127.0.0.1:8000`
- Web dashboard (`React + Vite`) on `http://127.0.0.1:5173`

The system is built for:

- spot only
- one configured symbol at a time (`BTCUSDT` default; `ETHUSDT` supported in paper mode)
- strict approval + audit controls

## 2) Prerequisites

- Python `3.11+`
- Poetry
- Node `20+`
- Docker Desktop (optional, recommended for one-command startup)

## 3) Initial Setup

### Backend

```bash
poetry install
```

### Frontend

```bash
cd apps/web
npm install
cd ../..
```

### Environment

```bash
copy .env.example .env
```

Minimum required values:

- `BINANCE_TESTNET=false` (recommended for stable paper data)
- `BINANCE_MARKET_DATA_BASE_URL=https://data-api.binance.vision`
- `TRADING_ACTIVE_STRATEGY=trend_ema`
- `TRADING_REQUIRE_DATA_READY=true`
- `TRADING_SPOT_POSITION_MODE=long_flat`
- `TRADING_PAPER_STARTING_EQUITY=10000`

Optional tuning trial for BTCUSDT paper mode:

- `TRADING_TIMEFRAMES=1h,8h` (slower regime filter, usually fewer trades)
- `TRADING_ACTIVE_STRATEGY=trend_ema_fast` (faster EMA profile + weak-regime filter)

Optional smart adaptive grid profile (paper mode):

- `TRADING_ACTIVE_STRATEGY=smart_grid_ai`
- `TRADING_TIMEFRAMES=1h,4h`
- `TRADING_GRID_LOOKBACK_1H=120`
- `TRADING_GRID_ATR_PERIOD_1H=14`
- `TRADING_GRID_LEVELS=6`
- `TRADING_GRID_SPACING_MODE=geometric` (`arithmetic` also supported)
- `TRADING_GRID_MIN_SPACING_BPS=25`
- `TRADING_GRID_MAX_SPACING_BPS=220`
- `TRADING_GRID_TREND_TILT=1.25`
- `TRADING_GRID_VOLATILITY_BLEND=0.7`
- `TRADING_GRID_TAKE_PROFIT_BUFFER=0.02`
- `TRADING_GRID_STOP_LOSS_BUFFER=0.05`
- `TRADING_GRID_COOLDOWN_SECONDS=0` (set to `60-300` to reduce churn)
- `TRADING_GRID_AUTO_INVENTORY_BOOTSTRAP=true`
- `TRADING_GRID_BOOTSTRAP_FRACTION=1.0`
- `TRADING_GRID_ENFORCE_FEE_FLOOR=false` (set `true` for stricter live readiness)
- `TRADING_GRID_MIN_NET_PROFIT_BPS=30`
- `TRADING_GRID_OUT_OF_BOUNDS_ALERT_COOLDOWN_MINUTES=60`
- `TRADING_ADVISOR_INTERVAL_CYCLES=30`

For secure secrets, use file-based secrets:

- create `.secrets/` (git-ignored)
- put one secret per file (for example `.secrets/BINANCE_API_KEY`)
- set `APP_SECRETS_DIR=./.secrets` in `.env`
- see `docs/SECRETS_MANAGEMENT.md`

PowerShell note:

- Use `$env:APP_SECRETS_DIR = "./.secrets"` for the current shell session.
- `APP_SECRETS_DIR=./.secrets` without `$env:` is not valid PowerShell syntax.

`BINANCE_API_KEY` / `BINANCE_API_SECRET` are only required for live/signed flows.

Risk env keys use `RISK_` prefix:

- `RISK_PER_TRADE`
- `RISK_MAX_DAILY_LOSS`
- `RISK_MAX_EXPOSURE`
- `RISK_FEE_BPS`
- `RISK_SLIPPAGE_BPS`

Recommended for production access control:

- `AUTH_ENABLED=true`
- `AUTH_SECRET_KEY=<long-random-secret>`
- `AUTH_ADMIN_PASSWORD=<strong-password>`
- `AUTH_OPERATOR_PASSWORD=<strong-password>`
- `AUTH_VIEWER_PASSWORD=<strong-password>`

## 4) Run The System

## Option A: Docker (recommended)

```bash
docker compose up --build -d
```

Warmup + start paper mode:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/market/data/fetch?days=30"
curl -s -X POST http://127.0.0.1:8000/api/system/state -H "Content-Type: application/json" -d '{"action":"resume","reason":"operator_start"}'
curl -s -X POST "http://127.0.0.1:8000/api/system/scheduler/start?interval_seconds=60"
```

## Option B: Local processes

Terminal 1:

```bash
poetry run uvicorn apps.api.main:app --reload
```

Terminal 2:

```bash
cd apps/web
npm run dev
```

## 5) First Login / Access Behavior

- If `AUTH_ENABLED=false`:
  - dashboard is open locally
  - API behaves as local admin for protected routes
- If `AUTH_ENABLED=true`:
  - dashboard requires login
  - API requires bearer token for protected routes

## 6) Dashboard Pages

- `Home`
  - health/readiness/state summary
  - equity curve
  - live market curve (configured pair, 1h candles + latest ticker)
- `Trading`
  - current position
  - recent orders/fills
  - explicit idle/waiting explanation when no trade executes
- `Config`
  - active runtime/risk configuration snapshot
- `AI Approvals`
  - pending proposals
  - approve/reject actions
  - real countdown to expiry
- `Logs`
  - audit event timeline
  - category/type/actor/search filters
- `Controls`
  - pause/resume/emergency stop
  - scheduler start/stop
  - run paper cycle
  - AI/optimizer actions

## 7) Daily Operating Workflow

1. Open `Home` and verify API readiness + system state.
2. Check `Trading` for position/orders/fills consistency.
3. Review pending items in `AI Approvals`.
4. Use `Controls` only when needed (pause/resume/emergency actions are audited).
5. Use `Logs` to validate every operational action and proposal decision.

## 8) Key Safety Rules

- no futures
- no margin
- no auto-resume after emergency stop
- AI only proposes; humans decide
- every critical action is audited

## 9) Validation Commands

Backend:

```bash
poetry run pytest -q
poetry run mypy packages apps/api
```

Frontend:

```bash
cd apps/web
npm run lint
npm run typecheck
npm run build
```

## 10) Common Paper-Mode Signals

- `risk_reason=no_inventory_to_sell`:
  - strategy emitted `SELL`, but paper position quantity is `0`.
- `risk_reason=max_exposure_reached`:
  - current position notional is already above `RISK_MAX_EXPOSURE * TRADING_PAPER_STARTING_EQUITY`.
  - increase `TRADING_PAPER_STARTING_EQUITY` (paper only) or reduce seeded position size.
- `risk_reason=already_in_position`:
  - spot mode is `long_flat`, and strategy emitted `BUY` while an open long already exists.

## 11) Starting Capital Guidance

Not financial advice. These values are operationally practical for this bot shape:

- paper testing baseline:
  - `TRADING_PAPER_STARTING_EQUITY=10000`
- live pilot (after acceptance criteria):
  - BTCUSDT: start around `1500-3000 USDT`
  - ETHUSDT: start around `1000-2000 USDT`

Why:

- this avoids tiny-notional noise after fees/slippage
- still keeps risk bounded while you validate behavior for several weeks before scaling

## 12) How To Test End-to-End (Paper Mode)

1. Confirm config is visible in the dashboard:
   - open `Config` page
   - verify `trading_pair`, `active_strategy`, grid parameters, and risk values
2. Warm up market data:
   - `POST /api/market/data/fetch?days=30`
3. Resume + start scheduler:
   - `POST /api/system/state` with `{"action":"resume","reason":"paper_start"}`
   - `POST /api/system/scheduler/start?interval_seconds=60`
4. Open `Controls` page and check `Paper Trading Startup Checklist`:
   - `Market warmup data ready` -> READY
   - `System state RUNNING` -> READY
   - `Scheduler active` -> READY
5. Validate cycle behavior:
   - run `Run One Paper Cycle` from `Controls`
   - inspect `Home` and `Trading` pages for signal/risk/execution updates
6. Validate Telegram:
   - in `Controls`, click `Refresh Notification Status`
   - if enabled, click `Send Telegram Test`
   - confirm the message arrives in your configured chat
7. Optional heartbeat:
   - set `TELEGRAM_HEARTBEAT_ENABLED=true`
   - set `TELEGRAM_HEARTBEAT_HOURS=4`
   - bot sends periodic status updates while scheduler is running
