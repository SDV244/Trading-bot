# User Guide

This guide explains how to install, run, and operate the Trading Bot platform.

## 1) What You Are Running

You are running two services:

- API (`FastAPI`) on `http://127.0.0.1:8000`
- Web dashboard (`React + Vite`) on `http://127.0.0.1:5173`

The system is built for:

- spot only
- single symbol (`BTCUSDT`)
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

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_TESTNET=true` (recommended for development)

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
- `Trading`
  - current position
  - recent orders/fills
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

