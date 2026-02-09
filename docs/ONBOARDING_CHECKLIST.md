# Onboarding Checklist

This checklist helps new developers and operators start safely and consistently.

## A) New Developer Checklist

## 1) Access + Tools

- [ ] Clone repository to local machine
- [ ] Install Python `3.11+`
- [ ] Install Poetry
- [ ] Install Node `20+`
- [ ] Install Docker Desktop

## 2) Project Setup

- [ ] Run `poetry install`
- [ ] Run `cd apps/web && npm install`
- [ ] Create `.env` from `.env.example`
- [ ] Set Binance testnet credentials
- [ ] Set `TRADING_LIVE_MODE=false`
- [ ] Set auth values for local RBAC testing (optional but recommended)

## 3) Verify Build + Tests

- [ ] `poetry run pytest -q`
- [ ] `poetry run mypy packages apps/api`
- [ ] `cd apps/web && npm run lint && npm run typecheck && npm run build`
- [ ] `docker compose up --build -d`
- [ ] Confirm API `/health` and Web dashboard load

## 4) Code Understanding

- [ ] Read `docs/USER_GUIDE.md`
- [ ] Read `docs/HOW_IT_WORKS.md`
- [ ] Read `docs/CODE_REFERENCE.md`
- [ ] Review `apps/api/main.py` and route files
- [ ] Review `packages/core/trading_cycle.py` and `packages/core/state.py`

## B) New Operator Checklist

## 1) Operational Access

- [ ] Receive operator credentials
- [ ] Confirm environment is paper mode for training (`TRADING_LIVE_MODE=false`)
- [ ] Log in to dashboard if auth enabled

## 2) Daily Operations

- [ ] Check `Home` for health/readiness/system state
- [ ] Check `Trading` for position and recent executions
- [ ] Review `AI Approvals` pending queue + countdowns
- [ ] Review `Logs` with filters for system/trade events
- [ ] Only use `Controls` when required and documented

## 3) Incident Actions

- [ ] Trigger emergency stop on unsafe behavior
- [ ] Capture audit event trail
- [ ] Escalate with event IDs and timestamps
- [ ] Do not resume until root cause is understood

## C) PR / Change Checklist (Engineering)

- [ ] Safety rules preserved (spot only, single symbol, no auto-resume)
- [ ] RBAC enforced for new protected endpoints
- [ ] Audit events added for new critical actions
- [ ] Unit tests added/updated
- [ ] Docs updated (`docs/*` + `README.md` if needed)
- [ ] Docker build and local smoke checks pass

## D) Production Readiness Gate

- [ ] Auth enabled with strong secrets
- [ ] Role credentials rotated and stored securely
- [ ] Testnet vs production keys validated
- [ ] Alerts/notifications tested (Telegram if enabled)
- [ ] Backup/restore path tested for SQLite
- [ ] Incident runbook reviewed by operators

