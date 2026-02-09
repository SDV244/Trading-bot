# Operations Runbook

This runbook covers common operational tasks and incident response actions.

## 1) Service Lifecycle

Start:

```bash
docker compose up --build -d
```

Status:

```bash
docker compose ps
```

Logs:

```bash
docker compose logs --tail=200 api
docker compose logs --tail=200 web
```

Stop:

```bash
docker compose down
```

## 2) Health Checks

- API liveness: `GET /health`
- API readiness: `GET /ready`
- Dashboard: open `http://127.0.0.1:5173`

## 3) Routine Checks (Daily)

1. Confirm system state is not `EMERGENCY_STOP`.
2. Confirm scheduler status and last run error is empty.
3. Review pending approvals + expiry countdown.
4. Review logs filtered by `category=system` and `category=trade`.
5. Verify no unexpected config changes in audit events.

## 4) Emergency Procedures

Immediate stop:

- UI: `Controls` -> `Emergency Stop`
- API: `POST /api/system/emergency-stop`

After emergency:

1. inspect `events_log` timeline
2. identify root cause (data issue, risk breach, adapter error, operator action)
3. apply remediation
4. resume only with authorized role/path (manual resume policy)

## 5) Approval Queue Timeout Handling

- Trigger manual expiry scan:
  - `POST /api/ai/approvals/expire-check`
- Validate all pending items have intentional owners and decisions.

## 6) Authentication Operations

When `AUTH_ENABLED=true`:

- rotate credentials by changing env vars and restarting API
- rotate `AUTH_SECRET_KEY` to invalidate all existing tokens
- keep admin credentials restricted

## 7) Database Operations

Migration:

```bash
poetry run alembic upgrade head
```

Backup (SQLite file):

- copy `./data/trading.db` while services are stopped for consistent backup

## 8) Build/Test Gate Before Deploy

```bash
poetry run pytest -q
poetry run mypy packages apps/api
cd apps/web && npm run lint && npm run typecheck && npm run build
```

## 9) Troubleshooting

If API fails startup:

- check `docker compose logs api`
- verify `.env` keys are valid
- verify DB path is writable

If web is blank or unauthorized:

- confirm `VITE_API_BASE_URL` points to API
- if auth enabled, verify login credentials
- inspect browser console + API `/api/auth/me`

If approvals/actions fail with `403`:

- role is insufficient for requested operation
- verify assigned account role and token freshness

