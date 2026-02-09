# Operator Quickstart (One Page)

This is the fastest safe workflow for daily operations.

## 1) Start Services

1. Ensure local secrets are present (`.secrets/`) and mounted via `APP_SECRETS_DIR`.
2. Start stack:

```bash
docker compose up --build -d
docker compose ps
```

Open:

- Web: `http://127.0.0.1:5173`
- API docs: `http://127.0.0.1:8000/docs`

## 2) Login and Verify

1. Sign in (if `AUTH_ENABLED=true`).
2. Check `Home`:
   - API health = healthy
   - readiness = true
   - system state is expected (`running` or `paused`)

## 3) Daily Checklist

1. `Trading` page:
   - check position
   - inspect recent orders/fills
2. `AI Approvals` page:
   - review pending proposals
   - watch countdown timers
   - approve/reject only with evidence
3. `Logs` page:
   - filter by `category=system`
   - review emergency/config changes
4. `Controls` page:
   - use only when necessary

## 4) Safe Control Actions

- Pause:
  - use for maintenance or uncertainty
- Resume:
  - only after validation
- Emergency Stop:
  - use immediately on unsafe behavior
  - no auto-resume policy

## 5) Incident Procedure (Fast)

1. Trigger emergency stop.
2. Open `Logs` and filter:
   - actor
   - category
   - search terms (`emergency`, `risk`, `approval`)
3. Capture:
   - timestamp
   - event IDs
   - summary
4. Escalate with findings.

## 6) Common API Commands

Health:

```bash
curl -s http://127.0.0.1:8000/health
```

State:

```bash
curl -s http://127.0.0.1:8000/api/system/state -H "Authorization: Bearer <token>"
```

Pending approvals:

```bash
curl -s "http://127.0.0.1:8000/api/ai/approvals?status=PENDING" -H "Authorization: Bearer <token>"
```

## 7) End of Day

1. Confirm no unresolved pending approvals.
2. Confirm no active incident.
3. Optionally archive logs/DB snapshot for audit retention.
