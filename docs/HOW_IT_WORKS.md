# How It Works

This document explains the runtime flow end-to-end.

## 1) High-Level Flow

1. Market data is fetched from Binance adapters.
2. Strategy produces a signal (`BUY` / `SELL` / `HOLD`).
3. Risk engine validates the signal and computes allowed sizing.
4. Execution engine performs paper (or gated live) order execution.
5. Database persists orders, fills, position updates, equity snapshots, metrics.
6. Audit logger records what happened, why, inputs, and actor.
7. AI advisor/optimizer proposes parameter or strategy changes.
8. Approval gate enforces human decision and expiry timeout behavior.

## 2) Runtime Components

- API layer (`apps/api`)
  - HTTP contract, validation, route authorization
- Core layer (`packages/core`)
  - trading cycle orchestration, risk, execution, metrics, AI, scheduler, state
- Adapter layer (`packages/adapters`)
  - Binance + Telegram integrations
- Storage
  - SQLite via SQLAlchemy models/repositories
- Frontend (`apps/web`)
  - local operational control plane

## 3) Trading Cycle Details

`packages/core/trading_cycle.py` runs one cycle:

1. load recent candles for configured symbol/timeframes
2. run active strategy
3. call risk engine to decide `ALLOW`/`BLOCK` and size
4. if allowed, send order request to paper engine
5. persist outcomes (`orders`, `fills`, `positions`)
6. compute and store metrics/equity snapshots
7. append audit event(s)

Before scheduler start, the system evaluates strategy candle requirements.
If `TRADING_REQUIRE_DATA_READY=true` and warmup is incomplete, scheduler start is blocked.
Paper risk sizing/equity baselines use `TRADING_PAPER_STARTING_EQUITY`.
When `TRADING_ACTIVE_STRATEGY=smart_grid_ai`, the cycle uses adaptive grid spacing
based on ATR/volatility plus 4h regime tilt.
Optional controls for smart-grid operations:
- `TRADING_GRID_SPACING_MODE=geometric|arithmetic`
- `TRADING_GRID_COOLDOWN_SECONDS` to avoid overtrading bursts
- `TRADING_GRID_AUTO_INVENTORY_BOOTSTRAP=true` to seed initial spot inventory when flat + SELL signal appears

## 4) State Machine

Defined in `packages/core/state.py`.

States:

- `RUNNING`
- `PAUSED`
- `EMERGENCY_STOP`

Rules:

- trading only allowed while `RUNNING`
- `EMERGENCY_STOP` requires explicit manual resume path
- transitions are audited and exposed via system endpoints

## 5) AI Proposal + Approval Flow

1. Advisor (`packages/core/ai/advisor.py`) builds context from current data/metrics and, when enabled,
   asks the configured LLM provider (`openai|anthropic|gemini|ollama`) for proposals.
   - LLM output is schema-filtered and key-whitelisted before entering approvals.
   - if configured fallback is enabled, built-in rule proposals are used when LLM output is empty/invalid.
   - for `smart_grid_ai`, advisor can emit grid-specific tuning proposals
     (`grid_levels`, spacing bounds, volatility blend) based on win-rate/drawdown.
2. Proposal is saved as `PENDING` in `approvals`.
3. Operator approves/rejects through API/web.
4. Expiry checker marks timed-out proposals as `EXPIRED`.
5. Expiry can trigger safety behavior (including emergency stop policy paths).
6. All steps are logged in `events_log`.
7. Scheduler invokes advisor every `TRADING_ADVISOR_INTERVAL_CYCLES` cycles.
8. Scheduler can emit Telegram heartbeat notifications every `TELEGRAM_HEARTBEAT_HOURS`.

## 6) Security Model

When `AUTH_ENABLED=true`:

- token-based login via `/api/auth/login`
- RBAC enforced per route:
  - `viewer`: read-only pages/endpoints
  - `operator`: operational actions (pause/resume/scheduler/approvals)
  - `admin`: highest privilege (manual resume after emergency, live order endpoint)

When `AUTH_ENABLED=false`:

- system behaves as local trusted mode for development

## 7) Data Model Summary

Core tables:

- `candles_cache`
- `orders`
- `fills`
- `positions`
- `equity_snapshots`
- `metrics_snapshots`
- `approvals`
- `events_log`
- `config`

Each table has targeted indexes for common operational queries.
