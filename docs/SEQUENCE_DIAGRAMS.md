# Sequence Diagrams

This file contains key runtime sequences in Mermaid format.

## 1) Paper Trading Cycle (Scheduler Driven)

```mermaid
sequenceDiagram
    autonumber
    participant Sch as Scheduler
    participant Cycle as TradingCycleService
    participant Strat as Strategy Engine
    participant Risk as Risk Engine
    participant Exec as Paper Engine
    participant DB as SQLite
    participant Audit as Audit Logger

    Sch->>Cycle: run_once()
    Cycle->>Strat: generate_signal(candles)
    Strat-->>Cycle: Signal(side, confidence, reason, indicators)
    Cycle->>Risk: evaluate(signal, portfolio_state)
    Risk-->>Cycle: RiskDecision(action, size, reason)

    alt decision = EXECUTE
        Cycle->>Exec: execute(order_request)
        Exec-->>Cycle: fill details
        Cycle->>DB: persist orders/fills/position
        Cycle->>DB: persist equity + metrics snapshots
        Cycle->>Audit: log trade/system events
    else decision = HOLD/BLOCK
        Cycle->>Audit: log risk block/hold reason
    end
```

## 2) AI Proposal and Human Approval

```mermaid
sequenceDiagram
    autonumber
    participant Ops as Operator (UI/Telegram)
    participant API as FastAPI
    participant Advisor as AI Advisor
    participant Gate as Approval Gate
    participant DB as SQLite
    participant State as State Manager

    Ops->>API: POST /api/ai/proposals/generate
    API->>Advisor: generate_proposals()
    Advisor-->>API: list[AIProposal]
    API->>Gate: create_approval(proposal)
    Gate->>DB: insert approval(status=PENDING, expires_at)
    API-->>Ops: pending approvals

    Ops->>API: POST /api/ai/approvals/{id}/approve or reject
    API->>Gate: approve/reject
    Gate->>DB: update status + decided_by + decided_at
    API-->>Ops: updated approval

    Note over API,Gate: periodic expiry check
    API->>Gate: expire_pending()
    Gate->>DB: set EXPIRED where pending and ttl elapsed
    Gate->>State: trigger emergency policy path when required
```

## 3) Auth + RBAC Protected Request

```mermaid
sequenceDiagram
    autonumber
    participant User as User
    participant Web as Web App
    participant API as FastAPI
    participant Auth as Auth Security
    participant Route as Protected Route Handler

    User->>Web: submit username/password
    Web->>API: POST /api/auth/login
    API->>Auth: validate credentials
    Auth-->>API: token + role + expiry
    API-->>Web: bearer token
    Web->>Web: store token

    Web->>API: GET/POST protected endpoint (Authorization: Bearer)
    API->>Auth: verify signature, expiry, role
    alt role allowed
        Auth-->>Route: AuthUser
        Route-->>Web: success response
    else role denied
        Auth-->>Web: 403 Forbidden
    end
```

## 4) Emergency Stop Path

```mermaid
sequenceDiagram
    autonumber
    participant Ops as Operator
    participant API as FastAPI
    participant State as State Manager
    participant Sch as Scheduler
    participant Audit as Audit Logger
    participant Tg as Telegram

    Ops->>API: POST /api/system/emergency-stop
    API->>State: force_emergency_stop(reason, actor)
    State-->>API: state = EMERGENCY_STOP
    API->>Sch: prevent trading execution paths
    API->>Audit: append emergency event
    API->>Tg: critical alert
    API-->>Ops: current state + can_trade=false
```

