# Secrets Management Guide

This project already supports file-based secrets via `APP_SECRETS_DIR`.

Use this flow to avoid exposing credentials in git, docs, screenshots, or shell history.

## 1) Rules

- Never put real secrets in `.env.example`.
- Do not commit `.env`, `.env.*`, `.secrets/`, or `secrets/`.
- Keep `TRADING_LIVE_MODE=false` unless you explicitly intend live trading.
- Use separate secrets for `dev`, `staging`, and `prod`.

## 2) Recommended Local Setup (Windows PowerShell)

Create local secrets directory:

```powershell
New-Item -ItemType Directory -Path .\.secrets -Force | Out-Null
```

Create secret files (one file per env var):

```powershell
Set-Content .\.secrets\BINANCE_API_KEY "your_binance_api_key"
Set-Content .\.secrets\BINANCE_API_SECRET "your_binance_api_secret"
Set-Content .\.secrets\TELEGRAM_BOT_TOKEN "your_telegram_bot_token"
Set-Content .\.secrets\TELEGRAM_CHAT_ID "your_telegram_chat_id"
Set-Content .\.secrets\AUTH_SECRET_KEY "your_long_random_secret"
Set-Content .\.secrets\AUTH_ADMIN_PASSWORD "your_admin_password"
Set-Content .\.secrets\AUTH_OPERATOR_PASSWORD "your_operator_password"
Set-Content .\.secrets\AUTH_VIEWER_PASSWORD "your_viewer_password"
```

Set `APP_SECRETS_DIR` in your `.env`:

```dotenv
APP_SECRETS_DIR=./.secrets
```

How it works:

- The app loads values from files named exactly like env vars (for example `BINANCE_API_KEY`).
- File content is used as the secret value.
- You can still keep non-sensitive settings in `.env`.

## 3) Docker Compose Setup

`docker-compose.yml` is configured to mount:

- host `./.secrets`
- container `/run/secrets/trading-bot`

and sets:

- `APP_SECRETS_DIR=/run/secrets/trading-bot`

Start:

```bash
docker compose up --build -d
```

## 4) Rotation Procedure

1. Rotate secret at provider side (Binance, Telegram, etc.).
2. Update corresponding file in `.secrets/`.
3. Restart service:

```bash
docker compose up -d --force-recreate api
```

For auth token invalidation after key rotation:

- rotate `AUTH_SECRET_KEY`
- restart API

## 5) Prevent Accidental Exposure

- Use placeholders only in docs and examples.
- Avoid pasting tokens in terminal commands; prefer file-based secrets.
- Before commit, scan diff for sensitive names:

```bash
git diff -- . ':(exclude).secrets/*' | rg -n "API_KEY|API_SECRET|TOKEN|PASSWORD|SECRET"
```

## 6) If A Secret Was Exposed

1. Revoke/rotate it immediately at provider.
2. Replace local copy.
3. Remove from git history if it was committed.
4. Rotate dependent credentials/tokens.

