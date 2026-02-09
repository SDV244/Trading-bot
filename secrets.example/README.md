# Local Secrets Template

Create a local `.secrets/` directory (ignored by git) and copy these file names:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `AUTH_SECRET_KEY`
- `AUTH_ADMIN_PASSWORD`
- `AUTH_OPERATOR_PASSWORD`
- `AUTH_VIEWER_PASSWORD`

Each file should contain only the secret value.

Then set:

```dotenv
APP_SECRETS_DIR=./.secrets
```

Do not commit `.secrets/`.
