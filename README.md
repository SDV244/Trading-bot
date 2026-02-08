# Trading Bot 🤖

AI-powered Binance BTCUSDT trading bot with continuous learning, paper trading validation, and comprehensive monitoring.

## Features

- **Spot Trading Only** - BTCUSDT on Binance
- **AI-Driven Optimization** - Deep RL (PPO) suggests strategy improvements
- **Paper Trading First** - Strict validation before live trading
- **Safety First** - 2h approval timeout, emergency stops, full audit trail
- **Web Dashboard** - Real-time monitoring and control
- **Telegram Alerts** - Trade summaries and critical notifications

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Poetry
- Binance API keys (Spot trading only, NO withdrawals)
- Telegram bot token

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Trading-bot.git
cd Trading-bot

# Install Python dependencies
poetry install

# Install frontend dependencies
cd apps/web && npm install && cd ../..

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
# Start backend API
poetry run uvicorn apps.api.main:app --reload

# Start frontend (separate terminal)
cd apps/web && npm run dev

# Run tests
poetry run pytest
```

## Project Structure

```
Trading-bot/
├── apps/
│   ├── api/          # FastAPI backend
│   └── web/          # React dashboard
├── packages/
│   ├── core/         # Business logic
│   ├── adapters/     # External integrations
│   └── research/     # Backtesting
├── tests/
└── data/             # SQLite database (gitignored)
```

## Safety Rules

> ⚠️ **Non-Negotiable**
>
> - Spot only, never futures
> - AI proposes, human approves
> - No auto-resume after emergency stop
> - Full audit trail for every action

## License

Private - All rights reserved
