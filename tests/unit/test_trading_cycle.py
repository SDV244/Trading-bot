"""Tests for trading cycle orchestration."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import packages.core.state as state_module
import packages.core.trading_cycle as cycle_module
from packages.core.config import reload_settings
from packages.core.database.models import (
    Base,
    Candle,
    EquitySnapshot,
    Fill,
    MetricsSnapshot,
    Order,
    Position,
)
from packages.core.strategies.base import Signal
from packages.core.trading_cycle import TradingCycleService


@pytest.fixture
async def db_session() -> AsyncSession:
    """Create in-memory database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Reset mutable singletons used by the trading loop."""
    state_module._state_manager = state_module.StateManager()  # type: ignore[attr-defined]
    cycle_module._trading_cycle_service = None  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure strategy-related settings are deterministic."""
    monkeypatch.setenv("TRADING_LIVE_MODE", "false")
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "trend_ema")
    monkeypatch.setenv("TRADING_REQUIRE_DATA_READY", "true")
    monkeypatch.setenv("TRADING_SPOT_POSITION_MODE", "long_flat")
    monkeypatch.setenv("TRADING_PAPER_STARTING_EQUITY", "10000")
    monkeypatch.setenv("TRADING_GRID_COOLDOWN_SECONDS", "0")
    reload_settings()


async def _seed_candles(
    session: AsyncSession,
    timeframe: str,
    count: int,
    start_price: Decimal,
    step: Decimal,
) -> None:
    hours = int(timeframe[:-1]) if timeframe.endswith("h") else 1
    interval = timedelta(hours=hours)
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - (interval * count)
    rows = []
    for i in range(count):
        open_time = start + (interval * i)
        close = start_price + (step * Decimal(i))
        rows.append(
            Candle(
                symbol="BTCUSDT",
                timeframe=timeframe,
                open_time=open_time,
                close_time=open_time + interval - timedelta(milliseconds=1),
                open=close - Decimal("5"),
                high=close + Decimal("10"),
                low=close - Decimal("10"),
                close=close,
                volume=Decimal("100"),
                quote_volume=Decimal("100000"),
                trades_count=100,
            )
        )
    session.add_all(rows)
    await session.flush()


@pytest.mark.asyncio
async def test_cycle_skips_when_system_not_running(db_session: AsyncSession) -> None:
    await _seed_candles(db_session, "1h", 80, Decimal("50000"), Decimal("8"))
    await _seed_candles(db_session, "4h", 80, Decimal("48000"), Decimal("40"))

    service = TradingCycleService()
    result = await service.run_once(db_session)

    assert result.executed is False
    assert result.risk_reason == "system_not_running"
    orders = (await db_session.execute(select(func.count(Order.id)))).scalar_one()
    assert orders == 0


@pytest.mark.asyncio
async def test_cycle_executes_and_persists_trade(db_session: AsyncSession) -> None:
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    result = await service.run_once(db_session)

    assert result.executed is True
    assert result.order_id is not None
    assert result.fill_id is not None

    order_count = (await db_session.execute(select(func.count(Order.id)))).scalar_one()
    fill_count = (await db_session.execute(select(func.count(Fill.id)))).scalar_one()
    position_count = (await db_session.execute(select(func.count(Position.id)))).scalar_one()
    equity_count = (await db_session.execute(select(func.count(EquitySnapshot.id)))).scalar_one()
    metrics_count = (await db_session.execute(select(func.count(MetricsSnapshot.id)))).scalar_one()

    assert order_count == 1
    assert fill_count == 1
    assert position_count == 1
    assert equity_count == 1
    assert metrics_count == 1


@pytest.mark.asyncio
async def test_long_flat_mode_blocks_repeated_buy_entries(db_session: AsyncSession) -> None:
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    first = await service.run_once(db_session)
    second = await service.run_once(db_session)

    assert first.executed is True
    assert second.executed is False
    assert second.risk_reason == "already_in_position"


@pytest.mark.asyncio
async def test_hold_cycle_persists_equity_and_metrics_snapshots(db_session: AsyncSession) -> None:
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "HoldStrategy",
        (),
        {
            "name": "hold_test",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="HOLD", confidence=0.0, reason="hold_test", indicators={})
            ),
        },
    )()

    result = await service.run_once(db_session)

    assert result.executed is False
    assert result.risk_reason == "signal_hold"
    equity_count = (await db_session.execute(select(func.count(EquitySnapshot.id)))).scalar_one()
    metrics_count = (await db_session.execute(select(func.count(MetricsSnapshot.id)))).scalar_one()
    assert equity_count == 1
    assert metrics_count == 1


@pytest.mark.asyncio
async def test_close_open_paper_position_executes_force_sell(db_session: AsyncSession) -> None:
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "BuyStrategy",
        (),
        {
            "name": "buy_test",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="BUY", confidence=0.9, reason="force_buy", indicators={})
            ),
        },
    )()
    first = await service.run_once(db_session)
    assert first.executed is True

    closed = await service.close_open_paper_position(
        db_session,
        reason="test_close_all",
        actor="test",
    )
    assert closed.executed is True
    assert closed.signal_side == "SELL"
    assert closed.risk_action == "FORCE_CLOSE"
    assert closed.risk_reason == "test_close_all"

    position_result = await db_session.execute(select(Position).where(Position.symbol == "BTCUSDT"))
    position = position_result.scalar_one()
    assert Decimal(str(position.quantity)) == Decimal("0")


@pytest.mark.asyncio
async def test_close_open_paper_position_noop_when_flat(db_session: AsyncSession) -> None:
    service = TradingCycleService()
    result = await service.close_open_paper_position(
        db_session,
        reason="test_close_all",
        actor="test",
    )
    assert result.executed is False
    assert result.risk_reason == "no_inventory_to_sell"


@pytest.mark.asyncio
async def test_global_stop_loss_triggers_emergency_stop(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_STOP_LOSS_ENABLED", "true")
    monkeypatch.setenv("TRADING_STOP_LOSS_GLOBAL_EQUITY_PCT", "0.10")
    monkeypatch.setenv("TRADING_STOP_LOSS_AUTO_CLOSE_POSITIONS", "false")
    reload_settings()
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    db_session.add(
        Position(
            symbol="BTCUSDT",
            side=None,
            quantity=Decimal("0"),
            avg_entry_price=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("-1500"),
            total_fees=Decimal("0"),
            is_paper=True,
        )
    )
    await db_session.flush()
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    result = await service.run_once(db_session)

    assert result.executed is False
    assert result.risk_reason == "stop_loss_global_equity_triggered"
    assert state_module.get_state_manager().state.value == "emergency_stop"


@pytest.mark.asyncio
async def test_live_global_stop_loss_uses_exchange_equity_snapshot(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_LIVE_MODE", "true")
    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    monkeypatch.setenv("TRADING_STOP_LOSS_ENABLED", "true")
    monkeypatch.setenv("TRADING_STOP_LOSS_GLOBAL_EQUITY_PCT", "0.10")
    monkeypatch.setenv("TRADING_STOP_LOSS_MAX_DRAWDOWN_PCT", "0.20")
    monkeypatch.setenv("TRADING_PAPER_STARTING_EQUITY", "1000")
    reload_settings()

    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    position = await service._get_or_create_position(db_session)  # noqa: SLF001
    mark_price = Decimal("50000")

    mock_live_adapter = AsyncMock()
    mock_live_adapter.get_account_balances.return_value = {
        "BTC": Decimal("1.0"),
        "USDT": Decimal("10000"),
    }
    mock_spot_adapter = AsyncMock()
    mock_spot_adapter.get_ticker_price.return_value = Decimal("70000")

    with patch("packages.adapters.binance_live.get_binance_live_adapter", return_value=mock_live_adapter), patch(
        "packages.adapters.binance_spot.get_binance_adapter", return_value=mock_spot_adapter
    ):
        result = await service._check_global_stop_loss(  # noqa: SLF001
            session=db_session,
            position=position,
            mark_price=mark_price,
        )

    assert result is None
    assert state_module.get_state_manager().state.value == "running"


@pytest.mark.asyncio
async def test_global_stop_loss_ignores_legacy_peak_from_previous_paper_baseline(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_STOP_LOSS_ENABLED", "true")
    monkeypatch.setenv("TRADING_STOP_LOSS_MAX_DRAWDOWN_PCT", "0.20")
    monkeypatch.setenv("TRADING_PAPER_STARTING_EQUITY", "1000")
    monkeypatch.setenv("TRADING_SPOT_POSITION_MODE", "incremental")
    reload_settings()

    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))

    # Legacy snapshots from an older run with higher baseline.
    db_session.add_all(
        [
            EquitySnapshot(
                equity=Decimal("9996.87543657"),
                available_balance=Decimal("9996.87543657"),
                unrealized_pnl=Decimal("0"),
                is_paper=True,
            ),
            EquitySnapshot(
                equity=Decimal("9990.00000000"),
                available_balance=Decimal("9990.00000000"),
                unrealized_pnl=Decimal("0"),
                is_paper=True,
            ),
            # New baseline campaign around 1,000 USDT should anchor peak scope.
            EquitySnapshot(
                equity=Decimal("1000.00000000"),
                available_balance=Decimal("1000.00000000"),
                unrealized_pnl=Decimal("0"),
                is_paper=True,
            ),
        ]
    )
    db_session.add(
        Position(
            symbol="BTCUSDT",
            side=None,
            quantity=Decimal("0"),
            avg_entry_price=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("-3.12456343"),
            total_fees=Decimal("0"),
            is_paper=True,
        )
    )
    await db_session.flush()
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "HoldStrategy",
        (),
        {
            "name": "hold_test",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="HOLD", confidence=0.0, reason="hold_test", indicators={})
            ),
        },
    )()

    result = await service.run_once(db_session)
    assert result.risk_reason != "stop_loss_drawdown_triggered"
    assert state_module.get_state_manager().state.value == "running"
    metrics_result = await db_session.execute(
        select(MetricsSnapshot).where(MetricsSnapshot.is_paper.is_(True)).order_by(MetricsSnapshot.id.desc()).limit(1)
    )
    latest_metrics = metrics_result.scalar_one()
    assert latest_metrics.max_drawdown < 0.2


@pytest.mark.asyncio
async def test_smart_grid_bootstraps_inventory_on_sell_when_flat(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_AUTO_INVENTORY_BOOTSTRAP", "true")
    reload_settings()
    await _seed_candles(db_session, "1h", 140, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 80, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "SellStrategy",
        (),
        {
            "name": "smart_grid_ai",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="SELL", confidence=0.8, reason="grid_sell_rebalance", indicators={})
            ),
        },
    )()

    result = await service.run_once(db_session)
    assert result.executed is True
    assert result.signal_side == "BUY"
    assert result.signal_reason == "grid_inventory_bootstrap"
    assert result.risk_reason == "grid_inventory_bootstrap"


@pytest.mark.asyncio
async def test_smart_grid_does_not_bootstrap_on_recenter_breakdown_sell(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_AUTO_INVENTORY_BOOTSTRAP", "true")
    reload_settings()
    await _seed_candles(db_session, "1h", 140, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 80, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "BreakdownSellStrategy",
        (),
        {
            "name": "smart_grid_ai",
            "generate_signal": staticmethod(
                lambda _context: Signal(
                    side="SELL",
                    confidence=0.8,
                    reason="grid_recentered_auto_breakdown_sell",
                    indicators={},
                )
            ),
        },
    )()

    result = await service.run_once(db_session)
    assert result.executed is False
    assert result.signal_side == "SELL"
    assert result.risk_reason == "no_inventory_to_sell"


@pytest.mark.asyncio
async def test_smart_grid_bootstraps_inventory_on_hold_when_initially_flat(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_AUTO_INVENTORY_BOOTSTRAP", "true")
    reload_settings()
    await _seed_candles(db_session, "1h", 140, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 80, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "HoldGridStrategy",
        (),
        {
            "name": "smart_grid_ai",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="HOLD", confidence=0.4, reason="grid_wait_inside_band", indicators={})
            ),
        },
    )()

    result = await service.run_once(db_session)
    assert result.executed is True
    assert result.signal_side == "BUY"
    assert result.signal_reason == "grid_inventory_bootstrap"
    assert result.risk_reason == "grid_inventory_bootstrap"


@pytest.mark.asyncio
async def test_smart_grid_does_not_bootstrap_on_hold_after_trade_history_exists(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_AUTO_INVENTORY_BOOTSTRAP", "true")
    reload_settings()
    await _seed_candles(db_session, "1h", 140, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 80, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    order = Order(
        client_order_id="paper_history_seed",
        exchange_order_id=None,
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=Decimal("0.001"),
        price=Decimal("50000"),
        status="FILLED",
        is_paper=True,
        strategy_name="smart_grid_ai",
        signal_reason="history_seed",
        config_version=1,
    )
    db_session.add(order)
    await db_session.flush()
    db_session.add(
        Fill(
            order_id=order.id,
            fill_id="paper_history_seed_fill",
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            fee=Decimal("0"),
            fee_asset="USDT",
            is_paper=True,
            slippage_bps=0.0,
        )
    )
    await db_session.flush()

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "HoldGridStrategy",
        (),
        {
            "name": "smart_grid_ai",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="HOLD", confidence=0.4, reason="grid_wait_inside_band", indicators={})
            ),
        },
    )()

    result = await service.run_once(db_session)
    assert result.executed is False
    assert result.signal_side == "HOLD"
    assert result.risk_reason == "signal_hold"


@pytest.mark.asyncio
async def test_smart_grid_bootstrap_respects_min_notional_with_fraction(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_AUTO_INVENTORY_BOOTSTRAP", "true")
    monkeypatch.setenv("TRADING_GRID_BOOTSTRAP_FRACTION", "0.6")
    monkeypatch.setenv("TRADING_PAPER_STARTING_EQUITY", "1000")
    monkeypatch.setenv("RISK_PER_TRADE", "0.015")
    reload_settings()
    await _seed_candles(db_session, "1h", 140, Decimal("70000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 80, Decimal("65000"), Decimal("30"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "HoldGridStrategy",
        (),
        {
            "name": "smart_grid_ai",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="HOLD", confidence=0.4, reason="grid_recentered_auto", indicators={})
            ),
        },
    )()

    result = await service.run_once(db_session)
    assert result.executed is True
    assert result.signal_side == "BUY"
    assert result.signal_reason == "grid_inventory_bootstrap"

    fill_result = await db_session.execute(select(Fill).order_by(Fill.id.desc()).limit(1))
    fill = fill_result.scalar_one()
    assert Decimal(str(fill.quantity)) * Decimal(str(fill.price)) >= Decimal("10")


@pytest.mark.asyncio
async def test_cooldown_blocks_back_to_back_trades(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_SPOT_POSITION_MODE", "incremental")
    monkeypatch.setenv("TRADING_GRID_COOLDOWN_SECONDS", "3600")
    reload_settings()
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "BuyStrategy",
        (),
        {
            "name": "cooldown_test",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="BUY", confidence=0.9, reason="force_buy", indicators={})
            ),
        },
    )()

    first = await service.run_once(db_session)
    second = await service.run_once(db_session)
    assert first.executed is True
    assert second.executed is False
    assert second.risk_reason.startswith("cooldown_active_")


@pytest.mark.asyncio
async def test_fee_floor_blocks_readiness_when_enabled(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_ENFORCE_FEE_FLOOR", "true")
    monkeypatch.setenv("TRADING_GRID_MIN_NET_PROFIT_BPS", "35")
    monkeypatch.setenv("TRADING_GRID_MIN_SPACING_BPS", "25")
    monkeypatch.setenv("RISK_FEE_BPS", "10")
    monkeypatch.setenv("RISK_SLIPPAGE_BPS", "5")
    reload_settings()
    await _seed_candles(db_session, "1h", 180, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))

    service = TradingCycleService()
    readiness = await service.get_data_readiness(db_session)
    assert readiness.data_ready is False
    assert any("fee_floor_not_met" in reason for reason in readiness.reasons)


@pytest.mark.asyncio
async def test_out_of_bounds_emits_alert_once_within_cooldown(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_OUT_OF_BOUNDS_ALERT_COOLDOWN_MINUTES", "60")
    reload_settings()
    await _seed_candles(db_session, "1h", 180, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "OutOfBoundsStrategy",
        (),
        {
            "name": "smart_grid_ai",
            "generate_signal": staticmethod(
                lambda _context: Signal(
                    side="HOLD",
                    confidence=0.2,
                    reason="grid_recenter_wait",
                    indicators={"grid_lower": 49000.0, "grid_upper": 51000.0},
                )
            ),
        },
    )()

    class _MockNotifier:
        enabled = True

        def __init__(self) -> None:
            self.calls = 0

        async def send_critical_alert(self, _title: str, _body: str) -> bool:
            self.calls += 1
            return True

    notifier = _MockNotifier()
    monkeypatch.setattr("packages.core.trading_cycle.get_telegram_notifier", lambda: notifier)

    first = await service.run_once(db_session)
    second = await service.run_once(db_session)
    assert first.executed is False
    assert second.executed is False
    assert notifier.calls == 1


@pytest.mark.asyncio
async def test_data_readiness_reports_missing_4h_warmup(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "trend_ema")
    reload_settings()
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 20, Decimal("45000"), Decimal("50"))

    service = TradingCycleService()
    readiness = await service.get_data_readiness(db_session)

    assert readiness.active_strategy == "trend_ema"
    assert readiness.data_ready is False
    assert readiness.timeframes["4h"].required == 50
    assert readiness.timeframes["4h"].available == 20
    assert readiness.timeframes["4h"].ready is False


@pytest.mark.asyncio
async def test_strategy_can_be_selected_from_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "mean_reversion_zscore")
    reload_settings()
    service = TradingCycleService()

    assert service.strategy.name == "mean_reversion_zscore"
    assert service.get_strategy_requirements()["1h"] == 31
    assert service.get_strategy_requirements()["4h"] == 0


@pytest.mark.asyncio
async def test_fast_trend_strategy_can_be_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "trend_ema_fast")
    reload_settings()
    service = TradingCycleService()

    assert service.strategy.name == "trend_ema_fast"
    assert service.get_strategy_requirements()["1h"] == 13
    assert service.get_strategy_requirements()["4h"] == 34


@pytest.mark.asyncio
async def test_smart_grid_strategy_can_be_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_LOOKBACK_1H", "140")
    monkeypatch.setenv("TRADING_GRID_ATR_PERIOD_1H", "18")
    monkeypatch.setenv("TRADING_GRID_RECENTER_MODE", "conservative")
    reload_settings()
    service = TradingCycleService()

    assert service.strategy.name == "smart_grid_ai"
    assert service.strategy.recenter_mode == "conservative"
    assert service.get_strategy_requirements()["1h"] == 140
    assert service.get_strategy_requirements()["4h"] == 55


def test_set_grid_recenter_mode_updates_active_smart_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    monkeypatch.setenv("TRADING_GRID_RECENTER_MODE", "aggressive")
    reload_settings()
    service = TradingCycleService()

    mode = service.set_grid_recenter_mode("conservative")
    assert mode == "conservative"
    assert service.settings.trading.grid_recenter_mode == "conservative"
    assert service.strategy.recenter_mode == "conservative"


def test_paper_engine_symbol_allowlist_tracks_configured_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_PAIR", "ETHUSDT")
    reload_settings()
    service = TradingCycleService()

    assert service.symbol == "ETHUSDT"
    assert service.paper_engine.allowed_symbols == {"ETHUSDT"}


def test_invalid_strategy_name_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "does_not_exist")
    reload_settings()
    with pytest.raises(ValueError, match="Invalid TRADING_ACTIVE_STRATEGY"):
        TradingCycleService()


def test_strategy_requirements_map_to_configured_timeframes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_TIMEFRAMES", "1h,8h")
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "trend_ema")
    reload_settings()
    service = TradingCycleService()

    requirements = service.get_strategy_requirements()
    assert requirements["1h"] == 20
    assert requirements["8h"] == 50
    assert "4h" not in requirements


def test_paper_starting_equity_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_PAPER_STARTING_EQUITY", "7500")
    reload_settings()
    service = TradingCycleService()
    assert service.starting_equity == Decimal("7500")


@pytest.mark.asyncio
async def test_daily_realized_pnl_blocks_when_max_daily_loss_exceeded(
    db_session: AsyncSession,
) -> None:
    await _seed_candles(db_session, "1h", 120, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 120, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    yesterday = datetime.now(UTC) - timedelta(days=1)
    today = datetime.now(UTC) - timedelta(minutes=1)

    buy_order = Order(
        client_order_id="seed_buy_daily_loss",
        exchange_order_id=None,
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=Decimal("0.10000000"),
        price=Decimal("50000"),
        status="FILLED",
        is_paper=True,
        strategy_name="seed",
        signal_reason="seed",
        config_version=1,
    )
    db_session.add(buy_order)
    await db_session.flush()
    db_session.add(
        Fill(
            order_id=buy_order.id,
            fill_id="seed_fill_buy_daily_loss",
            quantity=Decimal("0.10000000"),
            price=Decimal("50000"),
            fee=Decimal("0"),
            fee_asset="USDT",
            is_paper=True,
            slippage_bps=0.0,
            filled_at=yesterday,
        )
    )

    sell_order = Order(
        client_order_id="seed_sell_daily_loss",
        exchange_order_id=None,
        symbol="BTCUSDT",
        side="SELL",
        order_type="MARKET",
        quantity=Decimal("0.10000000"),
        price=Decimal("47000"),
        status="FILLED",
        is_paper=True,
        strategy_name="seed",
        signal_reason="seed",
        config_version=1,
    )
    db_session.add(sell_order)
    await db_session.flush()
    db_session.add(
        Fill(
            order_id=sell_order.id,
            fill_id="seed_fill_sell_daily_loss",
            quantity=Decimal("0.10000000"),
            price=Decimal("47000"),
            fee=Decimal("0"),
            fee_asset="USDT",
            is_paper=True,
            slippage_bps=0.0,
            filled_at=today,
        )
    )
    await db_session.flush()

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "BuyStrategy",
        (),
        {
            "name": "buy_test",
            "generate_signal": staticmethod(
                lambda _context: Signal(side="BUY", confidence=0.9, reason="force_buy", indicators={})
            ),
        },
    )()

    result = await service.run_once(db_session)
    assert result.executed is False
    assert result.risk_reason == "max_daily_loss_exceeded"


@pytest.mark.asyncio
async def test_smart_grid_prefers_limit_order_fill_when_trigger_is_touched(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "smart_grid_ai")
    reload_settings()
    await _seed_candles(db_session, "1h", 140, Decimal("50000"), Decimal("10"))
    await _seed_candles(db_session, "4h", 80, Decimal("45000"), Decimal("50"))
    state_module.get_state_manager().resume("run cycle", "test")

    service = TradingCycleService()
    service.strategy = type(  # type: ignore[method-assign]
        "SmartGridBuyStrategy",
        (),
        {
            "name": "smart_grid_ai",
            "generate_signal": staticmethod(
                lambda context: Signal(
                    side="BUY",
                    confidence=0.8,
                    reason="grid_buy_rebalance",
                    indicators={
                        "buy_trigger": float(context.candles_1h[-1].close),
                        "sell_trigger": float(context.candles_1h[-1].close + Decimal("100")),
                    },
                )
            ),
        },
    )()

    result = await service.run_once(db_session)
    assert result.executed is True

    order_result = await db_session.execute(select(Order).order_by(Order.id.desc()).limit(1))
    order = order_result.scalar_one()
    assert order.order_type == "LIMIT"
