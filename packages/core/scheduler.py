"""Background scheduler for periodic paper trading cycles."""

import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from packages.core.ai import get_ai_advisor, get_approval_gate
from packages.core.database.session import get_session
from packages.core.trading_cycle import CycleResult, get_trading_cycle_service


@dataclass(slots=True)
class SchedulerStatus:
    """Current scheduler status."""

    running: bool
    interval_seconds: int
    last_run_at: datetime | None
    last_error: str | None
    last_result: dict[str, Any] | None


class TradingScheduler:
    """Simple async loop scheduler for paper cycle execution."""

    def __init__(self, interval_seconds: int = 60, advisor_interval_cycles: int = 30) -> None:
        self.interval_seconds = interval_seconds
        self.advisor_interval_cycles = max(1, advisor_interval_cycles)
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_run_at: datetime | None = None
        self._last_error: str | None = None
        self._last_result: dict[str, Any] | None = None
        self._cycles = 0

    @property
    def running(self) -> bool:
        """Whether scheduler is currently active."""
        return self._running

    def status(self) -> SchedulerStatus:
        """Get scheduler status snapshot."""
        return SchedulerStatus(
            running=self._running,
            interval_seconds=self.interval_seconds,
            last_run_at=self._last_run_at,
            last_error=self._last_error,
            last_result=self._last_result,
        )

    def start(self, interval_seconds: int | None = None) -> bool:
        """
        Start scheduler loop.

        Returns True if started, False if already running.
        """
        if self._running:
            return False
        if interval_seconds is not None:
            self.interval_seconds = max(5, interval_seconds)

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Trading scheduler started (interval={self.interval_seconds}s)")
        return True

    async def stop(self) -> bool:
        """
        Stop scheduler loop.

        Returns True if stopped, False if it was not running.
        """
        if not self._running:
            return False

        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

        logger.info("Trading scheduler stopped")
        return True

    async def _run_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.interval_seconds)
            if not self._running:
                break

            self._last_run_at = datetime.now(UTC)
            try:
                async with get_session() as session:
                    await get_approval_gate().expire_pending(session)
                    result = await get_trading_cycle_service().run_once(session)
                    self._cycles += 1

                    if self._cycles % self.advisor_interval_cycles == 0:
                        pending = await get_approval_gate().list_approvals(
                            session, status="PENDING", limit=1
                        )
                        if not pending:
                            proposals = await get_ai_advisor().generate_proposals(session)
                            if proposals:
                                await get_approval_gate().create_approval(session, proposals[0])

                self._last_result = self._serialize_result(result)
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                logger.exception("Trading scheduler cycle failed")

    @staticmethod
    def _serialize_result(result: CycleResult) -> dict[str, Any]:
        payload = asdict(result)
        payload["executed_at"] = datetime.now(UTC).isoformat()
        return payload


_trading_scheduler: TradingScheduler | None = None


def get_trading_scheduler() -> TradingScheduler:
    """Get or create trading scheduler singleton."""
    global _trading_scheduler
    if _trading_scheduler is None:
        _trading_scheduler = TradingScheduler()
    return _trading_scheduler


async def close_trading_scheduler() -> None:
    """Stop scheduler if running."""
    global _trading_scheduler
    if _trading_scheduler is not None:
        await _trading_scheduler.stop()
        _trading_scheduler = None
