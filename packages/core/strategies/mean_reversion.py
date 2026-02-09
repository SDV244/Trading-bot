"""Mean reversion strategy using z-score and RSI guard."""

from dataclasses import dataclass
from decimal import Decimal
from statistics import mean, pstdev

from packages.core.strategies.base import Signal, Strategy, StrategyContext, registry


def _rsi(closes: list[Decimal], period: int = 14) -> Decimal:
    if len(closes) < period + 1:
        return Decimal("50")
    gains: list[Decimal] = []
    losses: list[Decimal] = []
    for i in range(1, period + 1):
        delta = closes[-i] - closes[-i - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(abs(delta))
    avg_gain = sum(gains) / Decimal(period)
    avg_loss = sum(losses) / Decimal(period)
    if avg_loss == 0:
        return Decimal("100")
    rs = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


@dataclass(slots=True)
class MeanReversionStrategy(Strategy):
    """1h z-score mean reversion with RSI/Bollinger-like guard."""

    name: str = "mean_reversion_zscore"
    lookback: int = 30
    z_threshold: float = 1.5
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    def generate_signal(self, context: StrategyContext) -> Signal:
        closes = [float(c.close) for c in context.candles_1h]
        if len(closes) < self.lookback + 1:
            return Signal(side="HOLD", confidence=0.0, reason="insufficient_data", indicators={})

        window = closes[-self.lookback :]
        mu = mean(window)
        sigma = pstdev(window) or 1e-9
        last = closes[-1]
        z = (last - mu) / sigma

        close_decimals = [Decimal(str(c.close)) for c in context.candles_1h]
        rsi = float(_rsi(close_decimals))

        indicators = {"zscore": z, "rsi": rsi, "mean": mu, "std": sigma}
        if z <= -self.z_threshold and rsi <= self.rsi_oversold:
            confidence = min(0.95, abs(z) / (self.z_threshold * 2))
            return Signal(side="BUY", confidence=confidence, reason="oversold_reversion", indicators=indicators)
        if z >= self.z_threshold and rsi >= self.rsi_overbought:
            confidence = min(0.95, abs(z) / (self.z_threshold * 2))
            return Signal(side="SELL", confidence=confidence, reason="overbought_reversion", indicators=indicators)
        return Signal(side="HOLD", confidence=0.2, reason="no_reversion_edge", indicators=indicators)


registry.register("mean_reversion_zscore", MeanReversionStrategy)
