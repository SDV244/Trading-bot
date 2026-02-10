"""Order book analytics used by execution/risk/AI context."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(slots=True, frozen=True)
class OrderBookSnapshot:
    """Computed order book features."""

    symbol: str
    best_bid: Decimal
    best_ask: Decimal
    spread_bps: float
    bid_depth_10: Decimal
    ask_depth_10: Decimal
    imbalance: float
    liquidity_score: float
    market_impact_1btc_bps: float

    def to_dict(self) -> dict[str, float]:
        """Serialize snapshot as float dict for API/LLM context."""
        return {
            "best_bid": float(self.best_bid),
            "best_ask": float(self.best_ask),
            "spread_bps": self.spread_bps,
            "bid_depth_10": float(self.bid_depth_10),
            "ask_depth_10": float(self.ask_depth_10),
            "imbalance": self.imbalance,
            "liquidity_score": self.liquidity_score,
            "market_impact_1btc_bps": self.market_impact_1btc_bps,
        }


class OrderBookAnalyzer:
    """Compute microstructure features from Binance depth payload."""

    @staticmethod
    def from_binance_depth(symbol: str, payload: dict[str, Any]) -> OrderBookSnapshot | None:
        bids_raw = payload.get("bids", [])
        asks_raw = payload.get("asks", [])
        if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
            return None
        bids = OrderBookAnalyzer._parse_levels(bids_raw)
        asks = OrderBookAnalyzer._parse_levels(asks_raw)
        if not bids or not asks:
            return None
        return OrderBookAnalyzer.analyze(symbol=symbol, bids=bids, asks=asks)

    @staticmethod
    def analyze(
        *,
        symbol: str,
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
    ) -> OrderBookSnapshot:
        best_bid = max(price for price, _qty in bids)
        best_ask = min(price for price, _qty in asks)
        midpoint = (best_bid + best_ask) / Decimal("2")
        spread_bps = 0.0
        if midpoint > 0:
            spread_bps = float(((best_ask - best_bid) / midpoint) * Decimal("10000"))

        top_bids = sorted(bids, key=lambda item: item[0], reverse=True)[:10]
        top_asks = sorted(asks, key=lambda item: item[0])[:10]
        bid_depth = sum((price * qty for price, qty in top_bids), start=Decimal("0"))
        ask_depth = sum((price * qty for price, qty in top_asks), start=Decimal("0"))
        imbalance = float((bid_depth - ask_depth) / (bid_depth + ask_depth + Decimal("0.00000001")))

        total_depth = bid_depth + ask_depth
        # Heuristic liquidity score on 0-10 scale (10 => deep and tight spread).
        depth_component = min(1.0, float(total_depth / Decimal("2000000")))
        spread_component = max(0.0, min(1.0, 1.0 - (spread_bps / 20.0)))
        liquidity_score = (depth_component * 6.0) + (spread_component * 4.0)

        impact_notional = Decimal("1") * midpoint
        one_sided_depth = max(Decimal("0.00000001"), min(bid_depth, ask_depth))
        market_impact_1btc = float((impact_notional / one_sided_depth) * Decimal("10000"))

        return OrderBookSnapshot(
            symbol=symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=max(0.0, spread_bps),
            bid_depth_10=bid_depth,
            ask_depth_10=ask_depth,
            imbalance=max(-1.0, min(1.0, imbalance)),
            liquidity_score=max(0.0, min(10.0, liquidity_score)),
            market_impact_1btc_bps=max(0.0, market_impact_1btc),
        )

    @staticmethod
    def _parse_levels(levels: list[Any]) -> list[tuple[Decimal, Decimal]]:
        parsed: list[tuple[Decimal, Decimal]] = []
        for item in levels:
            if not isinstance(item, list) or len(item) < 2:
                continue
            try:
                price = Decimal(str(item[0]))
                qty = Decimal(str(item[1]))
            except Exception:
                continue
            if price <= 0 or qty <= 0:
                continue
            parsed.append((price, qty))
        return parsed
