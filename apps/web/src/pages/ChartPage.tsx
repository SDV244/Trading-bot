import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type Fill, type MarketCandle, type Order } from "../api";
import { useDashboard } from "../dashboard";
import { formatDateTime, formatNumber } from "../utils";

type Marker = {
  id: number;
  side: "BUY" | "SELL" | null;
  x: number;
  y: number;
  price: number;
  filledAt: string;
};

const TIMEFRAME_OPTIONS = ["1m", "5m", "15m", "1h", "4h"] as const;
const POLL_MS = 5000;

function nearestCandleIndex(candles: MarketCandle[], targetMs: number): number {
  let bestIndex = 0;
  let bestDiff = Number.MAX_SAFE_INTEGER;
  candles.forEach((candle, index) => {
    const closeMs = new Date(candle.close_time).getTime();
    const diff = Math.abs(closeMs - targetMs);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestIndex = index;
    }
  });
  return bestIndex;
}

export function ChartPage() {
  const { data } = useDashboard();
  const symbol = data.config?.trading_pair ?? "BTCUSDT";
  const [timeframe, setTimeframe] = useState<string>("1h");
  const [candles, setCandles] = useState<MarketCandle[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [fills, setFills] = useState<Fill[]>([]);
  const [price, setPrice] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const [candlesResponse, ordersResponse, fillsResponse, priceResponse] = await Promise.all([
        api.getMarketCandles(symbol, timeframe, 240).catch(() => []),
        api.getOrders(200),
        api.getFills(200),
        api.getMarketPrice(symbol).catch(() => null),
      ]);
      setCandles([...candlesResponse].reverse());
      setOrders(ordersResponse);
      setFills(fillsResponse);
      setPrice(priceResponse?.price ?? null);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load chart data");
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      void refresh();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const chart = useMemo(() => {
    const width = 1000;
    const height = 360;
    const padX = 24;
    const padY = 20;
    const plotWidth = width - padX * 2;
    const plotHeight = height - padY * 2;

    const closeValues = candles.map((item) => Number(item.close)).filter((value) => Number.isFinite(value));
    if (closeValues.length < 2) {
      return {
        width,
        height,
        path: "",
        min: null,
        max: null,
        markers: [] as Marker[],
      };
    }

    const min = Math.min(...closeValues);
    const max = Math.max(...closeValues);
    const range = Math.max(max - min, 1e-9);

    const xForIndex = (index: number): number => padX + (index / (candles.length - 1)) * plotWidth;
    const yForPrice = (value: number): number => padY + ((max - value) / range) * plotHeight;

    const path = candles
      .map((item, index) => {
        const x = xForIndex(index);
        const y = yForPrice(Number(item.close));
        return `${x},${y}`;
      })
      .join(" ");

    const orderSideById = new Map<number, "BUY" | "SELL">();
    orders.forEach((order) => {
      if (order.side === "BUY" || order.side === "SELL") {
        orderSideById.set(order.id, order.side);
      }
    });

    const markers = fills
      .map((fill) => {
        const fillPrice = Number(fill.price);
        const fillMs = new Date(fill.filled_at).getTime();
        if (!Number.isFinite(fillPrice) || !Number.isFinite(fillMs) || candles.length === 0) {
          return null;
        }
        const candleIndex = nearestCandleIndex(candles, fillMs);
        return {
          id: fill.id,
          side: orderSideById.get(fill.order_id) ?? null,
          x: xForIndex(candleIndex),
          y: yForPrice(fillPrice),
          price: fillPrice,
          filledAt: fill.filled_at,
        } satisfies Marker;
      })
      .filter((item): item is Marker => item !== null);

    return { width, height, path, min, max, markers };
  }, [candles, fills, orders]);

  const latestClose = candles.length > 0 ? candles[candles.length - 1]?.close : null;
  const recentFills = useMemo(() => fills.slice(0, 12), [fills]);

  return (
    <section className="grid gap-4">
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-heading text-lg font-bold text-white">{symbol} Market Chart</h2>
            <p className="mt-1 text-xs text-slate-300">
              Live ticker: {formatNumber(price ?? latestClose, 2)} USDT
              {" | "}
              Timeframe: {timeframe}
              {" | "}
              Candles: {candles.length}
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <label htmlFor="timeframe" className="text-slate-300">Timeframe</label>
            <select
              id="timeframe"
              className="rounded-md border border-white/20 bg-white/10 px-2 py-1 text-slate-100"
              value={timeframe}
              onChange={(event) => setTimeframe(event.target.value)}
            >
              {TIMEFRAME_OPTIONS.map((option) => (
                <option key={option} value={option} className="bg-slate-900">
                  {option}
                </option>
              ))}
            </select>
            <button
              className="rounded-md border border-cyan-300/60 bg-cyan-500/10 px-2 py-1 text-cyan-100 hover:bg-cyan-500/20"
              onClick={() => void refresh()}
            >
              Refresh
            </button>
          </div>
        </div>

        {chart.path ? (
          <div className="mt-4 rounded-xl border border-white/10 bg-black/20 p-2">
            <svg viewBox={`0 0 ${chart.width} ${chart.height}`} className="h-72 w-full">
              {[0, 1, 2, 3, 4].map((line) => {
                const y = 20 + (line / 4) * (chart.height - 40);
                return (
                  <line
                    key={line}
                    x1={24}
                    x2={chart.width - 24}
                    y1={y}
                    y2={y}
                    stroke="rgba(148,163,184,0.2)"
                    strokeDasharray="4 6"
                  />
                );
              })}
              <polyline
                fill="none"
                stroke="url(#marketLineGradient)"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2.4"
                points={chart.path}
              />
              {chart.markers.map((marker) => (
                <g key={marker.id} transform={`translate(${marker.x},${marker.y})`}>
                  {marker.side === "BUY" ? (
                    <path d="M 0 -8 L 7 6 L -7 6 Z" fill="#34d399" />
                  ) : marker.side === "SELL" ? (
                    <path d="M 0 8 L 7 -6 L -7 -6 Z" fill="#fb7185" />
                  ) : (
                    <circle r="4" fill="#fde68a" />
                  )}
                  <title>
                    {`${marker.side ?? "UNKNOWN"} fill | ${formatDateTime(marker.filledAt)} | ${formatNumber(marker.price, 2)} USDT`}
                  </title>
                </g>
              ))}
              <defs>
                <linearGradient id="marketLineGradient" x1="0%" x2="100%" y1="0%" y2="0%">
                  <stop offset="0%" stopColor="#60a5fa" />
                  <stop offset="100%" stopColor="#22d3ee" />
                </linearGradient>
              </defs>
            </svg>
          </div>
        ) : (
          <div className="mt-4 rounded-xl border border-white/10 bg-black/20 p-4 text-xs text-slate-300">
            No candle data in cache for {timeframe}. Fetch/stream this timeframe first, or switch to `1h`/`4h`.
          </div>
        )}

        <div className="mt-3 flex flex-wrap gap-3 text-xs">
          <span className="rounded-full border border-emerald-300/50 bg-emerald-500/10 px-3 py-1 text-emerald-100">
            BUY marker: green triangle
          </span>
          <span className="rounded-full border border-rose-300/50 bg-rose-500/10 px-3 py-1 text-rose-100">
            SELL marker: red triangle
          </span>
          <span className="rounded-full border border-amber-300/50 bg-amber-500/10 px-3 py-1 text-amber-100">
            Unknown side: yellow dot
          </span>
          {chart.min !== null && chart.max !== null ? (
            <span className="rounded-full border border-slate-300/40 bg-slate-500/10 px-3 py-1 text-slate-200">
              Range: {formatNumber(chart.min, 2)} - {formatNumber(chart.max, 2)} USDT
            </span>
          ) : null}
        </div>
      </article>

      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h3 className="font-heading text-lg font-bold text-white">Recent Fills (for markers)</h3>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-slate-400">
                <th className="pb-2">Fill</th>
                <th className="pb-2">Order</th>
                <th className="pb-2">Time</th>
                <th className="pb-2">Qty</th>
                <th className="pb-2">Price</th>
              </tr>
            </thead>
            <tbody>
              {recentFills.map((fill) => (
                <tr key={fill.id} className="border-t border-white/5 text-slate-200">
                  <td className="py-2">{fill.id}</td>
                  <td className="py-2">{fill.order_id}</td>
                  <td className="py-2">{formatDateTime(fill.filled_at)}</td>
                  <td className="py-2">{formatNumber(fill.quantity, 6)}</td>
                  <td className="py-2">{formatNumber(fill.price, 2)}</td>
                </tr>
              ))}
              {recentFills.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-3 text-slate-400">No fills available yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </article>

      {loading ? (
        <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-xs text-slate-300">
          Refreshing chart...
        </div>
      ) : null}
      {error ? (
        <div className="rounded-xl border border-rose-300/40 bg-rose-500/10 p-4 text-xs text-rose-200">
          {error}
        </div>
      ) : null}
    </section>
  );
}

