import { useMemo } from "react";

import { useDashboard } from "../dashboard";
import { explainRiskReason, formatDateTime, formatNumber, formatTime } from "../utils";

export function TradingPage() {
  const { data } = useDashboard();
  const lastResult = data.scheduler?.last_result ?? data.lastCycle ?? null;
  const symbol = data.position?.symbol ?? data.config?.trading_pair ?? "BTCUSDT";
  const baseAsset = symbol.endsWith("USDT") ? symbol.replace("USDT", "") : symbol;
  const idleDetail = useMemo(() => {
    if (data.scheduler?.last_error) {
      return `Scheduler error: ${data.scheduler.last_error}`;
    }
    if (!data.scheduler?.running) {
      return "Scheduler is stopped. Start scheduler to keep analyzing market continuously.";
    }
    if (!data.systemReadiness?.can_trade) {
      return data.systemReadiness?.reasons.join(" | ") || "System is not RUNNING.";
    }
    if (lastResult && !lastResult.executed) {
      return explainRiskReason(lastResult.risk_reason);
    }
    return "Bot is active and evaluating every cycle.";
  }, [data.scheduler?.last_error, data.scheduler?.running, data.systemReadiness, lastResult]);
  const recentDecisionEvents = useMemo(
    () => data.events.filter((event) => event.event_category === "trade" || event.event_category === "risk").slice(0, 6),
    [data.events],
  );

  return (
    <section className="grid gap-4">
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Current Position</h2>
        <div className="mt-3 grid gap-2 text-xs text-slate-200 md:grid-cols-3">
          <p>Symbol: {symbol}</p>
          <p>Side: {data.position?.side ?? "flat"}</p>
          <p>Qty: {formatNumber(data.position?.quantity, 6)} {baseAsset}</p>
          <p>Avg Entry: {formatNumber(data.position?.avg_entry_price, 2)}</p>
          <p>Unrealized PnL: {formatNumber(data.position?.unrealized_pnl, 2)}</p>
          <p>Realized PnL: {formatNumber(data.position?.realized_pnl, 2)}</p>
        </div>
      </article>

      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Cycle Decision Feed</h2>
        <div className="mt-3 grid gap-2 text-xs text-slate-200 md:grid-cols-3">
          <p>Last Signal: {lastResult?.signal_side ?? "-"}</p>
          <p>Signal Reason: {lastResult?.signal_reason ?? "-"}</p>
          <p>Risk Action: {lastResult?.risk_action ?? "-"}</p>
          <p>Risk Reason: {lastResult?.risk_reason ?? "-"}</p>
          <p>Executed: {String(lastResult?.executed ?? false)}</p>
          <p>Decision Time: {formatDateTime(data.scheduler?.last_result?.executed_at ?? data.scheduler?.last_run_at)}</p>
        </div>
        <p className="mt-3 text-xs text-slate-300">{idleDetail}</p>
        {lastResult && !lastResult.executed ? (
          <div className="mt-3 rounded-lg border border-amber-300/30 bg-amber-500/10 p-3 text-xs text-amber-100">
            Waiting reason: {lastResult.risk_reason} ({lastResult.signal_reason})
          </div>
        ) : null}
      </article>

      <div className="grid gap-4 lg:grid-cols-2">
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <h2 className="font-heading text-lg font-bold text-white">Recent Orders</h2>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-slate-400">
                  <th className="pb-2">ID</th>
                  <th className="pb-2">Time</th>
                  <th className="pb-2">Side</th>
                  <th className="pb-2">Qty</th>
                  <th className="pb-2">Price</th>
                  <th className="pb-2">Notional</th>
                  <th className="pb-2">Strategy</th>
                  <th className="pb-2">Signal</th>
                  <th className="pb-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {data.orders.map((order) => (
                  <tr key={order.id} className="border-t border-white/5 text-slate-200">
                    <td className="py-2">{order.id}</td>
                    <td className="py-2">{formatTime(order.created_at)}</td>
                    <td className={order.side === "BUY" ? "py-2 text-mint" : "py-2 text-ember"}>{order.side}</td>
                    <td className="py-2">{formatNumber(order.quantity, 5)}</td>
                    <td className="py-2">{formatNumber(order.price, 2)}</td>
                    <td className="py-2">{formatNumber(Number(order.quantity) * Number(order.price ?? 0), 2)}</td>
                    <td className="py-2">{order.strategy_name}</td>
                    <td className="py-2">{order.signal_reason ?? "-"}</td>
                    <td className="py-2">{order.status}</td>
                  </tr>
                ))}
                {data.orders.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="py-3 text-slate-400">
                      No orders yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </article>

        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <h2 className="font-heading text-lg font-bold text-white">Recent Fills</h2>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-slate-400">
                  <th className="pb-2">ID</th>
                  <th className="pb-2">Time</th>
                  <th className="pb-2">Order</th>
                  <th className="pb-2">Qty</th>
                  <th className="pb-2">Price</th>
                  <th className="pb-2">Notional</th>
                  <th className="pb-2">Fee</th>
                  <th className="pb-2">Slippage</th>
                </tr>
              </thead>
              <tbody>
                {data.fills.map((fill) => (
                  <tr key={fill.id} className="border-t border-white/5 text-slate-200">
                    <td className="py-2">{fill.id}</td>
                    <td className="py-2">{formatTime(fill.filled_at)}</td>
                    <td className="py-2">{fill.order_id}</td>
                    <td className="py-2">{formatNumber(fill.quantity, 5)}</td>
                    <td className="py-2">{formatNumber(fill.price, 2)}</td>
                    <td className="py-2">{formatNumber(Number(fill.quantity) * Number(fill.price), 2)}</td>
                    <td className="py-2">{formatNumber(fill.fee, 6)}</td>
                    <td className="py-2">{fill.slippage_bps !== null ? `${formatNumber(fill.slippage_bps, 2)} bps` : "-"}</td>
                  </tr>
                ))}
                {data.fills.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-3 text-slate-400">
                      No fills yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </article>
      </div>

      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Recent Risk/Trade Events</h2>
        <div className="mt-3 space-y-2 text-xs text-slate-200">
          {recentDecisionEvents.map((event) => (
            <div key={event.id} className="rounded-lg border border-white/10 bg-black/10 p-2">
              <p className="text-slate-100">{event.summary}</p>
              <p className="mt-1 text-slate-400">
                {event.event_category}/{event.event_type} at {formatDateTime(event.created_at)} by {event.actor}
              </p>
            </div>
          ))}
          {recentDecisionEvents.length === 0 ? <p className="text-slate-400">No risk/trade events yet.</p> : null}
        </div>
      </article>
    </section>
  );
}
