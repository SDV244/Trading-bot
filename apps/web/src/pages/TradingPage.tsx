import { useMemo } from "react";

import { useDashboard } from "../dashboard";
import { explainRiskReason, formatDateTime, formatNumber, formatTime } from "../utils";

export function TradingPage() {
  const { data } = useDashboard();
  const lastResult = data.scheduler?.last_result ?? data.lastCycle ?? null;
  const reconciliation = lastResult?.reconciliation ?? null;
  const gridPreview = data.gridPreview;
  const symbol = data.position?.symbol ?? data.config?.trading_pair ?? "BTCUSDT";
  const baseAsset = data.funds?.base_asset ?? (symbol.endsWith("USDT") ? symbol.replace("USDT", "") : symbol);
  const quoteAsset = data.funds?.quote_asset ?? "USDT";
  const nextGridHint = useMemo(() => {
    if (!gridPreview) {
      return null;
    }
    const nearestBuy = [...gridPreview.buy_levels].sort((a, b) => Math.abs(a.distance_bps) - Math.abs(b.distance_bps))[0];
    const nearestSell = [...gridPreview.sell_levels].sort((a, b) => Math.abs(a.distance_bps) - Math.abs(b.distance_bps))[0];
    if (!nearestBuy && !nearestSell) {
      return "Grid levels are not available yet.";
    }
    if (!nearestBuy) {
      return `Nearest sell level: ${formatNumber(nearestSell?.price, 2)} (${formatNumber(nearestSell?.distance_bps, 1)} bps).`;
    }
    if (!nearestSell) {
      return `Nearest buy level: ${formatNumber(nearestBuy.price, 2)} (${formatNumber(nearestBuy.distance_bps, 1)} bps).`;
    }
    return `Nearest buy ${formatNumber(nearestBuy.price, 2)} (${formatNumber(nearestBuy.distance_bps, 1)} bps), nearest sell ${formatNumber(nearestSell.price, 2)} (${formatNumber(nearestSell.distance_bps, 1)} bps).`;
  }, [gridPreview]);
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
    if (gridPreview && gridPreview.bootstrap_eligible) {
      return "Smart-grid is flat and bootstrap is eligible. The next cycle can seed initial paper inventory.";
    }
    if (lastResult && !lastResult.executed) {
      return explainRiskReason(lastResult.risk_reason);
    }
    return "Bot is active and evaluating every cycle.";
  }, [data.scheduler?.last_error, data.scheduler?.running, data.systemReadiness, gridPreview, lastResult]);
  const recentDecisionEvents = useMemo(
    () => data.events.filter((event) => event.event_category === "trade" || event.event_category === "risk").slice(0, 6),
    [data.events],
  );

  return (
    <section className="grid gap-4">
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Wallet Funds</h2>
        {data.funds ? (
          <div className="mt-3 grid gap-2 text-xs text-slate-200 md:grid-cols-3">
            <p>
              {baseAsset}: {formatNumber(data.funds.base_balance, 6)}
            </p>
            <p>
              {quoteAsset}: {formatNumber(data.funds.quote_balance, 2)}
            </p>
            <p>
              {baseAsset} Value ({quoteAsset}): {formatNumber(data.funds.base_quote_value, 2)}
            </p>
            <p>
              Est. Total Equity: {formatNumber(data.funds.estimated_total_equity, 2)} {quoteAsset}
            </p>
            <p>
              Mark Price: {formatNumber(data.funds.mark_price, 2)} {quoteAsset}
            </p>
            <p>Source: {data.funds.source}</p>
          </div>
        ) : (
          <p className="mt-3 text-xs text-slate-400">
            Funds are not available yet. Ensure API is running and, in live mode, exchange credentials are valid.
          </p>
        )}
      </article>

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
          <p>Recon Reason: {reconciliation?.reason ?? "-"}</p>
          <p>Recon Diff: {reconciliation ? formatNumber(reconciliation.difference, 4) : "-"}</p>
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

      {gridPreview ? (
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <h2 className="font-heading text-lg font-bold text-white">Smart Grid Projection</h2>
          <div className="mt-3 grid gap-2 text-xs text-slate-200 md:grid-cols-4">
            <p>Price: {formatNumber(gridPreview.last_price, 2)}</p>
            <p>Center: {formatNumber(gridPreview.grid_center, 2)}</p>
            <p>Band Low: {formatNumber(gridPreview.grid_lower, 2)}</p>
            <p>Band High: {formatNumber(gridPreview.grid_upper, 2)}</p>
            <p>Buy Trigger: {formatNumber(gridPreview.buy_trigger, 2)}</p>
            <p>Sell Trigger: {formatNumber(gridPreview.sell_trigger, 2)}</p>
            <p>Grid Step: {formatNumber(gridPreview.grid_step, 2)}</p>
            <p>Spacing: {gridPreview.spacing_bps !== null ? `${formatNumber(gridPreview.spacing_bps, 1)} bps` : "-"}</p>
            <p>Signal: {gridPreview.signal_side}</p>
            <p>Reason: {gridPreview.signal_reason}</p>
            <p>Recentered: {String(gridPreview.recentered)}</p>
            <p>Mode: {gridPreview.recenter_mode}</p>
            <p>Position Qty: {formatNumber(gridPreview.position_quantity, 6)} {baseAsset}</p>
            <p>Bootstrap Eligible: {String(gridPreview.bootstrap_eligible)}</p>
            <p>TP Trigger: {formatNumber(gridPreview.take_profit_trigger, 2)}</p>
            <p>SL Trigger: {formatNumber(gridPreview.stop_loss_trigger, 2)}</p>
          </div>
          <p className="mt-3 text-xs text-slate-300">{nextGridHint ?? "Grid hint unavailable."}</p>
          <div className="mt-3 grid gap-4 lg:grid-cols-2">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-slate-400">
                    <th className="pb-2">BUY Lvl</th>
                    <th className="pb-2">Price</th>
                    <th className="pb-2">Distance</th>
                  </tr>
                </thead>
                <tbody>
                  {gridPreview.buy_levels.map((level) => (
                    <tr key={`buy-${level.level}`} className="border-t border-white/5 text-slate-200">
                      <td className="py-2">#{level.level}</td>
                      <td className="py-2">{formatNumber(level.price, 2)}</td>
                      <td className="py-2">{formatNumber(level.distance_bps, 1)} bps</td>
                    </tr>
                  ))}
                  {gridPreview.buy_levels.length === 0 ? (
                    <tr>
                      <td colSpan={3} className="py-3 text-slate-400">
                        No buy levels available.
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-slate-400">
                    <th className="pb-2">SELL Lvl</th>
                    <th className="pb-2">Price</th>
                    <th className="pb-2">Distance</th>
                  </tr>
                </thead>
                <tbody>
                  {gridPreview.sell_levels.map((level) => (
                    <tr key={`sell-${level.level}`} className="border-t border-white/5 text-slate-200">
                      <td className="py-2">#{level.level}</td>
                      <td className="py-2">{formatNumber(level.price, 2)}</td>
                      <td className="py-2">{formatNumber(level.distance_bps, 1)} bps</td>
                    </tr>
                  ))}
                  {gridPreview.sell_levels.length === 0 ? (
                    <tr>
                      <td colSpan={3} className="py-3 text-slate-400">
                        No sell levels available.
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>
        </article>
      ) : null}

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
