import { useEffect, useMemo, useState } from "react";

import { Sparkline } from "../components/Sparkline";
import { useDashboard } from "../dashboard";
import { explainRiskReason, formatCountdown, formatDateTime, formatNumber, formatRelativeSeconds } from "../utils";

type ActivityTone = "ok" | "wait" | "warn" | "error";

function toneClass(tone: ActivityTone): string {
  if (tone === "ok") {
    return "border-emerald-300/40 bg-emerald-500/10 text-emerald-100";
  }
  if (tone === "warn") {
    return "border-amber-300/40 bg-amber-500/10 text-amber-100";
  }
  if (tone === "error") {
    return "border-rose-300/40 bg-rose-500/10 text-rose-100";
  }
  return "border-sky-300/40 bg-sky-500/10 text-sky-100";
}

export function HomePage() {
  const { data, loading } = useDashboard();
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [livePricePoints, setLivePricePoints] = useState<Array<{ timestamp: string; price: number }>>([]);

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const priceRaw = data.marketPrice?.price;
    const timestamp = data.marketPrice?.timestamp;
    const price = Number(priceRaw);
    if (!timestamp || !Number.isFinite(price)) {
      return;
    }
    setLivePricePoints((prev) => {
      const last = prev[prev.length - 1];
      if (last?.timestamp === timestamp) {
        return prev;
      }
      const next = [...prev, { timestamp, price }];
      return next.slice(-240);
    });
  }, [data.marketPrice?.price, data.marketPrice?.timestamp]);

  const equityValues = data.equity.map((point) => Number(point.equity));
  const latestEquity = data.equity.length > 0 ? data.equity[data.equity.length - 1]?.equity : null;
  const marketCandlesChrono = useMemo(() => [...data.marketCandles].reverse(), [data.marketCandles]);
  const hasLiveSeries = livePricePoints.length >= 2;
  const marketValues = hasLiveSeries
    ? livePricePoints.map((point) => point.price)
    : marketCandlesChrono.map((candle) => Number(candle.close));
  const latestMarketClose = hasLiveSeries
    ? String(livePricePoints[livePricePoints.length - 1]?.price ?? "")
    : marketCandlesChrono.length > 0
      ? marketCandlesChrono[marketCandlesChrono.length - 1]?.close
      : null;
  const latestMarketPrice = data.marketPrice?.price ?? latestMarketClose ?? null;
  const marketChangePct = useMemo(() => {
    if (marketValues.length < 2) {
      return null;
    }
    const first = Number(marketValues[0]);
    const last = Number(marketValues[marketValues.length - 1]);
    if (!Number.isFinite(first) || !Number.isFinite(last) || first === 0) {
      return null;
    }
    return ((last - first) / first) * 100;
  }, [marketValues]);
  const lastResult = data.scheduler?.last_result ?? data.lastCycle ?? null;
  const reconciliation = lastResult?.reconciliation ?? null;
  const gridPreview = data.gridPreview;
  const tradeSymbol = lastResult?.symbol ?? data.config?.trading_pair ?? "BTCUSDT";
  const baseAsset = tradeSymbol.endsWith("USDT") ? tradeSymbol.replace("USDT", "") : tradeSymbol;

  const nextRunInSeconds = useMemo(() => {
    if (!data.scheduler?.running || !data.scheduler.last_run_at) {
      return null;
    }
    const lastRunMs = new Date(data.scheduler.last_run_at).getTime();
    if (!Number.isFinite(lastRunMs)) {
      return null;
    }
    const nextRunAtMs = lastRunMs + data.scheduler.interval_seconds * 1000;
    return Math.max(0, Math.floor((nextRunAtMs - nowMs) / 1000));
  }, [data.scheduler, nowMs]);

  const activity = useMemo(() => {
    if (data.scheduler?.last_error) {
      return {
        tone: "error" as const,
        label: "ERROR",
        detail: data.scheduler.last_error,
      };
    }

    if (!data.systemReadiness?.can_trade) {
      return {
        tone: "warn" as const,
        label: "WAITING_CONDITION",
        detail: data.systemReadiness?.reasons.join(" | ") || "System is not in RUNNING state.",
      };
    }

    if (!data.systemReadiness?.data_ready && data.systemReadiness?.require_data_ready) {
      return {
        tone: "warn" as const,
        label: "WAITING_CONDITION",
        detail: data.systemReadiness.reasons.join(" | "),
      };
    }

    if (!data.scheduler?.running) {
      return {
        tone: "wait" as const,
        label: "WAITING_CONDITION",
        detail: "Scheduler is stopped. Start scheduler or run manual cycle to analyze market.",
      };
    }

    if (!lastResult) {
      return {
        tone: "wait" as const,
        label: "ANALYZING",
        detail: "Scheduler is running. Waiting for first analysis cycle.",
      };
    }

    if (lastResult.executed) {
      return {
        tone: "ok" as const,
        label: "ACTION_EXECUTED",
        detail: `${lastResult.signal_side} filled (${formatNumber(lastResult.quantity, 6)} ${baseAsset})`,
      };
    }

    if (reconciliation && !reconciliation.within_critical_tolerance) {
      return {
        tone: "error" as const,
        label: "RECONCILIATION_ALERT",
        detail: `Critical mismatch detected: diff ${formatNumber(reconciliation.difference, 4)}.`,
      };
    }

    return {
      tone: "wait" as const,
      label: "WAITING_CONDITION",
      detail: explainRiskReason(lastResult.risk_reason),
    };
  }, [baseAsset, data.scheduler, data.systemReadiness, lastResult, reconciliation]);

  const expectedAction = useMemo(() => {
    const activeStrategy = data.systemReadiness?.active_strategy ?? data.config?.active_strategy ?? "";
    if (!data.systemReadiness?.can_trade) {
      return {
        label: "Resume Required",
        detail: data.systemReadiness?.reasons.join(" | ") || "System is not in RUNNING state.",
      };
    }
    if (!data.scheduler?.running) {
      return {
        label: "Scheduler Stopped",
        detail: "Start scheduler to continue automatic analysis cycles.",
      };
    }
    if (!data.systemReadiness?.data_ready && data.systemReadiness?.require_data_ready) {
      return {
        label: "Warmup Pending",
        detail: data.systemReadiness.reasons.join(" | "),
      };
    }
    if (activeStrategy !== "smart_grid_ai") {
      if (lastResult && !lastResult.executed) {
        return {
          label: "Risk Hold",
          detail: explainRiskReason(lastResult.risk_reason),
        };
      }
      return {
        label: "Monitoring",
        detail: `${activeStrategy || "Active strategy"} is running without grid-projection telemetry.`,
      };
    }
    if (!gridPreview) {
      if (lastResult && !lastResult.executed) {
        return {
          label: "Analyzing",
          detail: explainRiskReason(lastResult.risk_reason),
        };
      }
      return {
        label: "Analyzing",
        detail: "Waiting for smart-grid projection data from API cycles.",
      };
    }
    if (gridPreview.bootstrap_eligible) {
      return {
        label: "Bootstrap BUY",
        detail: "Next cycle can seed initial paper inventory so SELL levels can execute.",
      };
    }

    const last = Number(gridPreview.last_price);
    const buyTrigger = Number(gridPreview.buy_trigger ?? NaN);
    const sellTrigger = Number(gridPreview.sell_trigger ?? NaN);

    if (Number.isFinite(last) && Number.isFinite(buyTrigger) && Number.isFinite(sellTrigger) && last > 0) {
      const buyDistanceBps = ((buyTrigger - last) / last) * 10000;
      const sellDistanceBps = ((sellTrigger - last) / last) * 10000;
      const nearestIsBuy = Math.abs(buyDistanceBps) <= Math.abs(sellDistanceBps);
      if (nearestIsBuy) {
        return {
          label: "Watch BUY Trigger",
          detail: `Waiting for price to reach ${formatNumber(buyTrigger, 2)} (${formatNumber(buyDistanceBps, 1)} bps).`,
        };
      }
      return {
        label: "Watch SELL Trigger",
        detail: `Waiting for price to reach ${formatNumber(sellTrigger, 2)} (${formatNumber(sellDistanceBps, 1)} bps).`,
      };
    }

    if (lastResult && !lastResult.executed) {
      return {
        label: "Risk Hold",
        detail: explainRiskReason(lastResult.risk_reason),
      };
    }
    return {
      label: "Monitoring",
      detail: "Analyzing market and waiting for next valid trigger.",
    };
  }, [data.config?.active_strategy, data.scheduler, data.systemReadiness, gridPreview, lastResult]);

  return (
    <section className="grid gap-4">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">Equity</p>
          <p className="mt-3 font-heading text-2xl font-bold text-white">{formatNumber(latestEquity, 2)} USDT</p>
          <p className="mt-2 text-xs text-slate-400">API: {data.health?.status ?? "unknown"}</p>
        </article>
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">Total PnL</p>
          <p className="mt-3 font-heading text-2xl font-bold text-white">
            {formatNumber(data.metrics?.total_pnl, 2)} USDT
          </p>
          <p className="mt-2 text-xs text-slate-400">Fees: {formatNumber(data.metrics?.total_fees, 2)} USDT</p>
        </article>
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">Trades</p>
          <p className="mt-3 font-heading text-2xl font-bold text-white">{data.metrics?.total_trades ?? 0}</p>
          <p className="mt-2 text-xs text-slate-400">
            Win Rate:{" "}
            {data.metrics?.win_rate !== null && data.metrics?.win_rate !== undefined
              ? formatNumber(data.metrics.win_rate * 100, 1)
              : "-"}
            %
          </p>
        </article>
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">Market Price</p>
          <p className="mt-3 font-heading text-2xl font-bold text-white">{formatNumber(latestMarketPrice, 2)} USDT</p>
          <p className="mt-2 text-xs text-slate-400">
            {tradeSymbol} 1h change: {marketChangePct !== null ? `${formatNumber(marketChangePct, 2)}%` : "-"}
          </p>
        </article>
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">Engine</p>
          <p className="mt-3 font-heading text-2xl font-bold text-white">
            {activity.label}
          </p>
          <p className="mt-2 text-xs text-slate-400">
            Last run {formatRelativeSeconds(data.scheduler?.last_run_at, nowMs)}
          </p>
        </article>
      </div>

      <article className={`rounded-2xl border p-4 shadow-panel backdrop-blur ${toneClass(activity.tone)}`}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="font-heading text-lg font-bold">Bot Activity</h2>
            <p className="mt-1 text-xs">{activity.detail}</p>
          </div>
          <span className="inline-flex items-center gap-2 rounded-full border border-white/20 px-3 py-1 text-[11px] uppercase tracking-[0.2em]">
            <span className="h-2 w-2 animate-pulse rounded-full bg-current" />
            Live
          </span>
        </div>
        <div className="mt-4 grid gap-2 text-xs md:grid-cols-2">
          <p>Scheduler: {data.scheduler?.running ? "running" : "stopped"}</p>
          <p>
            Next cycle:{" "}
            {data.scheduler?.running ? (nextRunInSeconds !== null ? `in ${formatCountdown(nextRunInSeconds)}` : "pending first cycle") : "-"}
          </p>
          <p>Signal: {lastResult?.signal_side ?? "-"}</p>
          <p>Risk: {lastResult?.risk_action ?? "-"} ({lastResult?.risk_reason ?? "-"})</p>
          <p>Reconciliation: {reconciliation ? reconciliation.reason : "-"}</p>
          <p>Reconciliation Diff: {reconciliation ? formatNumber(reconciliation.difference, 4) : "-"}</p>
          <p>Executed: {String(lastResult?.executed ?? false)}</p>
          <p>Decision at: {formatDateTime(data.scheduler?.last_result?.executed_at ?? data.scheduler?.last_run_at)}</p>
        </div>
        <div className="mt-4 rounded-lg border border-white/20 bg-black/10 p-3 text-xs">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">Expected Next Action</p>
          <p className="mt-2 inline-flex items-center gap-2 rounded-full border border-sky-300/40 bg-sky-500/10 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-sky-100">
            {expectedAction.label}
          </p>
          <p className="mt-2 text-slate-200">{expectedAction.detail}</p>
        </div>
      </article>

      <div className="grid gap-4 lg:grid-cols-[2fr_2fr_1fr]">
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-heading text-lg font-bold text-white">Equity Curve</h2>
            <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">{data.equity.length} points</p>
          </div>
          <Sparkline points={equityValues} emptyLabel="No equity snapshots yet." />
        </article>
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-heading text-lg font-bold text-white">
              {tradeSymbol} Live Curve ({hasLiveSeries ? "tick" : "1h"})
            </h2>
            <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">
              {hasLiveSeries ? `${livePricePoints.length} ticks` : `${marketCandlesChrono.length} candles`}
            </p>
          </div>
          <Sparkline points={marketValues} emptyLabel="No market candles yet. Fetch data first." />
          <p className="mt-3 text-xs text-slate-300">
            Last ticker update: {formatDateTime(data.marketPrice?.timestamp ?? null)} (refreshes every ~5s)
          </p>
        </article>
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <h2 className="font-heading text-lg font-bold text-white">System Snapshot</h2>
          <div className="mt-3 space-y-2 text-xs text-slate-200">
            <p>State: {data.system?.state ?? "-"}</p>
            <p>Strategy: {data.systemReadiness?.active_strategy ?? "-"}</p>
            <p>Data Ready: {String(data.systemReadiness?.data_ready ?? false)}</p>
            <p>Reason: {data.system?.reason ?? "-"}</p>
            <p>Changed At: {formatDateTime(data.system?.changed_at)}</p>
            <p>Scheduler: {data.scheduler?.running ? "running" : "stopped"}</p>
            <p>Last Run: {formatDateTime(data.scheduler?.last_run_at)}</p>
            <p>Last Error: {data.scheduler?.last_error ?? "-"}</p>
            <p>Recon Reason: {reconciliation?.reason ?? "-"}</p>
            <p>Recon Critical: {reconciliation ? String(reconciliation.within_critical_tolerance) : "-"}</p>
            <p>Warmup Reasons: {data.systemReadiness?.reasons.join(" | ") || "-"}</p>
          </div>
        </article>
      </div>

      {loading ? (
        <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-xs text-slate-300">Refreshing data...</div>
      ) : null}
    </section>
  );
}
