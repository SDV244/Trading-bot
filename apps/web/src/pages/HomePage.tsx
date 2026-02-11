import { useEffect, useMemo, useState } from "react";

import { ActivityFeed } from "../components/ActivityFeed";
import { DualSeriesChart } from "../components/DualSeriesChart";
import { GridBandView } from "../components/GridBandView";
import { MetricCard } from "../components/MetricCard";
import { Panel } from "../components/Panel";
import { StatusCard } from "../components/StatusCard";
import { useDashboard } from "../dashboard";
import {
  explainRiskReason,
  formatCountdown,
  formatDateTime,
  formatNumber,
  formatRelativeSeconds,
} from "../utils";

type SeriesPoint = {
  x: string;
  y: number;
};

export function HomePage() {
  const { data, loading } = useDashboard();
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [livePricePoints, setLivePricePoints] = useState<Array<{ timestamp: string; price: number }>>([]);

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const rawPrice = data.marketPrice?.price;
    const timestamp = data.marketPrice?.timestamp;
    const price = Number(rawPrice);
    if (!timestamp || !Number.isFinite(price)) {
      return;
    }
    setLivePricePoints((prev) => {
      const last = prev[prev.length - 1];
      if (last?.timestamp === timestamp) {
        return prev;
      }
      const next = [...prev, { timestamp, price }];
      return next.slice(-600);
    });
  }, [data.marketPrice?.price, data.marketPrice?.timestamp]);

  const lastResult = data.scheduler?.last_result ?? data.lastCycle ?? null;
  const reconciliation = lastResult?.reconciliation ?? null;
  const tradeSymbol = data.config?.trading_pair ?? "BTCUSDT";

  const priceSeries = useMemo<SeriesPoint[]>(() => {
    if (livePricePoints.length >= 2) {
      return livePricePoints.map((point) => ({ x: point.timestamp, y: point.price }));
    }
    return [...data.marketCandles]
      .reverse()
      .map((candle) => ({ x: candle.close_time, y: Number(candle.close) }))
      .filter((point) => Number.isFinite(point.y));
  }, [data.marketCandles, livePricePoints]);

  const equitySeries = useMemo<SeriesPoint[]>(
    () =>
      data.equity
        .map((point) => ({ x: point.timestamp, y: Number(point.equity) }))
        .filter((point) => Number.isFinite(point.y)),
    [data.equity],
  );

  const marketChangePct = useMemo(() => {
    if (priceSeries.length < 2) {
      return null;
    }
    const first = priceSeries[0]?.y ?? 0;
    const last = priceSeries[priceSeries.length - 1]?.y ?? 0;
    if (!Number.isFinite(first) || !Number.isFinite(last) || first === 0) {
      return null;
    }
    return ((last - first) / first) * 100;
  }, [priceSeries]);

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

  const activitySummary = useMemo(() => {
    if (data.scheduler?.last_error) {
      return `Scheduler error: ${data.scheduler.last_error}`;
    }
    if (!data.scheduler?.running) {
      return "Scheduler is stopped. Start scheduler to continue analysis.";
    }
    if (!data.systemReadiness?.can_trade) {
      return data.systemReadiness?.reasons.join(" | ") || "System is not RUNNING.";
    }
    if (lastResult && !lastResult.executed) {
      return explainRiskReason(lastResult.risk_reason);
    }
    return "Bot is running and evaluating each cycle.";
  }, [data.scheduler, data.systemReadiness, lastResult]);

  const expectedAction = useMemo(() => {
    if (!data.systemReadiness?.can_trade) {
      return {
        label: "Resume Required",
        detail: data.systemReadiness?.reasons.join(" | ") || "System is paused or emergency-stopped.",
      };
    }
    if (!data.scheduler?.running) {
      return {
        label: "Scheduler Stopped",
        detail: "Start scheduler to process the next market cycle.",
      };
    }
    if (!data.systemReadiness?.data_ready && data.systemReadiness?.require_data_ready) {
      return {
        label: "Warmup Pending",
        detail: data.systemReadiness.reasons.join(" | "),
      };
    }
    if (data.gridPreview?.bootstrap_eligible) {
      return {
        label: "Bootstrap BUY",
        detail: "Grid has no inventory. Next cycle can seed initial position to enable SELL legs.",
      };
    }
    if (lastResult && !lastResult.executed) {
      return {
        label: "Analyzing",
        detail: explainRiskReason(lastResult.risk_reason),
      };
    }
    return {
      label: "Monitoring",
      detail: "Waiting for next valid trigger in active strategy.",
    };
  }, [data.gridPreview?.bootstrap_eligible, data.scheduler?.running, data.systemReadiness, lastResult]);

  const latestEquity = data.equity.length > 0 ? data.equity[data.equity.length - 1]?.equity : null;
  const latestPrice = data.marketPrice?.price ?? (priceSeries.length > 0 ? String(priceSeries[priceSeries.length - 1]?.y) : null);

  const regime = data.marketIntelligence?.regime?.regime ?? "unknown";
  const regimeConfidence = data.marketIntelligence?.regime?.confidence ?? 0;
  const orderBook = data.marketIntelligence?.order_book;
  const llmStatus = data.llmStatus;
  const llmRuntimeValue =
    llmStatus?.last_provider_used && llmStatus?.last_model_used
      ? `${llmStatus.last_provider_used}:${llmStatus.last_model_used}`
      : llmStatus
        ? `${llmStatus.provider}:${llmStatus.model}`
        : "Unavailable";
  const llmRuntimeDetail = llmStatus
    ? llmStatus.last_used_at
      ? `Last used ${formatDateTime(llmStatus.last_used_at)}`
      : llmStatus.fallback_enabled
        ? `Fallback ${llmStatus.fallback_provider}:${llmStatus.fallback_model}`
        : "No successful call yet"
    : "LLM status endpoint unavailable";
  const llmTone: "ok" | "warn" | "danger" | "info" = !llmStatus
    ? "warn"
    : llmStatus.enabled && llmStatus.configured
      ? llmStatus.last_fallback_used
        ? "info"
        : "ok"
      : "warn";

  return (
    <section className="grid gap-4">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        <StatusCard
          label="System"
          value={data.system?.state ?? "unknown"}
          detail={data.system?.reason ?? "-"}
          tone={
            data.system?.state === "running"
              ? "ok"
              : data.system?.state === "emergency_stop"
                ? "danger"
                : "warn"
          }
          pulse={data.system?.state === "running"}
        />
        <StatusCard
          label="Scheduler"
          value={data.scheduler?.running ? "Running" : "Stopped"}
          detail={
            data.scheduler?.running
              ? nextRunInSeconds !== null
                ? `Next in ${formatCountdown(nextRunInSeconds)}`
                : "Waiting first cycle"
              : "Start scheduler from Controls"
          }
          tone={data.scheduler?.running ? "ok" : "warn"}
        />
        <StatusCard
          label="Data Readiness"
          value={data.systemReadiness?.data_ready ? "Ready" : "Pending"}
          detail={
            data.systemReadiness?.data_ready
              ? `${data.systemReadiness.active_strategy} ready`
              : data.systemReadiness?.reasons.join(" | ") || "Warmup pending"
          }
          tone={data.systemReadiness?.data_ready ? "ok" : "warn"}
        />
        <StatusCard
          label="Emergency AI"
          value={data.emergencySettings?.enabled ? "Enabled" : "Disabled"}
          detail={`max proposals ${data.emergencySettings?.max_proposals ?? "-"}`}
          tone={data.emergencySettings?.enabled ? "info" : "warn"}
        />
        <StatusCard
          label="LLM Runtime"
          value={llmRuntimeValue}
          detail={llmRuntimeDetail}
          tone={llmTone}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[2fr_1fr]">
        <Panel
          title="Market + Equity Live"
          subtitle={`${tradeSymbol} and portfolio equity (auto-refresh ~5s)`}
          right={
            <span className="rounded-md border border-white/20 bg-white/5 px-2 py-1 text-[11px] text-slate-300">
              Last sync {formatRelativeSeconds(data.lastUpdatedAt, nowMs)}
            </span>
          }
        >
          <DualSeriesChart
            leftLabel={`${tradeSymbol} price`}
            leftSeries={priceSeries}
            rightLabel="Equity"
            rightSeries={equitySeries}
          />
        </Panel>

        <Panel title="Quick Metrics" subtitle="PnL, returns, and risk overview">
          <div className="grid gap-2">
            <MetricCard label="Equity" value={latestEquity} suffix=" USDT" helper="latest snapshot" />
            <MetricCard
              label="Market Price"
              value={latestPrice}
              suffix=" USDT"
              trend={marketChangePct}
              helper={tradeSymbol}
            />
            <MetricCard label="Total PnL" value={data.metrics?.total_pnl} suffix=" USDT" helper="all cycles" />
            <MetricCard
              label="Win Rate"
              value={data.metrics?.win_rate !== null && data.metrics?.win_rate !== undefined ? data.metrics.win_rate * 100 : null}
              suffix="%"
              helper={`${data.metrics?.winning_trades ?? 0}/${data.metrics?.total_trades ?? 0} wins`}
            />
            <MetricCard
              label="Max Drawdown"
              value={(data.metrics?.max_drawdown ?? 0) * 100}
              suffix="%"
              helper="risk depth"
            />
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr_1fr]">
        <Panel title="Expected Next Action" subtitle="What the bot is waiting for right now">
          <div className="rounded-lg border border-sky-300/40 bg-sky-500/10 p-3">
            <p className="text-[11px] uppercase tracking-[0.2em] text-sky-100">{expectedAction.label}</p>
            <p className="mt-2 text-xs text-slate-100">{expectedAction.detail}</p>
          </div>
          <p className="mt-3 text-xs text-slate-300">{activitySummary}</p>
          <div className="mt-3 grid gap-2 text-xs text-slate-300 md:grid-cols-2">
            <p>Signal: {lastResult?.signal_side ?? "-"}</p>
            <p>Risk: {lastResult?.risk_action ?? "-"} / {lastResult?.risk_reason ?? "-"}</p>
            <p>Executed: {String(lastResult?.executed ?? false)}</p>
            <p>Last run: {formatDateTime(data.scheduler?.last_run_at)}</p>
            <p>Reconciliation: {reconciliation?.reason ?? "-"}</p>
            <p>Recon diff: {reconciliation ? formatNumber(reconciliation.difference, 4) : "-"}</p>
          </div>
          <div className="mt-3">
            <GridBandView preview={data.gridPreview} />
          </div>
        </Panel>

        <Panel title="Market Intelligence" subtitle="Regime, order book, and context snapshot">
          <div className="space-y-2 text-xs text-slate-200">
            <p>
              Regime: <span className="text-white">{regime}</span>{" "}
              ({formatNumber(regimeConfidence * 100, 1)}%)
            </p>
            <p>Fear & Greed: {data.marketIntelligence?.context.fear_greed ?? "-"}</p>
            <p>Funding Rate: {formatNumber(data.marketIntelligence?.context.funding_rate ?? null, 6)}</p>
            <p>24h Change: {formatNumber((data.marketIntelligence?.context.change_24h ?? 0) * 100, 2)}%</p>
            <p>24h Volume: {formatNumber(data.marketIntelligence?.context.volume_24h ?? null, 2)}</p>
            <p>Candles Used: {data.marketIntelligence?.candles_used ?? "-"}</p>
          </div>
          <div className="mt-3 rounded-lg border border-white/10 bg-black/20 p-3 text-xs text-slate-200">
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-300">Order Book</p>
            <p className="mt-2">Spread: {orderBook ? `${formatNumber(orderBook.spread_bps, 2)} bps` : "-"}</p>
            <p>Imbalance: {orderBook ? formatNumber(orderBook.imbalance, 3) : "-"}</p>
            <p>Liquidity score: {orderBook ? formatNumber(orderBook.liquidity_score, 2) : "-"}</p>
            <p>Impact 1 BTC: {orderBook ? `${formatNumber(orderBook.market_impact_1btc_bps, 2)} bps` : "-"}</p>
          </div>
        </Panel>

        <Panel title="Recent Activity" subtitle="Trade, risk, AI, and system timeline">
          <ActivityFeed items={data.events} limit={8} />
        </Panel>
      </div>

      {loading ? (
        <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-xs text-slate-300">
          Refreshing dashboard...
        </div>
      ) : null}
    </section>
  );
}
