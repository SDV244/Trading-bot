import { Sparkline } from "../components/Sparkline";
import { useDashboard } from "../dashboard";
import { formatDateTime, formatNumber } from "../utils";

export function HomePage() {
  const { data, loading } = useDashboard();
  const equityValues = data.equity.map((point) => Number(point.equity));
  const latestEquity = data.equity.length > 0 ? data.equity[data.equity.length - 1]?.equity : null;

  return (
    <section className="grid gap-4">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
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
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">Readiness</p>
          <p className="mt-3 font-heading text-2xl font-bold text-white">{data.ready?.ready ? "Ready" : "Not ready"}</p>
          <p className="mt-2 text-xs text-slate-400">
            DB {String(data.ready?.database ?? false)} | Binance {String(data.ready?.binance ?? false)}
          </p>
        </article>
      </div>

      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-heading text-lg font-bold text-white">Equity Curve</h2>
            <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">{data.equity.length} points</p>
          </div>
          <Sparkline points={equityValues} />
        </article>
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <h2 className="font-heading text-lg font-bold text-white">System Snapshot</h2>
          <div className="mt-3 space-y-2 text-xs text-slate-200">
            <p>State: {data.system?.state ?? "-"}</p>
            <p>Reason: {data.system?.reason ?? "-"}</p>
            <p>Changed At: {formatDateTime(data.system?.changed_at)}</p>
            <p>Scheduler: {data.scheduler?.running ? "running" : "stopped"}</p>
            <p>Last Run: {formatDateTime(data.scheduler?.last_run_at)}</p>
            <p>Last Error: {data.scheduler?.last_error ?? "-"}</p>
          </div>
        </article>
      </div>

      {loading ? (
        <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-xs text-slate-300">Refreshing data...</div>
      ) : null}
    </section>
  );
}

