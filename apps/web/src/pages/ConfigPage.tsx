import { useDashboard } from "../dashboard";
import { formatNumber } from "../utils";

export function ConfigPage() {
  const { data } = useDashboard();
  const config = data.config;

  return (
    <section className="grid gap-4 md:grid-cols-2">
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Trading Config</h2>
        <div className="mt-3 space-y-2 text-xs text-slate-200">
          <p>Pair: {config?.trading_pair ?? "-"}</p>
          <p>Timeframes: {config?.timeframes.join(", ") ?? "-"}</p>
          <p>Live Mode: {String(config?.live_mode ?? false)}</p>
          <p>Approval Timeout: {config?.approval_timeout_hours ?? "-"} hours</p>
        </div>
      </article>
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Risk Config</h2>
        <div className="mt-3 space-y-2 text-xs text-slate-200">
          <p>Risk per Trade: {formatNumber((config?.risk_per_trade ?? 0) * 100, 3)}%</p>
          <p>Max Daily Loss: {formatNumber((config?.max_daily_loss ?? 0) * 100, 2)}%</p>
          <p>Max Exposure: {formatNumber((config?.max_exposure ?? 0) * 100, 2)}%</p>
          <p>Fee BPS: {config?.fee_bps ?? "-"}</p>
          <p>Slippage BPS: {config?.slippage_bps ?? "-"}</p>
        </div>
      </article>
      <article className="rounded-2xl border border-amber-400/30 bg-amber-500/10 p-4 text-xs text-amber-100 md:col-span-2">
        Runtime edits are intentionally disabled in this panel for safety. Config changes must pass AI proposal +
        approval flow and be captured in audit logs.
      </article>
    </section>
  );
}

