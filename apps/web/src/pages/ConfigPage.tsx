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
          <p>Active Strategy: {config?.active_strategy ?? "-"}</p>
          <p>Supported Strategies: {config?.supported_strategies.join(", ") ?? "-"}</p>
          <p>Live Mode: {String(config?.live_mode ?? false)}</p>
          <p>Require Data Ready: {String(config?.require_data_ready ?? false)}</p>
          <p>Spot Position Mode: {config?.spot_position_mode ?? "-"}</p>
          <p>AI Advisor Interval: every {config?.advisor_interval_cycles ?? "-"} cycles</p>
          <p>Min Scheduler Interval: {config?.min_cycle_interval_seconds ?? "-"}s</p>
          <p>Reconciliation Interval: every {config?.reconciliation_interval_cycles ?? "-"} cycles</p>
          <p>Reconciliation Warning Tol: {formatNumber(config?.reconciliation_warning_tolerance ?? 0, 4)}</p>
          <p>Reconciliation Critical Tol: {formatNumber(config?.reconciliation_critical_tolerance ?? 0, 4)}</p>
          <p>Paper Starting Equity: {formatNumber(config?.paper_starting_equity ?? 0, 2)} USDT</p>
          <p>Approval Timeout: {config?.approval_timeout_hours ?? "-"} hours</p>
          <p>Auto-Approve AI Suggestions: {String(config?.approval_auto_approve_enabled ?? false)}</p>
          <p>Emergency AI Supervisor Enabled: {String(config?.approval_emergency_ai_enabled ?? false)}</p>
          <p>Emergency AI Max Proposals: {config?.approval_emergency_max_proposals ?? "-"}</p>
        </div>
      </article>
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Smart Grid Params</h2>
        <div className="mt-3 space-y-2 text-xs text-slate-200">
          <p>Lookback 1h: {config?.grid_lookback_1h ?? "-"}</p>
          <p>ATR Period 1h: {config?.grid_atr_period_1h ?? "-"}</p>
          <p>Grid Levels: {config?.grid_levels ?? "-"}</p>
          <p>Spacing Mode: {config?.grid_spacing_mode ?? "-"}</p>
          <p>Grid Min Spacing: {config?.grid_min_spacing_bps ?? "-"} bps</p>
          <p>Grid Max Spacing: {config?.grid_max_spacing_bps ?? "-"} bps</p>
          <p>Trend Tilt: {formatNumber(config?.grid_trend_tilt ?? 0, 2)}</p>
          <p>Volatility Blend: {formatNumber(config?.grid_volatility_blend ?? 0, 2)}</p>
          <p>Take Profit Buffer: {formatNumber((config?.grid_take_profit_buffer ?? 0) * 100, 2)}%</p>
          <p>Stop Loss Buffer: {formatNumber((config?.grid_stop_loss_buffer ?? 0) * 100, 2)}%</p>
          <p>Cooldown: {config?.grid_cooldown_seconds ?? "-"} s</p>
          <p>Auto Bootstrap Inventory: {String(config?.grid_auto_inventory_bootstrap ?? false)}</p>
          <p>Bootstrap Fraction: {formatNumber((config?.grid_bootstrap_fraction ?? 0) * 100, 0)}%</p>
          <p>Enforce Fee Floor: {String(config?.grid_enforce_fee_floor ?? false)}</p>
          <p>Min Net Profit Floor: {config?.grid_min_net_profit_bps ?? "-"} bps</p>
          <p>Out-of-Bounds Alert Cooldown: {config?.grid_out_of_bounds_alert_cooldown_minutes ?? "-"} min</p>
          <p>Recenter Mode: {config?.grid_recenter_mode ?? "-"}</p>
          <p>Global Stop-Loss Enabled: {String(config?.stop_loss_enabled ?? false)}</p>
          <p>Global Equity Stop: {formatNumber((config?.stop_loss_global_equity_pct ?? 0) * 100, 2)}%</p>
          <p>Max Drawdown Stop: {formatNumber((config?.stop_loss_max_drawdown_pct ?? 0) * 100, 2)}%</p>
          <p>Auto-Close on Stop: {String(config?.stop_loss_auto_close_positions ?? false)}</p>
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
