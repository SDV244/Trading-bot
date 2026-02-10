export function formatNumber(value: string | number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) {
    return "-";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value);
  }
  return numeric.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleString();
}

export function formatTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleTimeString();
}

export function formatRelativeSeconds(value: string | null | undefined, nowMs: number = Date.now()): string {
  if (!value) {
    return "-";
  }
  const ts = new Date(value).getTime();
  if (!Number.isFinite(ts)) {
    return "-";
  }
  const diffSec = Math.max(0, Math.floor((nowMs - ts) / 1000));
  if (diffSec < 60) {
    return `${diffSec}s ago`;
  }
  const mins = Math.floor(diffSec / 60);
  const secs = diffSec % 60;
  return `${mins}m ${secs}s ago`;
}

export function formatCountdown(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs.toString().padStart(2, "0")}s`;
}

export function explainRiskReason(reason: string | null | undefined): string {
  const normalized = (reason ?? "").toLowerCase();
  if (normalized.startsWith("cooldown_active_")) {
    const rawSeconds = normalized.replace("cooldown_active_", "").replace("s", "");
    const seconds = Number(rawSeconds);
    if (Number.isFinite(seconds)) {
      return `Cooldown is active for another ${seconds}s to avoid overtrading.`;
    }
    return "Cooldown is active to avoid overtrading.";
  }
  if (normalized.startsWith("fee_floor_not_met")) {
    return "Grid spacing is too tight versus fee/slippage floor. Increase spacing or reduce costs.";
  }
  switch (normalized) {
    case "risk_checks_passed":
      return "All risk checks passed. Trade was allowed.";
    case "no_inventory_to_sell":
      return "Signal is SELL, but paper position is flat. Waiting for a BUY setup first.";
    case "already_in_position":
      return "Signal is BUY, but spot mode is long-flat and a position is already open.";
    case "max_exposure_reached":
      return "Position is at max allowed exposure. Waiting for exposure to reduce.";
    case "signal_hold":
      return "Strategy returned HOLD. No trade action this cycle.";
    case "bearish_wait_retest":
      return "Trend is bearish but no confirmed retest trigger yet. Bot keeps monitoring.";
    case "bullish_wait_retest":
      return "Trend is bullish but entry retest conditions are not met yet.";
    case "weak_regime":
      return "Market regime is weak/unclear. Bot is waiting for stronger directional structure.";
    case "grid_wait_inside_band":
      return "Price is inside the active grid band. Waiting for next grid trigger cross.";
    case "grid_recenter_wait":
      return "Price is outside the active band. Strategy is waiting for re-center conditions.";
    case "grid_recentered_auto":
      return "Price moved outside the active grid. Strategy auto-recentered the band and is monitoring next trigger.";
    case "grid_recentered_auto_breakout_buy":
      return "Price broke above the old band. Strategy auto-recentered and entered breakout BUY mode.";
    case "grid_recentered_auto_breakdown_sell":
      return "Price broke below the old band. Strategy auto-recentered and signaled defensive SELL.";
    case "grid_inventory_bootstrap":
      return "Smart grid seeded a starter BUY inventory so future SELL grid levels can execute.";
    case "insufficient_data":
      return "Indicators are still warming up with candle history.";
    case "below_min_notional":
    case "final_notional_below_minimum":
      return "Order size is below exchange minimum notional. Waiting for larger size opportunity.";
    case "insufficient_candles":
      return "Warmup data is incomplete. Waiting for required candles.";
    case "system_not_running":
      return "System is paused or emergency-stopped.";
    default:
      return reason ? `Waiting reason: ${reason}` : "No blocking reason reported.";
  }
}
