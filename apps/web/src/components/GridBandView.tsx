import type { GridPreview } from "../api";
import { formatNumber } from "../utils";

type GridBandViewProps = {
  preview: GridPreview | null;
};

function toPercent(position: number, min: number, max: number): number {
  if (!Number.isFinite(position) || !Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return 50;
  }
  return Math.max(0, Math.min(100, ((position - min) / (max - min)) * 100));
}

export function GridBandView({ preview }: GridBandViewProps) {
  if (!preview) {
    return (
      <div className="rounded-xl border border-white/10 bg-black/20 p-4 text-xs text-slate-300">
        Grid projection is not available yet.
      </div>
    );
  }

  const lower = Number(preview.grid_lower ?? NaN);
  const upper = Number(preview.grid_upper ?? NaN);
  const price = Number(preview.last_price ?? NaN);
  const buyTrigger = Number(preview.buy_trigger ?? NaN);
  const sellTrigger = Number(preview.sell_trigger ?? NaN);

  const low = Number.isFinite(lower) ? lower : price;
  const high = Number.isFinite(upper) ? upper : price;

  const pricePct = toPercent(price, low, high);
  const buyPct = toPercent(buyTrigger, low, high);
  const sellPct = toPercent(sellTrigger, low, high);

  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-3">
      <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">
        Grid Band ({preview.recenter_mode})
      </p>
      <div className="relative mt-3 h-10 rounded-lg border border-white/10 bg-slate-900/70">
        <div
          className="absolute inset-y-0 rounded-lg bg-emerald-500/15"
          style={{ left: `${Math.min(buyPct, sellPct)}%`, width: `${Math.abs(sellPct - buyPct)}%` }}
        />
        <div
          className="absolute inset-y-0 w-0 border-l-2 border-amber-300"
          style={{ left: `${pricePct}%` }}
          title={`Price ${formatNumber(price, 2)}`}
        />
        <div className="absolute bottom-0 top-0 w-0 border-l border-emerald-300/80" style={{ left: `${buyPct}%` }} />
        <div className="absolute bottom-0 top-0 w-0 border-l border-rose-300/80" style={{ left: `${sellPct}%` }} />
      </div>
      <div className="mt-2 grid gap-1 text-[11px] text-slate-300 md:grid-cols-2">
        <p>Lower: {formatNumber(preview.grid_lower, 2)}</p>
        <p>Upper: {formatNumber(preview.grid_upper, 2)}</p>
        <p>Buy trigger: {formatNumber(preview.buy_trigger, 2)}</p>
        <p>Sell trigger: {formatNumber(preview.sell_trigger, 2)}</p>
        <p>Signal: {preview.signal_side}</p>
        <p>Reason: {preview.signal_reason}</p>
      </div>
    </div>
  );
}
