import { formatNumber } from "../utils";

type MetricCardProps = {
  label: string;
  value: string | number | null | undefined;
  suffix?: string;
  digits?: number;
  trend?: number | null;
  helper?: string;
};

export function MetricCard({
  label,
  value,
  suffix = "",
  digits = 2,
  trend = null,
  helper,
}: MetricCardProps) {
  const trendLabel =
    trend === null || trend === undefined
      ? null
      : `${trend >= 0 ? "+" : ""}${formatNumber(trend, 2)}%`;
  const trendClass =
    trend === null || trend === undefined
      ? "text-slate-300"
      : trend >= 0
        ? "text-emerald-200"
        : "text-rose-200";
  return (
    <article className="rounded-xl border border-white/10 bg-black/10 p-3">
      <p className="text-[11px] uppercase tracking-[0.2em] text-slate-300">{label}</p>
      <p className="mt-2 font-heading text-xl font-bold text-white">
        {formatNumber(value, digits)}
        {suffix}
      </p>
      <div className="mt-1 flex items-center justify-between gap-2">
        <p className={`text-xs ${trendClass}`}>{trendLabel ?? "-"}</p>
        {helper ? <p className="text-[11px] text-slate-400">{helper}</p> : null}
      </div>
    </article>
  );
}
