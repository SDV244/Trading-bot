type BadgeProps = {
  value: string;
};

export function StateBadge({ value }: BadgeProps) {
  const normalized = value.toLowerCase();
  const cls =
    normalized === "running"
      ? "bg-emerald-400/20 text-emerald-200 border-emerald-300/40"
      : normalized === "paused"
        ? "bg-amber-400/20 text-amber-200 border-amber-300/40"
        : "bg-rose-500/20 text-rose-200 border-rose-300/40";

  return (
    <span className={`rounded-full border px-3 py-1 text-xs uppercase tracking-[0.2em] ${cls}`}>
      {value}
    </span>
  );
}

