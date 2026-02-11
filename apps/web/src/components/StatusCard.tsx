type StatusTone = "ok" | "warn" | "danger" | "info";

type StatusCardProps = {
  label: string;
  value: string;
  detail?: string;
  tone?: StatusTone;
  pulse?: boolean;
};

function toneClasses(tone: StatusTone): string {
  if (tone === "ok") {
    return "border-emerald-300/40 bg-emerald-500/10 text-emerald-100";
  }
  if (tone === "warn") {
    return "border-amber-300/40 bg-amber-500/10 text-amber-100";
  }
  if (tone === "danger") {
    return "border-rose-300/40 bg-rose-500/10 text-rose-100";
  }
  return "border-sky-300/40 bg-sky-500/10 text-sky-100";
}

export function StatusCard({
  label,
  value,
  detail,
  tone = "info",
  pulse = false,
}: StatusCardProps) {
  return (
    <article className={`rounded-xl border p-3 ${toneClasses(tone)}`}>
      <p className="text-[11px] uppercase tracking-[0.2em]">{label}</p>
      <p className="mt-2 flex items-center gap-2 font-heading text-xl font-bold">
        {pulse ? <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-current" /> : null}
        {value}
      </p>
      {detail ? <p className="mt-1 text-xs opacity-90">{detail}</p> : null}
    </article>
  );
}
