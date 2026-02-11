import type { AuditEvent } from "../api";
import { formatDateTime } from "../utils";

type ActivityFeedProps = {
  items: AuditEvent[];
  emptyLabel?: string;
  limit?: number;
};

function categoryTone(category: string): string {
  if (category === "risk") {
    return "border-amber-300/40 bg-amber-500/10 text-amber-100";
  }
  if (category === "trade") {
    return "border-emerald-300/40 bg-emerald-500/10 text-emerald-100";
  }
  if (category === "ai") {
    return "border-cyan-300/40 bg-cyan-500/10 text-cyan-100";
  }
  if (category === "system") {
    return "border-fuchsia-300/40 bg-fuchsia-500/10 text-fuchsia-100";
  }
  return "border-slate-300/40 bg-slate-500/10 text-slate-100";
}

export function ActivityFeed({
  items,
  emptyLabel = "No activity yet.",
  limit = 10,
}: ActivityFeedProps) {
  const rows = items.slice(0, limit);
  if (rows.length === 0) {
    return <p className="text-xs text-slate-400">{emptyLabel}</p>;
  }
  return (
    <div className="space-y-2">
      {rows.map((event) => (
        <article key={event.id} className={`rounded-lg border p-2 ${categoryTone(event.event_category)}`}>
          <div className="flex items-center justify-between gap-2">
            <p className="text-[11px] uppercase tracking-[0.15em]">
              {event.event_category}/{event.event_type}
            </p>
            <p className="text-[11px] opacity-80">{formatDateTime(event.created_at)}</p>
          </div>
          <p className="mt-1 text-xs">{event.summary}</p>
          <p className="mt-1 text-[11px] opacity-80">actor: {event.actor}</p>
        </article>
      ))}
    </div>
  );
}
