import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type AuditEvent } from "../api";
import { formatDateTime } from "../utils";

type FilterState = {
  category: string;
  eventType: string;
  actor: string;
  search: string;
  limit: number;
};

const DEFAULT_FILTERS: FilterState = {
  category: "",
  eventType: "",
  actor: "",
  search: "",
  limit: 200,
};

export function LogsPage() {
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchEvents = useCallback(async () => {
    try {
      setLoading(true);
      const rows = await api.listEvents({
        category: filters.category || undefined,
        event_type: filters.eventType || undefined,
        actor: filters.actor || undefined,
        search: filters.search || undefined,
        limit: filters.limit,
      });
      setEvents(rows);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch events");
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    void fetchEvents();
  }, [fetchEvents]);

  const categories = useMemo(() => Array.from(new Set(events.map((event) => event.event_category))).sort(), [events]);
  const eventTypes = useMemo(() => Array.from(new Set(events.map((event) => event.event_type))).sort(), [events]);
  const actors = useMemo(() => Array.from(new Set(events.map((event) => event.actor))).sort(), [events]);

  function updateFilter<K extends keyof FilterState>(key: K, value: FilterState[K]) {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <section className="grid gap-4">
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Audit Log Filters</h2>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <label className="text-xs text-slate-200">
            Category
            <select
              className="mt-1 w-full rounded border border-white/20 bg-black/20 p-2 text-xs"
              value={filters.category}
              onChange={(event) => updateFilter("category", event.target.value)}
            >
              <option value="">All</option>
              {categories.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-slate-200">
            Event Type
            <select
              className="mt-1 w-full rounded border border-white/20 bg-black/20 p-2 text-xs"
              value={filters.eventType}
              onChange={(event) => updateFilter("eventType", event.target.value)}
            >
              <option value="">All</option>
              {eventTypes.map((eventType) => (
                <option key={eventType} value={eventType}>
                  {eventType}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-slate-200">
            Actor
            <select
              className="mt-1 w-full rounded border border-white/20 bg-black/20 p-2 text-xs"
              value={filters.actor}
              onChange={(event) => updateFilter("actor", event.target.value)}
            >
              <option value="">All</option>
              {actors.map((actor) => (
                <option key={actor} value={actor}>
                  {actor}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-slate-200 md:col-span-2">
            Search Summary
            <input
              className="mt-1 w-full rounded border border-white/20 bg-black/20 p-2 text-xs"
              value={filters.search}
              onChange={(event) => updateFilter("search", event.target.value)}
              placeholder="text in summary"
            />
          </label>
          <label className="text-xs text-slate-200">
            Limit
            <input
              type="number"
              min={1}
              max={1000}
              className="mt-1 w-full rounded border border-white/20 bg-black/20 p-2 text-xs"
              value={filters.limit}
              onChange={(event) => updateFilter("limit", Number(event.target.value))}
            />
          </label>
        </div>
        <div className="mt-3 flex gap-2">
          <button
            className="rounded border border-mint/50 bg-mint/15 px-3 py-1 text-xs text-mint"
            onClick={() => void fetchEvents()}
          >
            Apply Filters
          </button>
          <button
            className="rounded border border-white/30 bg-white/5 px-3 py-1 text-xs text-slate-200"
            onClick={() => setFilters(DEFAULT_FILTERS)}
          >
            Reset
          </button>
        </div>
      </article>

      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Events</h2>
        {error ? <p className="mt-3 text-xs text-rose-200">{error}</p> : null}
        {loading ? <p className="mt-3 text-xs text-slate-300">Loading...</p> : null}
        <div className="mt-3 max-h-[560px] overflow-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-slate-400">
                <th className="pb-2">Time</th>
                <th className="pb-2">Category</th>
                <th className="pb-2">Type</th>
                <th className="pb-2">Actor</th>
                <th className="pb-2">Summary</th>
              </tr>
            </thead>
            <tbody>
              {events.map((event) => (
                <tr key={event.id} className="border-t border-white/5 text-slate-200">
                  <td className="py-2">{formatDateTime(event.created_at)}</td>
                  <td className="py-2">{event.event_category}</td>
                  <td className="py-2">{event.event_type}</td>
                  <td className="py-2">{event.actor}</td>
                  <td className="py-2">{event.summary}</td>
                </tr>
              ))}
              {events.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-3 text-slate-400">
                    No events found.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}

