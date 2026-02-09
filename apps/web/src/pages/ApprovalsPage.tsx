import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { useAuth } from "../auth";
import { useDashboard } from "../dashboard";
import { formatDateTime, formatNumber } from "../utils";

function formatRemaining(expiresAt: string, nowMs: number): string {
  const diff = new Date(expiresAt).getTime() - nowMs;
  if (diff <= 0) {
    return "expired";
  }
  const totalSeconds = Math.floor(diff / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${hours}h ${minutes}m ${seconds}s`;
}

export function ApprovalsPage() {
  const { user } = useAuth();
  const { data, refresh } = useDashboard();
  const [nowMs, setNowMs] = useState(Date.now());
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pending = useMemo(
    () => data.approvals.filter((approval) => approval.status === "PENDING"),
    [data.approvals],
  );

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  async function decide(approvalId: number, action: "approve" | "reject") {
    try {
      setError(null);
      setMessage(`${action === "approve" ? "Approving" : "Rejecting"} proposal...`);
      if (action === "approve") {
        await api.approveProposal(approvalId, user?.username ?? "web_user");
      } else {
        await api.rejectProposal(approvalId, user?.username ?? "web_user");
      }
      await refresh();
      setMessage("Decision submitted");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Decision failed");
    } finally {
      window.setTimeout(() => setMessage(null), 1400);
    }
  }

  return (
    <section className="grid gap-4 lg:grid-cols-2">
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <div className="flex items-center justify-between">
          <h2 className="font-heading text-lg font-bold text-white">Pending Approvals</h2>
          <span className="text-xs text-slate-300">{pending.length} pending</span>
        </div>
        {message ? <p className="mt-3 text-xs text-mint">{message}</p> : null}
        {error ? <p className="mt-3 text-xs text-rose-200">{error}</p> : null}
        <div className="mt-3 space-y-3">
          {pending.map((approval) => (
            <div key={approval.id} className="rounded-lg border border-white/10 bg-black/20 p-3">
              <p className="font-heading text-sm font-bold text-white">
                #{approval.id} {approval.title}
              </p>
              <p className="mt-1 text-[11px] text-slate-300">
                confidence {formatNumber(approval.confidence * 100, 1)}% | remaining{" "}
                {formatRemaining(approval.expires_at, nowMs)}
              </p>
              <p className="mt-2 text-xs text-slate-200">{approval.description}</p>
              <div className="mt-3 flex gap-2">
                <button
                  className="rounded border border-emerald-300/50 bg-emerald-500/10 px-2 py-1 text-[11px] text-emerald-200"
                  onClick={() => void decide(approval.id, "approve")}
                >
                  Approve
                </button>
                <button
                  className="rounded border border-rose-300/50 bg-rose-500/10 px-2 py-1 text-[11px] text-rose-200"
                  onClick={() => void decide(approval.id, "reject")}
                >
                  Reject
                </button>
              </div>
            </div>
          ))}
          {pending.length === 0 ? <p className="text-xs text-slate-400">No pending approvals.</p> : null}
        </div>
      </article>

      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Recent Decisions</h2>
        <div className="mt-3 max-h-[540px] space-y-2 overflow-auto">
          {data.approvals
            .filter((approval) => approval.status !== "PENDING")
            .slice(0, 20)
            .map((approval) => (
              <div key={approval.id} className="rounded-lg border border-white/10 bg-black/20 p-2">
                <p className="text-sm text-white">
                  #{approval.id} {approval.title}
                </p>
                <p className="mt-1 text-[11px] text-slate-400">
                  {approval.status} by {approval.decided_by ?? "-"} at {formatDateTime(approval.decided_at)}
                </p>
              </div>
            ))}
          {data.approvals.filter((approval) => approval.status !== "PENDING").length === 0 ? (
            <p className="text-xs text-slate-400">No decisions yet.</p>
          ) : null}
        </div>
      </article>
    </section>
  );
}

