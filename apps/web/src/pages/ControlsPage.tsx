import { useEffect, useMemo, useState } from "react";

import { api, type NotificationStatus } from "../api";
import { hasMinRole, useAuth } from "../auth";
import { useDashboard } from "../dashboard";

export function ControlsPage() {
  const { user } = useAuth();
  const { data, refresh, runPaperCycle } = useDashboard();
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notificationStatus, setNotificationStatus] = useState<NotificationStatus | null>(null);
  const [notificationHint, setNotificationHint] = useState<string | null>(null);

  async function refreshNotificationStatus() {
    try {
      const status = await api.getNotificationStatus();
      setNotificationStatus(status);
      setNotificationHint(
        status.enabled
          ? "Telegram is configured."
          : "Telegram is not configured yet. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in secrets.",
      );
    } catch (e) {
      setNotificationHint(e instanceof Error ? e.message : "Failed to read notification status");
    }
  }

  useEffect(() => {
    void refreshNotificationStatus();
  }, []);

  async function runAction(label: string, action: () => Promise<void>) {
    try {
      setError(null);
      setMessage(label);
      await action();
      await refresh();
      await refreshNotificationStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Action failed");
    } finally {
      window.setTimeout(() => setMessage(null), 1200);
    }
  }

  const checklist = useMemo(
    () => [
      {
        label: "Market warmup data ready",
        ready: Boolean(data.systemReadiness?.data_ready),
        detail: data.systemReadiness?.reasons.join(" | ") || "Warmup complete.",
      },
      {
        label: "System state RUNNING",
        ready: data.system?.state === "running",
        detail: data.system?.reason ?? "No state reason available.",
      },
      {
        label: "Scheduler active",
        ready: Boolean(data.scheduler?.running),
        detail: data.scheduler?.running ? "Scheduler is cycling." : "Start scheduler to run periodic cycles.",
      },
      {
        label: "Telegram notifications ready",
        ready: Boolean(notificationStatus?.enabled),
        detail: notificationStatus?.enabled
          ? "Bot token + chat id detected."
          : "Configure secrets and click Refresh Notification Status.",
      },
    ],
    [data.scheduler?.running, data.system?.reason, data.system?.state, data.systemReadiness?.data_ready, data.systemReadiness?.reasons, notificationStatus?.enabled],
  );

  return (
    <section className="grid gap-4 md:grid-cols-2">
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur md:col-span-2">
        <h2 className="font-heading text-lg font-bold text-white">Paper Trading Startup Checklist</h2>
        <div className="mt-3 grid gap-2 text-xs text-slate-200 md:grid-cols-2">
          {checklist.map((item) => (
            <div key={item.label} className="rounded-lg border border-white/10 bg-black/10 p-3">
              <p className={item.ready ? "text-emerald-200" : "text-amber-200"}>
                {item.ready ? "READY" : "WAIT"} - {item.label}
              </p>
              <p className="mt-1 text-slate-300">{item.detail}</p>
            </div>
          ))}
        </div>
      </article>

      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">System Controls</h2>
        {message ? <p className="mt-3 text-xs text-mint">{message}</p> : null}
        {error ? <p className="mt-3 text-xs text-rose-200">{error}</p> : null}
        {data.systemReadiness && !data.systemReadiness.data_ready && data.systemReadiness.require_data_ready ? (
          <p className="mt-3 text-xs text-amber-200">
            Scheduler warmup blocked: {data.systemReadiness.reasons.join(" | ")}
          </p>
        ) : null}
        <div className="mt-4 grid gap-2">
          <button
            className="rounded-lg border border-mint/50 bg-mint/15 px-3 py-2 text-left text-xs text-mint hover:bg-mint/20"
            onClick={() => void runAction("Resuming system...", () => api.setSystemState("resume", "dashboard_resume").then(() => undefined))}
          >
            Resume Trading
          </button>
          <button
            className="rounded-lg border border-amber-400/50 bg-amber-400/10 px-3 py-2 text-left text-xs text-amber-200 hover:bg-amber-400/20"
            onClick={() => void runAction("Pausing system...", () => api.setSystemState("pause", "dashboard_pause").then(() => undefined))}
          >
            Pause Trading
          </button>
          <button
            className="rounded-lg border border-rose-400/60 bg-rose-500/10 px-3 py-2 text-left text-xs text-rose-200 hover:bg-rose-500/20"
            onClick={() =>
              void runAction("Emergency stop triggered...", () =>
                api.setSystemState("emergency_stop", "dashboard_emergency_stop").then(() => undefined),
              )
            }
          >
            Emergency Stop
          </button>
          <button
            className="rounded-lg border border-sky-300/60 bg-sky-500/10 px-3 py-2 text-left text-xs text-sky-200 hover:bg-sky-500/20"
            onClick={() => void runAction("Running one paper cycle...", runPaperCycle)}
          >
            Run One Paper Cycle
          </button>
          {data.scheduler?.running ? (
            <button
              className="rounded-lg border border-rose-300/50 bg-rose-500/10 px-3 py-2 text-left text-xs text-rose-200 hover:bg-rose-500/20"
              onClick={() => void runAction("Stopping scheduler...", () => api.stopScheduler().then(() => undefined))}
            >
              Stop Scheduler
            </button>
          ) : (
            <button
              className="rounded-lg border border-emerald-300/50 bg-emerald-500/10 px-3 py-2 text-left text-xs text-emerald-200 hover:bg-emerald-500/20"
              onClick={() => void runAction("Starting scheduler...", () => api.startScheduler(60).then(() => undefined))}
            >
              Start Scheduler (60s)
            </button>
          )}
          <button
            className="rounded-lg border border-cyan-300/60 bg-cyan-500/10 px-3 py-2 text-left text-xs text-cyan-100 hover:bg-cyan-500/20"
            onClick={() =>
              void runAction("Sending Telegram test notification...", async () => {
                const result = await api.sendTestNotification(
                  "Trading Bot test notification",
                  "Connectivity check from Controls page",
                );
                setNotificationHint(result.message);
              })
            }
          >
            Send Telegram Test
          </button>
          <button
            className="rounded-lg border border-slate-300/40 bg-slate-500/10 px-3 py-2 text-left text-xs text-slate-200 hover:bg-slate-500/20"
            onClick={() => void runAction("Refreshing notification status...", refreshNotificationStatus)}
          >
            Refresh Notification Status
          </button>
        </div>
      </article>

      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">AI & Optimization</h2>
        <div className="mt-4 grid gap-2">
          <button
            className="rounded-lg border border-indigo-300/60 bg-indigo-500/10 px-3 py-2 text-left text-xs text-indigo-200 hover:bg-indigo-500/20"
            onClick={() => void runAction("Generating AI proposals...", () => api.generateProposals().then(() => undefined))}
          >
            Generate AI Proposals
          </button>
          <button
            className="rounded-lg border border-violet-300/60 bg-violet-500/10 px-3 py-2 text-left text-xs text-violet-200 hover:bg-violet-500/20"
            onClick={() => void runAction("Training DRL optimizer...", () => api.trainOptimizer(1024).then(() => undefined))}
          >
            Train DRL Optimizer
          </button>
          <button
            className="rounded-lg border border-fuchsia-300/60 bg-fuchsia-500/10 px-3 py-2 text-left text-xs text-fuchsia-200 hover:bg-fuchsia-500/20"
            onClick={() => void runAction("Checking expired approvals...", () => api.expireApprovals().then(() => undefined))}
          >
            Expire Pending Approvals
          </button>
        </div>

        {hasMinRole(user?.role, "admin") ? (
          <div className="mt-6 rounded-lg border border-rose-400/40 bg-rose-500/10 p-3">
            <p className="text-xs text-rose-100">
              Admin-only live controls are exposed through `/api/trading/live/order` and should be used with strict
              change management.
            </p>
          </div>
        ) : null}

        <div className="mt-6 rounded-lg border border-white/10 bg-black/10 p-3 text-xs text-slate-200">
          <p>Telegram Enabled: {String(notificationStatus?.enabled ?? false)}</p>
          <p>Token Present: {String(notificationStatus?.has_bot_token ?? false)}</p>
          <p>Chat ID Present: {String(notificationStatus?.has_chat_id ?? false)}</p>
          <p className="mt-1 text-slate-300">{notificationHint ?? "-"}</p>
        </div>
      </article>
    </section>
  );
}
