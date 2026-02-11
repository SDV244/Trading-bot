import { useMemo, useState } from "react";

import { api, type StrategyAnalysis } from "../api";
import { hasMinRole, useAuth } from "../auth";
import { Panel } from "../components/Panel";
import { StatusCard } from "../components/StatusCard";
import { useDashboard } from "../dashboard";
import { formatDateTime, formatNumber } from "../utils";

export function IntelligencePage() {
  const { user } = useAuth();
  const { data, refresh } = useDashboard();
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runningCheck, setRunningCheck] = useState(false);
  const [strategyAnalysis, setStrategyAnalysis] = useState<StrategyAnalysis | null>(null);
  const [targetMaxProposals, setTargetMaxProposals] = useState(
    data.emergencySettings?.max_proposals ?? 3,
  );

  const approvalsPending = useMemo(
    () => data.approvals.filter((approval) => approval.status === "PENDING"),
    [data.approvals],
  );
  const aiEvents = useMemo(
    () =>
      data.events
        .filter((event) => event.event_category === "ai" || event.event_type.includes("emergency_ai"))
        .slice(0, 10),
    [data.events],
  );
  const canOperate = hasMinRole(user?.role, "operator");
  const autoApprove = data.config?.approval_auto_approve_enabled ?? false;
  const emergencyEnabled = data.emergencySettings?.enabled ?? data.config?.approval_emergency_ai_enabled ?? true;
  const emergencyMax = data.emergencySettings?.max_proposals ?? data.config?.approval_emergency_max_proposals ?? 3;
  const llmStatus = data.llmStatus;

  async function runAction(label: string, fn: () => Promise<void>) {
    try {
      setError(null);
      setMessage(label);
      setRunningCheck(true);
      await fn();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Action failed");
    } finally {
      setRunningCheck(false);
      window.setTimeout(() => setMessage(null), 1600);
    }
  }

  return (
    <section className="grid gap-4">
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        <StatusCard
          label="LLM"
          value={llmStatus?.enabled ? "Enabled" : "Disabled"}
          detail={
            llmStatus
              ? `${llmStatus.provider}:${llmStatus.model}`
              : "status unavailable"
          }
          tone={llmStatus?.enabled ? "ok" : "warn"}
        />
        <StatusCard
          label="Multi-Agent"
          value={data.config?.multiagent_enabled ? "Enabled" : "Disabled"}
          detail={`max proposals ${data.config?.multiagent_max_proposals ?? "-"}`}
          tone={data.config?.multiagent_enabled ? "ok" : "warn"}
        />
        <StatusCard
          label="Auto-Approve"
          value={autoApprove ? "On" : "Off"}
          detail="AI proposals auto-apply"
          tone={autoApprove ? "ok" : "info"}
        />
        <StatusCard
          label="Emergency AI"
          value={emergencyEnabled ? "On" : "Off"}
          detail={`max ${emergencyMax}`}
          tone={emergencyEnabled ? "ok" : "warn"}
        />
        <StatusCard
          label="Pending Queue"
          value={String(approvalsPending.length)}
          detail={approvalsPending.length > 0 ? "waiting approvals" : "queue clear"}
          tone={approvalsPending.length > 0 ? "warn" : "ok"}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <Panel title="Emergency Supervisor" subtitle="AI analysis for each emergency stop">
          <div className="grid gap-3 text-xs md:grid-cols-2">
            <div className="rounded-lg border border-white/10 bg-black/20 p-3">
              <p className="text-slate-300">Last analysis event</p>
              <p className="mt-2 font-heading text-base text-white">
                {data.emergencyAnalysis?.found
                  ? data.emergencyAnalysis.event_type
                  : "No emergency analysis yet"}
              </p>
              <p className="mt-1 text-slate-300">{data.emergencyAnalysis?.summary ?? "-"}</p>
              <p className="mt-1 text-slate-400">{formatDateTime(data.emergencyAnalysis?.created_at ?? null)}</p>
            </div>
            <div className="rounded-lg border border-white/10 bg-black/20 p-3">
              <p className="text-slate-300">Runtime settings</p>
              <div className="mt-2 flex items-center gap-2">
                <button
                  className={`rounded border px-2 py-1 text-[11px] ${
                    emergencyEnabled
                      ? "border-amber-300/50 bg-amber-500/10 text-amber-100"
                      : "border-emerald-300/50 bg-emerald-500/10 text-emerald-100"
                  }`}
                  disabled={!canOperate || runningCheck}
                  onClick={() =>
                    void runAction(
                      `${emergencyEnabled ? "Disabling" : "Enabling"} emergency AI...`,
                      () =>
                        api.setEmergencyAISettings({
                          enabled: !emergencyEnabled,
                          reason: "ai_center_toggle",
                        }).then(() => undefined),
                    )
                  }
                >
                  {emergencyEnabled ? "Disable" : "Enable"}
                </button>
                <label className="text-[11px] text-slate-300" htmlFor="max-proposals">
                  Max proposals
                </label>
                <input
                  id="max-proposals"
                  type="number"
                  min={1}
                  max={10}
                  value={targetMaxProposals}
                  className="w-16 rounded border border-white/20 bg-slate-900/80 px-2 py-1 text-[11px] text-slate-100"
                  onChange={(event) => setTargetMaxProposals(Number(event.target.value))}
                />
                <button
                  className="rounded border border-cyan-300/60 bg-cyan-500/10 px-2 py-1 text-[11px] text-cyan-100"
                  disabled={!canOperate || runningCheck}
                  onClick={() =>
                    void runAction("Updating emergency AI proposal limit...", () =>
                      api
                        .setEmergencyAISettings({
                          max_proposals: Math.max(1, Math.min(10, targetMaxProposals)),
                          reason: "ai_center_max_update",
                        })
                        .then(() => undefined),
                    )
                  }
                >
                  Apply
                </button>
              </div>
              <button
                className="mt-3 rounded border border-fuchsia-300/60 bg-fuchsia-500/10 px-2 py-1 text-[11px] text-fuchsia-100"
                disabled={!canOperate || runningCheck}
                onClick={() =>
                  void runAction("Running emergency analysis now...", () =>
                    api
                      .analyzeEmergencyNow({
                        reason: data.system?.reason ?? "manual_run",
                        source: "ai_center_manual",
                        metadata: {
                          state: data.system?.state ?? "unknown",
                        },
                      })
                      .then(() => undefined),
                  )
                }
              >
                Analyze Current Incident
              </button>
            </div>
          </div>
          {message ? <p className="mt-3 text-xs text-mint">{message}</p> : null}
          {error ? <p className="mt-3 text-xs text-rose-200">{error}</p> : null}
        </Panel>

        <Panel title="Agent Diagnostics" subtitle="Runtime AI health and orchestration">
          <div className="space-y-2 text-xs text-slate-200">
            <p>LLM enabled: {String(llmStatus?.enabled ?? false)}</p>
            <p>Primary model: {llmStatus ? `${llmStatus.provider}:${llmStatus.model}` : "-"}</p>
            <p>
              Fallback model: {llmStatus?.fallback_enabled ? `${llmStatus.fallback_provider}:${llmStatus.fallback_model}` : "disabled"}
            </p>
            <p>
              Last model used:{" "}
              {llmStatus?.last_provider_used && llmStatus?.last_model_used
                ? `${llmStatus.last_provider_used}:${llmStatus.last_model_used}`
                : "-"}
            </p>
            <p>Last call time: {formatDateTime(llmStatus?.last_used_at ?? null)}</p>
            <p>Last call used fallback: {String(llmStatus?.last_fallback_used ?? false)}</p>
            <p>Multi-agent max proposals: {data.config?.multiagent_max_proposals ?? "-"}</p>
            <p>Multi-agent min confidence: {formatNumber(data.config?.multiagent_min_confidence ?? null, 2)}</p>
            <p>Meta-agent enabled: {String(data.config?.multiagent_meta_agent_enabled ?? false)}</p>
            <p>Auto-approve enabled: {String(autoApprove)}</p>
            <p>Approval timeout (hours): {data.config?.approval_timeout_hours ?? "-"}</p>
            {llmStatus?.last_error ? (
              <p className="rounded border border-amber-300/30 bg-amber-500/10 p-2 text-amber-100">
                Last LLM error: {llmStatus.last_error}
              </p>
            ) : null}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              className="rounded border border-indigo-300/60 bg-indigo-500/10 px-2 py-1 text-[11px] text-indigo-100"
              disabled={!canOperate || runningCheck}
              onClick={() => void runAction("Testing multi-agent...", () => api.testMultiAgent().then(() => undefined))}
            >
              Test Multi-Agent
            </button>
            <button
              className="rounded border border-sky-300/60 bg-sky-500/10 px-2 py-1 text-[11px] text-sky-100"
              disabled={!canOperate || runningCheck}
              onClick={() =>
                void runAction("Generating AI proposals...", () => api.generateProposals().then(() => undefined))
              }
            >
              Generate Proposals
            </button>
            <button
              className="rounded border border-emerald-300/60 bg-emerald-500/10 px-2 py-1 text-[11px] text-emerald-100"
              disabled={!canOperate || runningCheck}
              onClick={() =>
                void runAction("Analyzing strategy improvements...", () =>
                  api.getStrategyAnalysis().then((payload) => {
                    setStrategyAnalysis(payload);
                  }),
                )
              }
            >
              Analyze Strategy
            </button>
          </div>
          <div className="mt-3 rounded-lg border border-white/10 bg-black/20 p-3 text-xs text-slate-200">
            <p className="text-slate-300">
              {strategyAnalysis
                ? `Strategy ${strategyAnalysis.active_strategy} (${strategyAnalysis.symbol})`
                : "Run strategy analysis to get improvement recommendations."}
            </p>
            {strategyAnalysis ? (
              <>
                <p className="mt-1 text-slate-400">Generated {formatDateTime(strategyAnalysis.generated_at)}</p>
                <p className="mt-1 text-slate-300">
                  Recommendations: {strategyAnalysis.recommendations.length}
                </p>
                <div className="mt-2 space-y-2">
                  {strategyAnalysis.recommendations.slice(0, 3).map((rec) => (
                    <article key={`${rec.proposal_type}:${rec.title}`} className="rounded border border-white/10 bg-black/20 p-2">
                      <p className="font-heading text-white">{rec.title}</p>
                      <p className="text-slate-300">{rec.description}</p>
                      <p className="text-[11px] text-slate-400">
                        {rec.proposal_type} · confidence {formatNumber(rec.confidence * 100, 1)}%
                      </p>
                    </article>
                  ))}
                </div>
              </>
            ) : null}
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.15fr_1fr]">
        <Panel title="Pending Approval Queue" subtitle="Latest AI proposals awaiting review">
          <div className="max-h-[360px] space-y-2 overflow-auto">
            {approvalsPending.map((approval) => (
              <article key={approval.id} className="rounded-lg border border-white/10 bg-black/20 p-3 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-heading text-sm text-white">
                    #{approval.id} {approval.title}
                  </p>
                  <p className="text-slate-400">{formatNumber(approval.confidence * 100, 1)}%</p>
                </div>
                <p className="mt-1 text-slate-300">{approval.description}</p>
                <p className="mt-1 text-[11px] text-slate-400">expires {formatDateTime(approval.expires_at)}</p>
              </article>
            ))}
            {approvalsPending.length === 0 ? (
              <p className="text-xs text-slate-400">No pending approvals.</p>
            ) : null}
          </div>
        </Panel>

        <Panel title="AI Event Feed" subtitle="Recent AI and emergency-audit events">
          <div className="max-h-[360px] space-y-2 overflow-auto">
            {aiEvents.map((event) => (
              <article key={event.id} className="rounded-lg border border-white/10 bg-black/20 p-2 text-xs text-slate-200">
                <p className="text-[11px] uppercase tracking-[0.14em] text-slate-300">
                  {event.event_category}/{event.event_type}
                </p>
                <p className="mt-1">{event.summary}</p>
                <p className="mt-1 text-[11px] text-slate-400">
                  {formatDateTime(event.created_at)} by {event.actor}
                </p>
              </article>
            ))}
            {aiEvents.length === 0 ? <p className="text-xs text-slate-400">No AI events yet.</p> : null}
          </div>
        </Panel>
      </div>
    </section>
  );
}
