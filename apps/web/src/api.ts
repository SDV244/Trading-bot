export type SystemState = {
  state: string;
  reason: string;
  changed_at: string;
  changed_by: string;
  can_trade: boolean;
};

export type SchedulerStatus = {
  running: boolean;
  interval_seconds: number;
  last_run_at: string | null;
  last_error: string | null;
  last_result: Record<string, unknown> | null;
};

export type Position = {
  symbol: string;
  side: string | null;
  quantity: string;
  avg_entry_price: string;
  unrealized_pnl: string;
  realized_pnl: string;
  total_fees: string;
  is_paper: boolean;
};

export type Metrics = {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number | null;
  total_pnl: string;
  total_fees: string;
  max_drawdown: number;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  profit_factor: number | null;
};

export type TradingConfig = {
  trading_pair: string;
  timeframes: string[];
  live_mode: boolean;
  risk_per_trade: number;
  max_daily_loss: number;
  max_exposure: number;
  fee_bps: number;
  slippage_bps: number;
  approval_timeout_hours: number;
};

export type Order = {
  id: number;
  client_order_id: string;
  symbol: string;
  side: string;
  order_type: string;
  quantity: string;
  price: string | null;
  status: string;
  is_paper: boolean;
  strategy_name: string;
  signal_reason: string | null;
  created_at: string;
};

export type Fill = {
  id: number;
  order_id: number;
  fill_id: string;
  quantity: string;
  price: string;
  fee: string;
  fee_asset: string;
  is_paper: boolean;
  slippage_bps: number | null;
  filled_at: string;
};

export type EquityPoint = {
  timestamp: string;
  equity: string;
  available_balance: string;
  unrealized_pnl: string;
};

export type PaperCycleResult = {
  symbol: string;
  signal_side: string;
  signal_reason: string;
  risk_action: string;
  risk_reason: string;
  executed: boolean;
  order_id: number | null;
  fill_id: string | null;
  quantity: string;
  price: string | null;
};

export type LiveOrderResponse = {
  accepted: boolean;
  reason: string;
  order_id: string | null;
  quantity: string | null;
  price: string | null;
};

export type Approval = {
  id: number;
  proposal_type: string;
  title: string;
  description: string;
  diff: Record<string, unknown>;
  expected_impact: string | null;
  evidence: Record<string, unknown> | null;
  confidence: number;
  status: string;
  ttl_hours: number;
  expires_at: string;
  decided_by: string | null;
  decided_at: string | null;
  created_at: string;
};

export type AuditEvent = {
  id: number;
  event_type: string;
  event_category: string;
  summary: string;
  details: Record<string, unknown> | null;
  inputs: Record<string, unknown> | null;
  config_version: number;
  actor: string;
  created_at: string;
};

export type LoginResponse = {
  access_token: string;
  token_type: string;
  expires_at: string;
  role: string;
  username: string;
};

export type AuthStatus = {
  auth_enabled: boolean;
  username: string;
  role: string;
};

export type Role = "viewer" | "operator" | "admin";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const TOKEN_KEY = "tb_access_token";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getAccessToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAccessToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAccessToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  headers.set("Content-Type", "application/json");
  const token = getAccessToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(`${response.status} ${response.statusText}: ${body}`, response.status);
  }

  return (await response.json()) as T;
}

function buildQuery(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") {
      search.set(key, String(value));
    }
  });
  const query = search.toString();
  return query.length > 0 ? `?${query}` : "";
}

export const api = {
  health: () => fetchJson<{ status: string; timestamp: string; version: string }>("/health"),
  ready: () => fetchJson<{ ready: boolean; database: boolean; binance: boolean }>("/ready"),
  login: (username: string, password: string) =>
    fetchJson<LoginResponse>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  me: () => fetchJson<AuthStatus>("/api/auth/me"),
  getSystemState: () => fetchJson<SystemState>("/api/system/state"),
  setSystemState: (action: "pause" | "resume" | "emergency_stop" | "manual_resume", reason: string) =>
    fetchJson<SystemState>("/api/system/state", {
      method: "POST",
      body: JSON.stringify({ action, reason, changed_by: "web_dashboard" }),
    }),
  getScheduler: () => fetchJson<SchedulerStatus>("/api/system/scheduler"),
  startScheduler: (intervalSeconds: number) =>
    fetchJson<SchedulerStatus>(`/api/system/scheduler/start?interval_seconds=${intervalSeconds}`, {
      method: "POST",
    }),
  stopScheduler: () =>
    fetchJson<SchedulerStatus>("/api/system/scheduler/stop", {
      method: "POST",
    }),
  runPaperCycle: () =>
    fetchJson<PaperCycleResult>("/api/trading/paper/cycle", {
      method: "POST",
    }),
  getTradingConfig: () => fetchJson<TradingConfig>("/api/trading/config"),
  getPosition: () => fetchJson<Position>("/api/trading/position"),
  getMetrics: () => fetchJson<Metrics>("/api/trading/metrics"),
  getOrders: (limit = 10, status?: string) =>
    fetchJson<Order[]>(`/api/trading/orders${buildQuery({ limit, status })}`),
  getFills: (limit = 10) => fetchJson<Fill[]>(`/api/trading/fills${buildQuery({ limit })}`),
  getEquityHistory: (days = 30) =>
    fetchJson<EquityPoint[]>(`/api/trading/equity/history${buildQuery({ days })}`),
  listApprovals: (status?: string, limit = 100) =>
    fetchJson<Approval[]>(`/api/ai/approvals${buildQuery({ status, limit })}`),
  generateProposals: () =>
    fetchJson<Approval[]>("/api/ai/proposals/generate", {
      method: "POST",
    }),
  approveProposal: (id: number, decidedBy: string) =>
    fetchJson<Approval>(`/api/ai/approvals/${id}/approve`, {
      method: "POST",
      body: JSON.stringify({ decided_by: decidedBy, reason: "approved_from_dashboard" }),
    }),
  rejectProposal: (id: number, decidedBy: string) =>
    fetchJson<Approval>(`/api/ai/approvals/${id}/reject`, {
      method: "POST",
      body: JSON.stringify({ decided_by: decidedBy, reason: "rejected_from_dashboard" }),
    }),
  expireApprovals: () =>
    fetchJson<{ expired_count: number }>("/api/ai/approvals/expire-check", {
      method: "POST",
    }),
  listEvents: (filters?: {
    category?: string;
    event_type?: string;
    actor?: string;
    search?: string;
    start_at?: string;
    end_at?: string;
    limit?: number;
  }) => fetchJson<AuditEvent[]>(`/api/ai/events${buildQuery(filters ?? {})}`),
  trainOptimizer: (timesteps = 1024) =>
    fetchJson<{ trained: boolean; data_points: number; timesteps: number }>("/api/ai/optimizer/train", {
      method: "POST",
      body: JSON.stringify({ timesteps }),
    }),
  getOptimizerProposal: () =>
    fetchJson<{
      title: string;
      diff: Record<string, unknown>;
      expected_impact: string;
      evidence: Record<string, unknown>;
      confidence: number;
    }>("/api/ai/optimizer/propose"),
  submitLiveOrder: (payload: {
    side: "BUY" | "SELL";
    quantity: string;
    ui_confirmed: boolean;
    reauthenticated: boolean;
    safety_acknowledged: boolean;
    reason: string;
  }) =>
    fetchJson<LiveOrderResponse>("/api/trading/live/order", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

