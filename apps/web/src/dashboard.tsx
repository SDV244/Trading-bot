import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import {
  api,
  type Approval,
  type AuditEvent,
  type EquityPoint,
  type Fill,
  type GridPreview,
  type MarketCandle,
  type MarketPrice,
  type Metrics,
  type Order,
  type PaperCycleResult,
  type Position,
  type SchedulerStatus,
  type SystemReadiness,
  type SystemState,
  type TradingConfig,
} from "./api";

type DashboardData = {
  health: { status: string; timestamp: string; version: string } | null;
  ready: { ready: boolean; database: boolean; binance: boolean } | null;
  lastUpdatedAt: string | null;
  systemReadiness: SystemReadiness | null;
  system: SystemState | null;
  scheduler: SchedulerStatus | null;
  config: TradingConfig | null;
  position: Position | null;
  metrics: Metrics | null;
  orders: Order[];
  fills: Fill[];
  equity: EquityPoint[];
  approvals: Approval[];
  events: AuditEvent[];
  marketCandles: MarketCandle[];
  marketPrice: MarketPrice | null;
  gridPreview: GridPreview | null;
  lastCycle: PaperCycleResult | null;
};

type DashboardContextValue = {
  data: DashboardData;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  runPaperCycle: () => Promise<void>;
};

const DashboardContext = createContext<DashboardContextValue | null>(null);
const REFRESH_MS = 5000;

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<DashboardData>({
    health: null,
    ready: null,
    lastUpdatedAt: null,
    systemReadiness: null,
    system: null,
    scheduler: null,
    config: null,
    position: null,
    metrics: null,
    orders: [],
    fills: [],
    equity: [],
    approvals: [],
    events: [],
    marketCandles: [],
    marketPrice: null,
    gridPreview: null,
    lastCycle: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const config = await api.getTradingConfig();
      const symbol = config.trading_pair;
      const [health, ready, systemReadiness, system, scheduler, position, metrics, orders, fills, equity, approvals, events, marketCandles, marketPrice, gridPreview] =
        await Promise.all([
          api.health(),
          api.ready(),
          api.getSystemReadiness(),
          api.getSystemState(),
          api.getScheduler(),
          api.getPosition(),
          api.getMetrics(),
          api.getOrders(8),
          api.getFills(8),
          api.getEquityHistory(30),
          api.listApprovals(undefined, 50),
          api.listEvents({ limit: 20 }),
          api.getMarketCandles(symbol, "1h", 120).catch(() => []),
          api.getMarketPrice(symbol).catch(() => null),
          api.getGridPreview().catch(() => null),
        ]);
      setData((prev) => ({
        ...prev,
        health,
        ready,
        lastUpdatedAt: new Date().toISOString(),
        systemReadiness,
        system,
        scheduler,
        config,
        position,
        metrics,
        orders,
        fills,
        equity,
        approvals,
        events,
        marketCandles,
        marketPrice,
        gridPreview,
      }));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch dashboard data");
    } finally {
      setLoading(false);
    }
  }, []);

  const runPaperCycle = useCallback(async () => {
    const cycle = await api.runPaperCycle();
    setData((prev) => ({ ...prev, lastCycle: cycle }));
    await refresh();
  }, [refresh]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      void refresh();
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const value = useMemo<DashboardContextValue>(
    () => ({
      data,
      loading,
      error,
      refresh,
      runPaperCycle,
    }),
    [data, error, loading, refresh, runPaperCycle],
  );

  return <DashboardContext.Provider value={value}>{children}</DashboardContext.Provider>;
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error("useDashboard must be used inside DashboardProvider");
  }
  return context;
}
