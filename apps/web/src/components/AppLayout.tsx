import { NavLink, Outlet } from "react-router-dom";

import { useAuth } from "../auth";
import { useDashboard } from "../dashboard";
import { StateBadge } from "./Badge";

const NAV_ITEMS = [
  { to: "/", label: "Home" },
  { to: "/trading", label: "Trading" },
  { to: "/config", label: "Config" },
  { to: "/approvals", label: "AI Approvals" },
  { to: "/logs", label: "Logs" },
  { to: "/controls", label: "Controls" },
];

export function AppLayout() {
  const { user, authEnabled, logout } = useAuth();
  const { data, error } = useDashboard();

  return (
    <div className="min-h-screen bg-gradient-to-br from-ink via-sea to-[#102736] text-slate">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-24 top-12 h-72 w-72 rounded-full bg-mint/20 blur-3xl" />
        <div className="absolute right-0 top-36 h-80 w-80 rounded-full bg-ember/15 blur-3xl" />
      </div>

      <main className="relative mx-auto max-w-7xl px-4 py-8 md:px-8">
        <header className="mb-6 animate-rise">
          <p className="font-body text-[11px] uppercase tracking-[0.32em] text-mint/80">spot btcusdt control deck</p>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="font-heading text-3xl font-black tracking-tight text-white md:text-4xl">
                Trading Bot Dashboard
              </h1>
              {data.system ? <StateBadge value={data.system.state} /> : null}
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="rounded-md border border-white/20 bg-white/5 px-2 py-1">
                {authEnabled ? `Role: ${user?.role ?? "guest"}` : "Auth: disabled"}
              </span>
              {authEnabled ? (
                <button
                  className="rounded-md border border-rose-300/60 bg-rose-500/10 px-2 py-1 text-rose-200"
                  onClick={logout}
                >
                  Logout
                </button>
              ) : null}
            </div>
          </div>
        </header>

        <nav className="mb-6 flex flex-wrap gap-2 rounded-xl border border-white/10 bg-white/5 p-2 backdrop-blur">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-xs uppercase tracking-[0.18em] ${
                  isActive ? "bg-mint/20 text-mint" : "text-slate-300 hover:bg-white/10"
                }`
              }
              to={item.to}
              end={item.to === "/"}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        {error ? (
          <div className="mb-5 rounded-xl border border-rose-400/40 bg-rose-900/20 p-3 font-body text-xs text-rose-200">
            {error}
          </div>
        ) : null}

        <Outlet />
      </main>
    </div>
  );
}

