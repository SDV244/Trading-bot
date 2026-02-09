import { useState, type FormEvent } from "react";
import { Navigate } from "react-router-dom";

import { useAuth } from "../auth";

export function LoginPage() {
  const { authEnabled, user, login, loading } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (loading) {
    return <div className="p-6 text-sm text-slate-200">Loading session...</div>;
  }

  if (!authEnabled || user) {
    return <Navigate to="/" replace />;
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(username, password);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-ink via-sea to-[#102736] px-4">
      <form
        onSubmit={onSubmit}
        className="w-full max-w-sm rounded-2xl border border-white/10 bg-white/5 p-6 shadow-panel backdrop-blur"
      >
        <h1 className="font-heading text-2xl font-bold text-white">Sign In</h1>
        <p className="mt-2 text-xs text-slate-300">Authenticate to access trading controls and approvals.</p>

        {error ? (
          <div className="mt-4 rounded-lg border border-rose-400/40 bg-rose-900/20 p-2 text-xs text-rose-200">
            {error}
          </div>
        ) : null}

        <label className="mt-4 block text-xs uppercase tracking-[0.18em] text-slate-300">Username</label>
        <input
          className="mt-1 w-full rounded-lg border border-white/20 bg-black/20 px-3 py-2 text-sm text-white outline-none focus:border-mint/60"
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          autoComplete="username"
          required
        />

        <label className="mt-4 block text-xs uppercase tracking-[0.18em] text-slate-300">Password</label>
        <input
          type="password"
          className="mt-1 w-full rounded-lg border border-white/20 bg-black/20 px-3 py-2 text-sm text-white outline-none focus:border-mint/60"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          autoComplete="current-password"
          required
        />

        <button
          type="submit"
          disabled={submitting}
          className="mt-5 w-full rounded-lg border border-mint/50 bg-mint/15 px-3 py-2 text-sm text-mint disabled:opacity-50"
        >
          {submitting ? "Signing in..." : "Sign In"}
        </button>
      </form>
    </div>
  );
}
