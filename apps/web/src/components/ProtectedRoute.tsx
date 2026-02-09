import { Navigate } from "react-router-dom";

import { hasMinRole, useAuth } from "../auth";
import type { Role } from "../api";

type ProtectedRouteProps = {
  minRole?: Role;
  children: JSX.Element;
};

export function ProtectedRoute({ minRole = "viewer", children }: ProtectedRouteProps) {
  const { loading, user, authEnabled } = useAuth();

  if (loading) {
    return (
      <div className="rounded-xl border border-white/10 bg-white/5 p-6 text-sm text-slate-200">
        Loading session...
      </div>
    );
  }

  if (!authEnabled) {
    return children;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (!hasMinRole(user.role, minRole)) {
    return (
      <div className="rounded-xl border border-rose-300/30 bg-rose-900/20 p-6 text-sm text-rose-200">
        You do not have permission to access this section.
      </div>
    );
  }

  return children;
}

