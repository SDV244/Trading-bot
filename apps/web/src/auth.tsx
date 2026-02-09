import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { ApiError, api, clearAccessToken, setAccessToken, type AuthStatus, type Role } from "./api";

type AuthContextValue = {
  loading: boolean;
  authEnabled: boolean;
  user: AuthStatus | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshProfile: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const ROLE_ORDER: Record<Role, number> = {
  viewer: 0,
  operator: 1,
  admin: 2,
};

export function hasMinRole(role: string | undefined, minRole: Role): boolean {
  if (!role) {
    return false;
  }
  const current = ROLE_ORDER[role as Role];
  return current >= ROLE_ORDER[minRole];
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<AuthStatus | null>(null);
  const [authEnabled, setAuthEnabled] = useState(false);

  const refreshProfile = useCallback(async () => {
    try {
      const profile = await api.me();
      setUser(profile);
      setAuthEnabled(profile.auth_enabled);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        clearAccessToken();
        setUser(null);
        setAuthEnabled(true);
      } else {
        throw error;
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshProfile();
  }, [refreshProfile]);

  const login = useCallback(
    async (username: string, password: string) => {
      const response = await api.login(username, password);
      setAccessToken(response.access_token);
      await refreshProfile();
    },
    [refreshProfile],
  );

  const logout = useCallback(() => {
    clearAccessToken();
    if (authEnabled) {
      setUser(null);
    }
  }, [authEnabled]);

  const value = useMemo<AuthContextValue>(
    () => ({
      loading,
      user,
      authEnabled,
      login,
      logout,
      refreshProfile,
    }),
    [authEnabled, loading, login, logout, refreshProfile, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
}

