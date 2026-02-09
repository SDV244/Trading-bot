import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AuthProvider, useAuth } from "./auth";
import { AppLayout } from "./components/AppLayout";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { DashboardProvider } from "./dashboard";
import { ApprovalsPage } from "./pages/ApprovalsPage";
import { ConfigPage } from "./pages/ConfigPage";
import { ControlsPage } from "./pages/ControlsPage";
import { HomePage } from "./pages/HomePage";
import { LoginPage } from "./pages/LoginPage";
import { LogsPage } from "./pages/LogsPage";
import { TradingPage } from "./pages/TradingPage";

function AppRoutes() {
  const { authEnabled, user } = useAuth();
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/"
        element={
          <ProtectedRoute minRole="viewer">
            <DashboardProvider>
              <AppLayout />
            </DashboardProvider>
          </ProtectedRoute>
        }
      >
        <Route index element={<HomePage />} />
        <Route path="trading" element={<TradingPage />} />
        <Route path="config" element={<ConfigPage />} />
        <Route
          path="approvals"
          element={
            <ProtectedRoute minRole="operator">
              <ApprovalsPage />
            </ProtectedRoute>
          }
        />
        <Route path="logs" element={<LogsPage />} />
        <Route
          path="controls"
          element={
            <ProtectedRoute minRole="operator">
              <ControlsPage />
            </ProtectedRoute>
          }
        />
      </Route>
      <Route
        path="*"
        element={<Navigate to={authEnabled && !user ? "/login" : "/"} replace />}
      />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  );
}
