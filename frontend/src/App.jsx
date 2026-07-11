import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { AuthProvider, useAuth } from "./context/AuthContext";
import { ThemeProvider } from "./context/ThemeContext";
import DisclaimerBanner from "./components/DisclaimerBanner";
import Navbar from "./components/Navbar";

import DashboardPage from "./pages/DashboardPage";
import HistoryPage from "./pages/HistoryPage";
import PredictionDetailPage from "./pages/PredictionDetailPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import AboutPage from "./pages/AboutPage";

// NOTE: LoginPage, RegisterPage, and ProtectedRoute are intentionally kept
// in src/pages and src/components but not routed here -- see the "Future
// work" section in README.md for how to re-enable a visible login screen.

function AppShell({ children }) {
  return (
    <div className="min-h-screen bg-void text-primary">
      <DisclaimerBanner />
      <Navbar />
      <main>{children}</main>
    </div>
  );
}

function AppRoutes() {
  const { loading, error } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-void font-mono text-sm text-muted">
        Setting up your session&hellip;
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-2 bg-void px-4 text-center">
        <p className="font-mono text-sm text-critical">{error}</p>
        <p className="text-xs text-muted">
          Start the backend (see README) and reload this page.
        </p>
      </div>
    );
  }

  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:id" element={<PredictionDetailPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}
