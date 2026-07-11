import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { authApi } from "../api/client";

const AuthContext = createContext(null);

/**
 * NOTE on auth: the backend fully implements and enforces JWT auth
 * (register/login/refresh, tested in backend/tests/test_auth.py). What's
 * disabled here is only the *manual* login/register UI -- on first load,
 * if there's no stored session yet, this silently provisions a random
 * "guest" account so the demo has zero signup friction. The account is
 * real (rows in the users/predictions tables, a real JWT), just never
 * shown to the person using the app. Re-enabling a visible login screen
 * is a matter of restoring <ProtectedRoute> and the /login, /register
 * routes in App.jsx -- see "Future work" in the README.
 */
async function provisionAnonymousSession() {
  const id = crypto.randomUUID();
  // .example is reserved by RFC 2606 specifically for placeholder addresses
  // that will never be real mailboxes -- unlike .local/.invalid, it passes
  // the backend's email-validator syntax check (verified directly against
  // it, not assumed).
  const email = `guest-${id}@bloodprint.example`;
  const password = `${crypto.randomUUID()}${crypto.randomUUID()}`;
  await authApi.register({ email, password, full_name: "Guest" });
  await authApi.login({ email, password });
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refreshUser = useCallback(async () => {
    try {
      const me = await authApi.me();
      setUser(me);
    } catch {
      setUser(null);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      setLoading(true);
      setError(null);
      try {
        if (!authApi.isAuthenticated()) {
          await provisionAnonymousSession();
        }
        if (!cancelled) await refreshUser();
      } catch {
        if (!cancelled) setError("Couldn't reach the BloodPrint API. Confirm the backend is running.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    bootstrap();
    const onExpired = () => bootstrap();
    window.addEventListener("bp:auth-expired", onExpired);
    return () => {
      cancelled = true;
      window.removeEventListener("bp:auth-expired", onExpired);
    };
  }, [refreshUser]);

  const login = useCallback(async (email, password) => {
    await authApi.login({ email, password });
    await refreshUser();
  }, [refreshUser]);

  const register = useCallback(async (email, password, fullName) => {
    await authApi.register({ email, password, full_name: fullName });
    await login(email, password);
  }, [login]);

  const logout = useCallback(() => {
    authApi.logout();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, error, login, register, logout, isAuthenticated: Boolean(user) }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
