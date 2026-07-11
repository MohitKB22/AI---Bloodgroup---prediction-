import axios from "axios";

const client = axios.create({
  baseURL: "/api/v1",
  timeout: 60000,
});

function getStoredTokens() {
  const raw = localStorage.getItem("bp_tokens");
  return raw ? JSON.parse(raw) : null;
}

function setStoredTokens(tokens) {
  if (tokens) localStorage.setItem("bp_tokens", JSON.stringify(tokens));
  else localStorage.removeItem("bp_tokens");
}

client.interceptors.request.use((config) => {
  const tokens = getStoredTokens();
  if (tokens?.access_token) {
    config.headers.Authorization = `Bearer ${tokens.access_token}`;
  }
  return config;
});

let refreshPromise = null;

client.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;
    if (error.response?.status !== 401 || original._retried) {
      return Promise.reject(error);
    }

    const tokens = getStoredTokens();
    if (!tokens?.refresh_token) {
      setStoredTokens(null);
      window.dispatchEvent(new CustomEvent("bp:auth-expired"));
      return Promise.reject(error);
    }

    original._retried = true;
    try {
      // Coalesce concurrent 401s into a single refresh call.
      refreshPromise ??= axios.post("/api/v1/auth/refresh", { refresh_token: tokens.refresh_token });
      const { data } = await refreshPromise;
      setStoredTokens(data);
      original.headers.Authorization = `Bearer ${data.access_token}`;
      return client(original);
    } catch (refreshError) {
      setStoredTokens(null);
      window.dispatchEvent(new CustomEvent("bp:auth-expired"));
      return Promise.reject(refreshError);
    } finally {
      refreshPromise = null;
    }
  }
);

export const authApi = {
  register: (payload) => client.post("/auth/register", payload),
  login: (payload) => client.post("/auth/login", payload).then((res) => {
    setStoredTokens(res.data);
    return res.data;
  }),
  logout: () => setStoredTokens(null),
  me: () => client.get("/auth/me").then((res) => res.data),
  isAuthenticated: () => Boolean(getStoredTokens()?.access_token),
};

export const predictionsApi = {
  create: (file, { useTta = true, explain = true } = {}) => {
    const form = new FormData();
    form.append("file", file);
    return client
      .post(`/predictions?use_tta=${useTta}&explain=${explain}`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((res) => res.data);
  },
  list: (page = 1, pageSize = 20) =>
    client.get("/predictions", { params: { page, page_size: pageSize } }).then((res) => res.data),
  get: (id) => client.get(`/predictions/${id}`).then((res) => res.data),
  reportUrl: (id) => `/api/v1/predictions/${id}/report.pdf`,
};

export const healthApi = {
  ready: () => client.get("/health/ready").then((res) => res.data),
};

export default client;
