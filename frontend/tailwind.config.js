/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        void: "var(--bp-bg-void)",
        panel: "var(--bp-bg-panel)",
        "panel-raised": "var(--bp-bg-panel-raised)",
        line: "var(--bp-line)",
        primary: "var(--bp-text-primary)",
        muted: "var(--bp-text-muted)",
        signal: "var(--bp-accent-signal)",
        uncertainty: "var(--bp-accent-uncertainty)",
        critical: "var(--bp-accent-critical)",
      },
      fontFamily: {
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
        sans: ["'IBM Plex Sans'", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      backgroundImage: {
        grid: "linear-gradient(var(--bp-line) 1px, transparent 1px), linear-gradient(90deg, var(--bp-line) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "24px 24px",
      },
    },
  },
  plugins: [],
};
