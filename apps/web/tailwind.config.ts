import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#081321",
        sea: "#0f3042",
        mint: "#2dd4bf",
        ember: "#fb923c",
        slate: "#d0d8e5",
      },
      fontFamily: {
        heading: ["Chivo", "system-ui", "sans-serif"],
        body: ["IBM Plex Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        panel: "0 16px 40px rgba(8, 19, 33, 0.35)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        rise: "rise 450ms ease-out both",
      },
    },
  },
  plugins: [],
};

export default config;
