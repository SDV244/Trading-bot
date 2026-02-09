import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Monorepo: load VITE_* vars from repository root .env/.env.local files.
  envDir: "../../",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
});
