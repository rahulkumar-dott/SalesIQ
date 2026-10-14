import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev the UI calls /api/*, proxied to the FastAPI server (python api.py) with the prefix removed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
