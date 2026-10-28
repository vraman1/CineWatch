import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev (npm run dev) API and media requests are proxied to the Flask backend on port 5000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:5000", "/media": "http://127.0.0.1:5000" },
  },
});
