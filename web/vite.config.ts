import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is written into the Python package, so end users never need Node.
// In development, `npm run dev` serves the app and proxies the API to `logogram serve --dev`.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../src/logogram/web_dist",
    emptyOutDir: true,
    sourcemap: false,
    assetsInlineLimit: 0,
    modulePreload: { polyfill: false },
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8765", changeOrigin: false },
      "/ws": { target: "ws://127.0.0.1:8765", ws: true, changeOrigin: false },
    },
  },
});
