import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // relative asset paths: the same build works at / (local server) and /ateflow/ (GitHub Pages)
  base: "./",
  worker: { format: "es" },
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: {
    outDir: "../ateflow/static",
    emptyOutDir: true,
  },
});
