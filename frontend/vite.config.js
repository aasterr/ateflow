import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // relative asset paths: the same build works at / (vite preview) and /ateflow/ (GitHub Pages)
  base: "./",
  worker: { format: "es" },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
