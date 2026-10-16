import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

// Kept separate from vite.config.ts so the dev/build config stays free of test-only settings.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    // CSS is stubbed in tests, except the theme tokens: theme-contrast.test.ts reads them to check text contrast.
    css: { include: [/styles\/globals\.css/] },
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
