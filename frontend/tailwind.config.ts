import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-be-vietnam)", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
