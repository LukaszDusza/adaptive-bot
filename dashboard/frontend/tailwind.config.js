/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // Enable dark mode with class strategy
  theme: {
    extend: {
      colors: {
        // Dark theme color palette (inspired by nof1.ai)
        dark: {
          bg: '#0a0a0f',
          card: '#111118',
          border: '#1f1f28',
          text: {
            primary: '#e8e8ea',
            secondary: '#a0a0ab',
            muted: '#6b6b76'
          }
        },
        // Trading colors
        profit: {
          light: '#10b981',
          DEFAULT: '#059669',
          dark: '#047857'
        },
        loss: {
          light: '#ef4444',
          DEFAULT: '#dc2626',
          dark: '#b91c1c'
        },
        warning: {
          light: '#f59e0b',
          DEFAULT: '#d97706',
          dark: '#b45309'
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace']
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
