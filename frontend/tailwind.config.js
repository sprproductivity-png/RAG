/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#070b14',
          900: '#0b0f19',
          800: '#0f172a',
          700: '#1e293b'
        }
      }
    },
  },
  plugins: [],
}

