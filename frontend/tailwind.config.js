/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: '#090d16',
        darkCard: '#121824',
        darkCardHover: '#1a2233',
        darkBorder: '#1f293d',
        accentCyan: '#00f2fe',
        accentGreen: '#00e676',
        accentAmber: '#ffb300',
        accentRose: '#ff1744'
      },
      fontFamily: {
        sans: ['Outfit', 'sans-serif']
      }
    }
  },
  plugins: []
}
