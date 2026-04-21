/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        cherry: "#4B0F1A",
        wine: "#29070E",
        merlot: "#3B0C17",
        ivory: "#F7F0E7",
        gold: "#C9A76A"
      },
      fontFamily: {
        display: ["Cormorant Garamond", "serif"],
        body: ["Plus Jakarta Sans", "sans-serif"]
      },
      boxShadow: {
        glow: "0 0 30px rgba(201, 167, 106, 0.23)"
      }
    }
  },
  plugins: []
};
