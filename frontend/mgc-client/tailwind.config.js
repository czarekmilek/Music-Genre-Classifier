/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,ts}"],
  theme: {
    extend: {
      colors: {
        "func-button-primary-background": "rgba(var(--func-button-primary-bg))",
        "func-button-secondary-background":
          "rgba(var(--func-button-secondary-bg))",
        "func-button-active-primary-background":
          "rgba(var(--func-button-active-primary-bg))",
        "func-button-active-secondary-background":
          "rgba(var(--func-button-active-secondary-bg))",
      },
    },
  },
  plugins: [],
};
