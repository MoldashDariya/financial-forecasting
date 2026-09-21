import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // In dev the frontend calls /api/... and Vite forwards it to the FastAPI backend, so no CORS setup is needed.
    // Start the backend on port 8012 (see README)
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8012',
        changeOrigin: true,
      },
    },
  },
})
