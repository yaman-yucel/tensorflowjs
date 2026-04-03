import fs from 'fs'
import path from 'path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const certPath = path.resolve(__dirname, 'certs/cert.pem')
const keyPath  = path.resolve(__dirname, 'certs/key.pem')
const hasCerts = fs.existsSync(certPath) && fs.existsSync(keyPath)

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    host: true,
    https: hasCerts
      ? { cert: fs.readFileSync(certPath), key: fs.readFileSync(keyPath) }
      : undefined,
  },
  build: {
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        manualChunks: {
          tfjs: ['@tensorflow/tfjs'],
        },
      },
    },
  },
})
