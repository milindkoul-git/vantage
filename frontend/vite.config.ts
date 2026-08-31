import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../src/vantage/dashboard/static',
    emptyOutDir: false,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name].[ext]',
        // Stable filenames, because the dashboard server serves this directory
        // directly and index.html references the entry by name. Hashes would
        // leave every previous build's chunk behind on disk.
        manualChunks: {
          react: ['react', 'react-dom'],
        },
      }
    }
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/stream.mjpg': 'http://127.0.0.1:8080',
      '/snapshot.jpg': 'http://127.0.0.1:8080',
      '/static': 'http://127.0.0.1:8080',
    }
  }
})
