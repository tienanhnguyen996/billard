import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8080',
      '^/.*\\.mp4$': 'http://localhost:8080',
      '^/.*\\.json$': 'http://localhost:8080',
      '^/.*\\.js$': 'http://localhost:8080',
      '^/.*\\.png$': 'http://localhost:8080',
      '^/.*\\.jpg$': 'http://localhost:8080'
    }
  },
  build: {
    outDir: path.resolve(import.meta.dirname, '../dist'),
    emptyOutDir: true
  }
});
