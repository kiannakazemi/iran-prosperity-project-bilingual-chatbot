import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
//
// The app is deployed under iranprosperityproject.com/assistant — set the
// build base to /assistant/ so all asset URLs (JS bundles, CSS, public files)
// are emitted with that prefix. Vite also auto-rewrites root-relative paths
// in index.html. In application code, refer to public/ assets via
// `${import.meta.env.BASE_URL}<path>` so the prefix is picked up in both
// dev (localhost:5173/assistant/…) and production.
export default defineConfig({
  base: '/assistant/',
  plugins: [react()],
})
