# Profit App Frontend

Next.js dashboard for the Profit App paper-trading system. It is designed to be deployed on Vercel while the Python FastAPI backend runs separately.

## Local Development

```powershell
cd frontend
npm install
npm run dev -- --port 3000
```

The app expects the backend API at `NEXT_PUBLIC_API_BASE_URL`.

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8021"
npm run dev -- --port 3000
```

## Vercel Deployment

1. Create a Vercel project with root directory set to `frontend`.
2. Add environment variable `NEXT_PUBLIC_API_BASE_URL` pointing to the deployed FastAPI backend.
3. Optionally add `BACKEND_API_BASE_URL` with the same backend URL for the server-side proxy route.
4. Deploy with `npm ci` and `npm run build`.

The production frontend intentionally requires `NEXT_PUBLIC_API_BASE_URL`. If it is missing, the UI will render a clear configuration error instead of attempting to call `127.0.0.1` from the user's browser.

## Backend CORS

Set the backend environment variable to your Vercel domain:

```powershell
$env:PROFIT_APP_CORS_ORIGINS="https://your-vercel-app.vercel.app"
```

For local testing, the backend defaults to permissive CORS.

## Important Runtime Note

This repository uses Vercel for the Next.js frontend only. The current FastAPI backend should run on a separate Python host because it depends on PostgreSQL, model artifacts on disk, and heavier ML runtimes that are not a good fit for Vercel serverless execution.
