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
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000"
npm run dev -- --port 3000
```

## Vercel Deployment

1. Create a Vercel project with root directory set to `frontend`.
2. Add environment variable `NEXT_PUBLIC_API_BASE_URL` pointing to the deployed FastAPI backend.
3. Deploy with the default Next.js build command: `npm run build`.

## Backend CORS

Set the backend environment variable to your Vercel domain:

```powershell
$env:PROFIT_APP_CORS_ORIGINS="https://your-vercel-app.vercel.app"
```

For local testing, the backend defaults to permissive CORS.
