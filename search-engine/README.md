# Search Engine MVP

This folder contains an isolated MVP for the geology search engine UI:

- `backend_proxy/` is a FastAPI proxy layer for:
  - relevance model search forwarding
  - clustered reranking orchestration
  - query expansion passthrough
  - SerpApi-backed Google and Bing comparison results
  - demo query loading
- `frontend/` is a React + Vite dashboard that calls only the proxy API

## Run Order

Start the original services first from the repo root:

```powershell
python backend/app.py
python -m cluster_service
```

Then start the isolated MVP services from this folder:

```powershell
python -m uvicorn backend_proxy.app:app --host 127.0.0.1 --port 8020 --reload
cd frontend
npm install
npm run dev
```

## Environment

Copy `.env.example` to `.env` inside this folder and set `SERPAPI_API_KEY`.

For the frontend, copy `frontend/.env.example` to `frontend/.env` if you need a
different dev proxy target than the default `http://127.0.0.1:8020`.

During local development, the React app should call `/api/...` through Vite's
dev proxy. That avoids browser CORS issues entirely.
