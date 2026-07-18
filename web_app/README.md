# AgriBloom web app

This is the farmer-facing web frontend. It is intentionally framework-free so
it stays small, fast, and installable on low-end phones.

Run it with the API bridge:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

The frontend calls `POST /api/analyze` and renders the backend
`farmer_advice` contract as action cards.

