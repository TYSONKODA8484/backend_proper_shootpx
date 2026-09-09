# ShootPX New Backend — Design

Date: 2026-09-08

## Goal
A minimal, clean FastAPI backend boilerplate that runs on `http://localhost:8000`.
No database, no auth. Just enough structure to grow into.

## Stack
- FastAPI
- uvicorn[standard]
- pydantic-settings (config from `.env`)

## Folder structure
```
new_backend/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI instance, CORS, router include, startup log
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py      # Settings from .env
│   ├── routes/
│   │   ├── __init__.py
│   │   └── health.py      # GET /  and GET /health
│   ├── models/
│   │   └── __init__.py    # empty, ready for later
│   └── services/
│       └── __init__.py    # empty, ready for later
├── tests/
│   └── test_health.py
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py
└── README.md
```

## Endpoints
- `GET /`        -> `{"service": "shootpx-backend", "status": "ok"}`
- `GET /health`  -> `{"status": "healthy"}`
- `GET /docs`    -> Swagger UI (automatic)

## Config (`.env`)
- `APP_NAME=shootpx-backend`
- `ENV=development`
- `HOST=127.0.0.1`
- `PORT=8000`

## Run
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```
Prints: `ShootPX backend running on http://localhost:8000`

## Testing
`tests/test_health.py` — FastAPI TestClient asserts `/health` returns 200.
