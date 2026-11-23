# ⚡ GridCast AI v2 — Smart Grid Electricity Forecasting

**Group 14** | AI/ML — Time Series Forecasting  
Team: Lawrence Sarkar, Soumyadeep Nath, Sayon De, Pravin Agarwal

---

## Architecture

```
Browser (Next.js + Tailwind + Recharts)
        ↕  REST API (JSON)
FastAPI backend  →  ARIMA / SARIMA / LSTM (GPU via TensorFlow + CUDA)
```

---

## GPU Setup (RTX 4050)

Install CUDA 12.x from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), then:

```bash
pip install tensorflow[and-cuda]
```

TensorFlow will automatically detect and use the GPU. The backend logs `[LSTM] Training on GPU: ...` on startup.

---

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Or double-click `start-backend.bat`.

Backend runs at [http://localhost:8000](http://localhost:8000)  
API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Or double-click `start-frontend.bat`.

Frontend runs at [http://localhost:3000](http://localhost:3000)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/dataset/sample` | Load built-in sample dataset |
| POST | `/api/dataset/upload` | Upload a CSV file |
| GET | `/api/dataset/stats` | Dataset statistics |
| GET | `/api/dataset/historical` | Full time series |
| GET | `/api/dataset/daily` | Daily aggregates |
| POST | `/api/forecast/run` | Train model + generate forecast |
| GET | `/api/metrics` | Model evaluation results |
| GET | `/api/report/pdf` | Download PDF report |
| GET | `/api/health` | Server + GPU status |

---

## CSV Format

```csv
timestamp,consumption,temperature,humidity
2024-01-01 00:00:00,3.52,12.3,65.1
```

Minimum 48 rows. `timestamp` and `consumption` are required.

---

## Project Structure

```
GridCast AI/
├── backend/
│   ├── main.py              ← FastAPI app
│   ├── models/              ← ARIMA, SARIMA, LSTM (GPU)
│   ├── preprocessing/
│   ├── forecasting/
│   ├── evaluation/
│   ├── reporting/
│   ├── utils/
│   ├── data/
│   └── requirements.txt
├── frontend/
│   ├── app/                 ← Next.js pages
│   ├── components/          ← Sidebar, KPI cards, charts, pages
│   └── lib/api.ts           ← API client
├── start-backend.bat
├── start-frontend.bat
└── README.md
```

---

## Legacy Streamlit App

The original Streamlit app is still in the root (`app.py`) and works independently.
