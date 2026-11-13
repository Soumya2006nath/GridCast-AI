# ⚡ GridCast AI — Smart Grid Electricity Consumption Forecasting

**Group 14** | AI/ML — Time Series Forecasting  
Team: Lawrence Sarkar, Soumyadeep Nath, Sayon De, Pravin Agarwal

---

## What It Does

GridCast AI analyzes historical smart-meter data and predicts electricity consumption for the next **7 days** using three models:

| Model | Type |
|---|---|
| ARIMA | Statistical — AutoRegressive Integrated Moving Average |
| SARIMA | Statistical — Seasonal ARIMA |
| LSTM | Deep Learning — Long Short-Term Memory |

---

## Setup

### 1. Install Python 3.11+

Download from [python.org](https://www.python.org/downloads/).

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> LSTM training requires TensorFlow. On Windows you may need:
> `pip install tensorflow-cpu` if GPU drivers are unavailable.

### 4. Run the app

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Usage

1. Click **Use Sample Dataset** (sidebar) or upload your own CSV.
2. Required columns: `timestamp`, `consumption`
3. Navigate to **Forecasting**, select a model, and click **Generate Forecast**.
4. View metrics on **Model Performance**.
5. Download a PDF report from **Reports**.

### CSV Format

```csv
timestamp,consumption,temperature,humidity
2022-01-01 00:00:00,3.52,12.3,65.1
2022-01-01 01:00:00,3.11,11.9,66.2
...
```

Minimum 48 rows. Timestamps must be parseable by `pandas.to_datetime`.

---

## Project Structure

```
gridcast-ai/
├── app.py                    # Streamlit dashboard entry point
├── requirements.txt
├── README.md
├── data/
│   └── sample_energy_data.csv
├── models/
│   ├── arima_model.py
│   ├── sarima_model.py
│   └── lstm_model.py
├── preprocessing/
│   └── data_processor.py
├── forecasting/
│   └── forecast_engine.py
├── evaluation/
│   └── metrics.py
├── visualization/
│   └── charts.py
├── reporting/
│   └── report_generator.py
└── utils/
    └── validation.py
```

---

## Evaluation Metrics

- **MAE** — Mean Absolute Error (primary metric)
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error

All metrics are calculated on a held-out chronological test set (last 30 days). No future data leaks into training.

---

## Notes

- The app runs entirely locally — no paid APIs required.
- LSTM training takes 1–3 minutes depending on dataset size and hardware.
- Uploaded datasets are only held in session memory and not persisted to disk.
