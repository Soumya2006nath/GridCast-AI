"""
app.py
GridCast AI — Smart Grid Electricity Consumption Forecasting Dashboard.
Group 14 | Streamlit entry-point.

Run: streamlit run app.py
"""

import sys
import os

# Allow absolute imports from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np

from preprocessing.data_processor import DataProcessor
from forecasting.forecast_engine import ForecastEngine
from visualization.charts import (
    historical_line, daily_pattern, weekly_trend, monthly_trend,
    consumption_distribution, forecast_chart, model_comparison_bar,
    actual_vs_predicted,
)
from utils.validation import validate_dataframe

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="GridCast AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #0a1628; color: #e0e6ed; }
    section[data-testid="stSidebar"] { background-color: #0d1b2a; border-right: 1px solid #1a2a3a; }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown p { color: #e0e6ed; }

    /* KPI cards */
    .kpi-card {
        background: #0f2035;
        border: 1px solid #1a3a5a;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 12px; color: #90a4ae; letter-spacing: 0.08em; text-transform: uppercase; }
    .kpi-value { font-size: 26px; font-weight: 700; color: #00d4ff; margin: 4px 0; }
    .kpi-unit  { font-size: 11px; color: #607d8b; }

    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #1e90ff;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Metric table */
    .metric-table th { background: #1e3a5f; color: #00d4ff; font-size: 12px; }
    .metric-table td { font-size: 12px; color: #e0e6ed; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e90ff, #00d4ff);
        color: #0a1628;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.4rem;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Success/warning/error boxes */
    .stAlert { border-radius: 8px; }

    /* Plotly chart backgrounds already set in charts.py */
    div[data-testid="stPlotlyChart"] > div { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# Session-state helpers
# ══════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "processor": None,
        "engine": None,
        "data_loaded": False,
        "train_df": None,
        "test_df": None,
        "processed_df": None,
        "arima_done": False,
        "sarima_done": False,
        "lstm_done": False,
        "arima_metrics": None,
        "sarima_metrics": None,
        "lstm_metrics": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ GridCast AI")
    st.markdown("*Smart Grid Forecasting — Group 14*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Data Analysis", "Forecasting", "Model Performance", "Reports"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Load Dataset**")
    upload = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

    use_sample = st.button("Use Sample Dataset")

    if upload or use_sample:
        try:
            if upload:
                raw_df = pd.read_csv(upload)
            else:
                sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_energy_data.csv")
                raw_df = pd.read_csv(sample_path)

            # Validate
            val = validate_dataframe(raw_df.copy())
            if not val["valid"]:
                for err in val["errors"]:
                    st.error(err)
                st.stop()
            for warn in val["warnings"]:
                st.warning(warn)

            # Process
            proc = DataProcessor(resample_freq="h")
            proc.load_csv(os.path.join(os.path.dirname(__file__), "data", "sample_energy_data.csv") if use_sample else upload)
            processed = proc.process(raw_df)

            train, test = proc.train_test_split(processed, test_days=30)

            engine = ForecastEngine(resample_freq="h")
            engine.set_data(train, test)

            st.session_state.processor = proc
            st.session_state.engine = engine
            st.session_state.processed_df = processed
            st.session_state.train_df = train
            st.session_state.test_df = test
            st.session_state.data_loaded = True
            st.session_state.arima_done = False
            st.session_state.sarima_done = False
            st.session_state.lstm_done = False
            st.session_state.arima_metrics = None
            st.session_state.sarima_metrics = None
            st.session_state.lstm_metrics = None

            st.success("Dataset loaded ✓")

        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

    st.markdown("---")
    st.markdown("**Team Members**")
    for m in ["Lawrence Sarkar", "Soumyadeep Nath", "Sayon De", "Pravin Agarwal"]:
        st.markdown(f"· {m}")


# ══════════════════════════════════════════════════════════════════════
# Shared state references
# ══════════════════════════════════════════════════════════════════════

data_loaded = st.session_state.data_loaded
proc: DataProcessor = st.session_state.processor
engine: ForecastEngine = st.session_state.engine
processed_df: pd.DataFrame = st.session_state.processed_df


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown('<div class="section-header">⚡ GridCast AI — Overview Dashboard</div>', unsafe_allow_html=True)

    if not data_loaded:
        st.info("👈 Upload a CSV or load the sample dataset from the sidebar to begin.")
        st.markdown("""
        **GridCast AI** is a machine-learning powered electricity consumption forecasting system.

        It supports three forecasting models:
        - **ARIMA** — AutoRegressive Integrated Moving Average
        - **SARIMA** — Seasonal ARIMA (captures daily/weekly cycles)
        - **LSTM** — Deep learning sequence model

        **Quick start:** Click *Use Sample Dataset* in the sidebar.
        """)
    else:
        stats = proc.stats

        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, unit in [
            (c1, "Total Consumption", f"{stats['total_kwh']:,.0f}", "kWh"),
            (c2, "Avg per Record", f"{stats['mean_kwh']:.4f}", "kWh"),
            (c3, "Peak Record", f"{stats['max_kwh']:.4f}", "kWh"),
            (c4, "Records", f"{stats['n_records']:,}", ""),
        ]:
            col.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)

        # Date range
        st.markdown(
            f"**Dataset:** {stats['start_date']} → {stats['end_date']} "
            f"&nbsp;|&nbsp; Train / Test split: last 30 days held out",
            unsafe_allow_html=False,
        )

        # Historical chart
        st.plotly_chart(historical_line(processed_df), use_container_width=True)

        # Latest model metrics if any
        if engine.has_results():
            st.markdown("---")
            st.markdown("**Latest Model Evaluation (Test Set)**")
            m_data = engine.get_all_metrics()
            rows = []
            for mname, mv in m_data.items():
                rows.append({
                    "Model": mname,
                    "MAE": f"{mv.get('MAE',0):.4f}",
                    "RMSE": f"{mv.get('RMSE',0):.4f}",
                    "MAPE (%)": f"{mv['MAPE']:.2f}" if mv.get("MAPE") is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            best = engine.get_best_model()
            st.success(f"Best model by MAE: **{best}**")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Analysis
# ══════════════════════════════════════════════════════════════════════

elif page == "Data Analysis":
    st.markdown('<div class="section-header">📊 Data Analysis</div>', unsafe_allow_html=True)

    if not data_loaded:
        st.info("Load a dataset first.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Dataset Preview", "Daily Pattern", "Weekly Trend", "Monthly Trend", "Distribution"
        ])

        with tab1:
            st.markdown("**Raw data preview (first 100 rows)**")
            st.dataframe(processed_df.head(100), use_container_width=True)
            stats = proc.stats
            st.markdown(f"""
            | Stat | Value |
            |---|---|
            | Records | {stats['n_records']:,} |
            | Start | {stats['start_date']} |
            | End | {stats['end_date']} |
            | Mean kWh | {stats['mean_kwh']:.4f} |
            | Max kWh | {stats['max_kwh']:.4f} |
            | Min kWh | {stats['min_kwh']:.4f} |
            | Std Dev | {stats['std_kwh']:.4f} |
            """)

        with tab2:
            daily = proc.get_daily()
            st.plotly_chart(daily_pattern(daily), use_container_width=True)

        with tab3:
            weekly = proc.get_weekly()
            st.plotly_chart(weekly_trend(weekly), use_container_width=True)

        with tab4:
            monthly = proc.get_monthly()
            if len(monthly) >= 2:
                st.plotly_chart(monthly_trend(monthly), use_container_width=True)
            else:
                st.info("Not enough data for monthly trend (need at least 2 months).")

        with tab5:
            st.plotly_chart(consumption_distribution(processed_df), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — Forecasting
# ══════════════════════════════════════════════════════════════════════

elif page == "Forecasting":
    st.markdown('<div class="section-header">🔮 7-Day Forecasting</div>', unsafe_allow_html=True)

    if not data_loaded:
        st.info("Load a dataset first.")
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("**Select Model**")
            model_choice = st.selectbox(
                "Model",
                ["ARIMA", "SARIMA", "LSTM"],
                label_visibility="collapsed",
            )

            if model_choice == "LSTM":
                lookback = st.slider("Lookback window (hours)", 24, 168, 48, 12)
                epochs = st.slider("Training epochs", 5, 50, 20, 5)
            else:
                lookback, epochs = 48, 20

            run_btn = st.button(f"▶ Generate {model_choice} Forecast")

        with col_right:
            already_done = {
                "ARIMA": st.session_state.arima_done,
                "SARIMA": st.session_state.sarima_done,
                "LSTM": st.session_state.lstm_done,
            }

            if run_btn or already_done.get(model_choice):
                if run_btn:
                    with st.spinner(f"Training {model_choice}… please wait."):
                        try:
                            if model_choice == "ARIMA":
                                metrics = engine.run_arima()
                                st.session_state.arima_done = True
                                st.session_state.arima_metrics = metrics
                            elif model_choice == "SARIMA":
                                metrics = engine.run_sarima()
                                st.session_state.sarima_done = True
                                st.session_state.sarima_metrics = metrics
                            else:
                                metrics = engine.run_lstm(lookback=lookback, epochs=epochs)
                                st.session_state.lstm_done = True
                                st.session_state.lstm_metrics = metrics
                        except Exception as e:
                            st.error(f"Model training failed: {e}")
                            st.stop()

                if engine._forecasts.get(model_choice) is not None:
                    forecast_df = engine.get_forecast_df(model_choice)
                    summary = engine.get_forecast_summary(model_choice)
                    historical_avg = float(proc.get_daily()["consumption"].mean())

                    # Summary KPIs
                    k1, k2, k3 = st.columns(3)
                    for col, label, val in [
                        (k1, "7-Day Total", f"{summary['total_7d_kwh']:.2f} kWh"),
                        (k2, "Avg Daily", f"{summary['avg_daily_kwh']:.2f} kWh"),
                        (k3, "Peak Day", f"{summary['peak_daily_kwh']:.2f} kWh"),
                    ]:
                        col.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">{label}</div>
                            <div class="kpi-value">{val}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.plotly_chart(
                        forecast_chart(forecast_df, historical_avg, model_choice),
                        use_container_width=True,
                    )
                    st.markdown("**Forecast Table**")
                    st.dataframe(
                        forecast_df.style.format({"Predicted_kWh": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info(f"Click the button to run {model_choice} forecasting.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════

elif page == "Model Performance":
    st.markdown('<div class="section-header">📈 Model Performance</div>', unsafe_allow_html=True)

    if not data_loaded:
        st.info("Load a dataset first.")
    elif not engine.has_results():
        st.info("No models trained yet. Go to **Forecasting** and run at least one model.")
    else:
        metrics = engine.get_all_metrics()
        best = engine.get_best_model()

        # Comparison table
        rows = []
        for mname, mv in metrics.items():
            rows.append({
                "Model": f"⭐ {mname}" if mname == best else mname,
                "MAE": round(mv.get("MAE", 0), 4),
                "RMSE": round(mv.get("RMSE", 0), 4),
                "MAPE (%)": round(mv["MAPE"], 2) if mv.get("MAPE") is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.success(f"Best model (lowest MAE): **{best}**")

        # Bar chart comparison
        st.plotly_chart(model_comparison_bar(metrics), use_container_width=True)

        # Actual vs Predicted per model
        st.markdown("---")
        st.markdown("**Actual vs Predicted on Test Set**")

        test_series = st.session_state.test_df["consumption"]
        for mname in metrics:
            m_obj = engine._models.get(mname)
            if m_obj is None:
                continue
            try:
                if mname == "ARIMA":
                    preds = m_obj.predict_in_sample(len(test_series))
                elif mname == "SARIMA":
                    preds = m_obj.predict_in_sample(len(test_series))
                else:  # LSTM
                    preds = m_obj.predict_in_sample(
                        st.session_state.train_df["consumption"], len(test_series)
                    )
                st.plotly_chart(
                    actual_vs_predicted(test_series, preds, mname),
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Could not plot {mname}: {e}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — Reports
# ══════════════════════════════════════════════════════════════════════

elif page == "Reports":
    st.markdown('<div class="section-header">📄 Reports</div>', unsafe_allow_html=True)

    if not data_loaded:
        st.info("Load a dataset first.")
    elif not engine.has_results():
        st.info("Train at least one model before generating a report.")
    else:
        st.markdown("Generate a downloadable PDF report with dataset statistics, model performance, and the 7-day forecast.")

        metrics = engine.get_all_metrics()
        best = engine.get_best_model()
        forecast_df = engine.get_forecast_df(best)

        if st.button("📥 Generate PDF Report"):
            with st.spinner("Building PDF…"):
                try:
                    from reporting.report_generator import generate_pdf
                    pdf_bytes = generate_pdf(
                        stats=proc.stats,
                        metrics=metrics,
                        forecast_df=forecast_df,
                        best_model_name=best,
                    )
                    st.download_button(
                        label="⬇ Download Report PDF",
                        data=pdf_bytes,
                        file_name="GridCast_AI_Report.pdf",
                        mime="application/pdf",
                    )
                    st.success("Report ready. Click the button above to download.")
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

        # Preview summary on screen
        st.markdown("---")
        st.markdown("**Report Preview**")
        st.markdown(f"**Best Model:** {best}")
        st.markdown("**7-Day Forecast Table:**")
        st.dataframe(
            forecast_df.style.format({"Predicted_kWh": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Model Metrics:**")
        rows = []
        for mname, mv in metrics.items():
            rows.append({
                "Model": mname,
                "MAE": round(mv.get("MAE", 0), 4),
                "RMSE": round(mv.get("RMSE", 0), 4),
                "MAPE (%)": round(mv["MAPE"], 2) if mv.get("MAPE") is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
