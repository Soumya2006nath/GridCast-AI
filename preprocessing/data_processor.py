"""
preprocessing/data_processor.py
Handles all data loading, cleaning, and resampling for GridCast AI.
No future information is used during preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


class DataProcessor:
    """
    Loads, cleans, resamples, and returns training/test splits.
    All operations are strictly chronological — no look-ahead.
    """

    def __init__(self, resample_freq: str = "h"):
        """
        Parameters
        ----------
        resample_freq : str
            Pandas offset alias for resampling. Default 'h' (hourly).
            Use 'D' for daily, '30min' for 30-minute intervals, etc.
        """
        self.resample_freq = resample_freq
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.stats: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_csv(self, path_or_buffer) -> pd.DataFrame:
        """Load CSV from a file path or a file-like object."""
        df = pd.read_csv(path_or_buffer)
        df.columns = [c.lower().strip() for c in df.columns]
        self.raw_df = df
        return df

    def process(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        1. Parse timestamps
        2. Sort chronologically
        3. Clip negatives
        4. Aggregate duplicate timestamps (mean)
        5. Resample to consistent frequency
        6. Interpolate missing values (time-based)
        7. Compute stats
        Returns the processed DataFrame indexed by datetime.
        """
        if df is None:
            df = self.raw_df
        if df is None:
            raise ValueError("No dataframe provided.")

        df = df.copy()

        # 1. Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 2. Sort
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 3. Numeric conversion & clip negatives
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df["consumption"] = df["consumption"].clip(lower=0)

        # 4. Aggregate duplicates
        df = df.groupby("timestamp", as_index=False).agg({"consumption": "mean"})

        # 5. Set index
        df = df.set_index("timestamp")

        # 6. Resample
        df = df.resample(self.resample_freq).mean()

        # 7. Interpolate missing
        df["consumption"] = df["consumption"].interpolate(method="time")
        # Fill any remaining edge NaNs with forward/back fill
        df["consumption"] = df["consumption"].ffill().bfill()

        # 8. Stats (computed on clean data, no leakage concern)
        self.stats = self._compute_stats(df)
        self.processed_df = df
        return df

    def train_test_split(
        self, df: pd.DataFrame, test_days: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological split — test set is the LAST `test_days` days.
        Never shuffles. Never uses future data in training.
        """
        cutoff = df.index.max() - pd.Timedelta(days=test_days)
        train = df[df.index <= cutoff].copy()
        test = df[df.index > cutoff].copy()
        return train, test

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(df: pd.DataFrame) -> dict:
        c = df["consumption"]
        return {
            "total_kwh": float(c.sum()),
            "mean_kwh": float(c.mean()),
            "max_kwh": float(c.max()),
            "min_kwh": float(c.min()),
            "std_kwh": float(c.std()),
            "n_records": int(len(c)),
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "n_missing_raw": int(c.isna().sum()),
        }

    def get_daily(self) -> pd.DataFrame:
        """Return daily aggregated consumption."""
        if self.processed_df is None:
            raise ValueError("Call process() first.")
        return self.processed_df["consumption"].resample("D").sum().to_frame()

    def get_weekly(self) -> pd.DataFrame:
        """Return weekly aggregated consumption."""
        if self.processed_df is None:
            raise ValueError("Call process() first.")
        return self.processed_df["consumption"].resample("W").sum().to_frame()

    def get_monthly(self) -> pd.DataFrame:
        """Return monthly aggregated consumption."""
        if self.processed_df is None:
            raise ValueError("Call process() first.")
        return self.processed_df["consumption"].resample("ME").sum().to_frame()
