"""
utils/validation.py
Dataset validation utilities for GridCast AI.
"""
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = {"timestamp", "consumption"}


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Validate a dataframe for required columns, types, and data quality.
    Returns a dict with keys:
        valid (bool), errors (list[str]), warnings (list[str])
    """
    errors = []
    warnings = []

    # --- Required columns ---
    missing = REQUIRED_COLUMNS - set(c.lower().strip() for c in df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # --- Timestamp parseable ---
    try:
        pd.to_datetime(df["timestamp"])
    except Exception as e:
        errors.append(f"Cannot parse 'timestamp' column: {e}")

    # --- Consumption numeric ---
    if not pd.api.types.is_numeric_dtype(df["consumption"]):
        try:
            pd.to_numeric(df["consumption"], errors="raise")
        except Exception:
            errors.append("'consumption' column must be numeric.")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # --- Minimum rows ---
    if len(df) < 48:
        errors.append("Dataset must have at least 48 rows for meaningful forecasting.")

    # --- Missing values ---
    n_missing = df["consumption"].isna().sum()
    if n_missing > 0:
        pct = n_missing / len(df) * 100
        if pct > 20:
            warnings.append(f"{n_missing} missing consumption values ({pct:.1f}%). High proportion may reduce accuracy.")
        else:
            warnings.append(f"{n_missing} missing consumption values ({pct:.1f}%). These will be interpolated.")

    # --- Negative values ---
    neg = (pd.to_numeric(df["consumption"], errors="coerce") < 0).sum()
    if neg > 0:
        warnings.append(f"{neg} negative consumption values detected. These will be clipped to zero.")

    # --- Duplicate timestamps ---
    n_dup = df.duplicated(subset=["timestamp"]).sum()
    if n_dup > 0:
        warnings.append(f"{n_dup} duplicate timestamps detected. Duplicates will be aggregated.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
