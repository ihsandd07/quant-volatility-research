"""Load and align a daily Treasury-yield proxy."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RISK_FREE_PATH = ROOT / "data" / "raw" / "dgs3mo.csv"
TRADING_DAYS = 252


def load_risk_free_returns(trading_dates):
    """Convert lagged 3-month Treasury yields to approximate daily returns."""
    dates = pd.DatetimeIndex(trading_dates).sort_values().unique()

    if len(dates) == 0:
        raise ValueError("Trading dates cannot be empty.")

    if not RISK_FREE_PATH.exists():
        raise FileNotFoundError(
            f"Missing Treasury-yield data: {RISK_FREE_PATH}"
        )

    raw = pd.read_csv(
        RISK_FREE_PATH,
        parse_dates=["observation_date"],
    )

    if not {"observation_date", "DGS3MO"}.issubset(raw.columns):
        raise ValueError("Treasury data must contain observation_date and DGS3MO.")

    rates = raw.set_index("observation_date")["DGS3MO"]
    rates = pd.to_numeric(rates, errors="coerce").dropna().sort_index()

    if rates.empty:
        raise ValueError("No valid Treasury-yield observations found.")

    # Use the observation only from the following calendar day.
    rates.index = rates.index + pd.Timedelta(days=1)

    annual_yield = rates / 100.0

    if (annual_yield <= -1).any():
        raise ValueError("Invalid annualized Treasury yield.")

    daily_returns = (1.0 + annual_yield) ** (1.0 / TRADING_DAYS) - 1.0

    aligned = daily_returns.reindex(dates, method="ffill")

    if aligned.isna().any():
        raise ValueError("Treasury data does not cover all requested dates.")

    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Non-finite daily Treasury returns found.")

    aligned.name = "risk_free_return"
    return aligned
