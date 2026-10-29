
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_prices() -> pd.DataFrame:
    """Load and validate the saved adjusted-price panel."""
    path = RAW_DIR / "adjusted_prices.csv"

    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    prices = prices.sort_index()

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates found in price data.")

    if prices.isna().any().any():
        raise ValueError(
            "Missing prices detected. Resolve them before calculating returns."
        )

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    return prices


def calculate_returns(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate simple returns and log returns."""
    if prices.empty:
        raise ValueError("Price data is empty.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    simple_returns = prices.pct_change(fill_method=None).iloc[1:]
    log_returns = np.log(prices).diff().iloc[1:]

    if not np.isfinite(simple_returns.to_numpy()).all():
        raise ValueError("Simple returns contain invalid values.")

    if not np.isfinite(log_returns.to_numpy()).all():
        raise ValueError("Log returns contain invalid values.")

    return simple_returns, log_returns


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_prices()
    simple_returns, log_returns = calculate_returns(prices)

    simple_path = PROCESSED_DIR / "simple_returns.csv"
    log_path = PROCESSED_DIR / "log_returns.csv"

    simple_returns.to_csv(simple_path, index_label="Date")
    log_returns.to_csv(log_path, index_label="Date")

    print("Return calculations completed.")
    print(f"Price observations: {len(prices):,}")
    print(f"Return observations: {len(simple_returns):,}")
    print(f"Assets: {len(prices.columns)}")
    print(f"First return date: {simple_returns.index.min().date()}")
    print(f"Last return date: {simple_returns.index.max().date()}")

    print("\nSimple return summary:")
    summary = simple_returns.describe()
    summary.loc[summary.index != "count"] *= 100
    print(summary.round(3))

    print(f"\nSaved: {simple_path}")
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    main()
