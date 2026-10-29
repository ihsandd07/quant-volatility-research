
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "IWM", "EFA", "AGG", "GLD", "VNQ"]
START_DATE = "2019-01-01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_prices() -> pd.DataFrame:
    """Download daily adjusted prices for the six ETFs."""
    data = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )

    if data.empty:
        raise RuntimeError(
            "No data returned. Check your internet connection "
            "and Yahoo Finance availability."
        )

    # With auto_adjust=True, Close contains adjusted prices.
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError(
            "Unexpected data format: expected multiple ETF columns."
        )

    if "Close" in data.columns.get_level_values(0):
        prices = data["Close"].copy()
    elif "Close" in data.columns.get_level_values(1):
        prices = data.xs("Close", axis=1, level=1).copy()
    else:
        raise ValueError("Could not locate closing-price columns.")

    missing_tickers = set(TICKERS) - set(prices.columns)
    if missing_tickers:
        raise ValueError(
            f"Missing ETF columns: {sorted(missing_tickers)}"
        )

    prices = prices[TICKERS].copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.mask(prices <= 0)

    return prices


def validate_prices(prices: pd.DataFrame) -> None:
    """Validate the structure and basic quality of price data."""
    if prices.empty:
        raise ValueError("Price dataset is empty.")

    if list(prices.columns) != TICKERS:
        raise ValueError("Unexpected ETF columns or order.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dates are not in chronological order.")

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates detected.")

    if prices.isna().all().any():
        bad = prices.columns[prices.isna().all()].tolist()
        raise ValueError(f"Entirely missing series: {bad}")

    print("\nMissing observations (%):")
    print((prices.isna().mean() * 100).round(3))

    print("\nDataset overview")
    print(f"Start: {prices.index.min().date()}")
    print(f"End:   {prices.index.max().date()}")
    print(f"Rows:  {len(prices):,}")
    print(f"ETFs:  {len(prices.columns)}")

    print("\nFirst five observations:")
    print(prices.head())

    print("\nSummary statistics:")
    print(prices.describe().round(2))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading ETF price data...")
    prices = download_prices()
    validate_prices(prices)

    output_path = RAW_DIR / "adjusted_prices.csv"
    prices.to_csv(output_path, index_label="Date")

    print(f"\nSaved prices to: {output_path}")


if __name__ == "__main__":
    main()
