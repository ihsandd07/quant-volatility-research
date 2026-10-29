
from pathlib import Path

import pandas as pd
from scipy.stats import skew, kurtosis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "simple_returns.csv"


def load_returns() -> pd.DataFrame:
    """Load daily simple returns."""
    returns = pd.read_csv(
        RETURNS_PATH,
        index_col="Date",
        parse_dates=True,
    )

    if returns.empty:
        raise ValueError("Return dataset is empty.")

    if returns.isna().any().any():
        raise ValueError("Missing returns detected.")

    return returns


def distribution_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for each asset."""
    results = {}

    for ticker in returns.columns:
        series = returns[ticker]

        results[ticker] = {
            "Mean daily return (%)": series.mean() * 100,
            "Daily volatility (%)": series.std() * 100,
            "Skewness": skew(series, bias=False),
            "Excess kurtosis": kurtosis(
                series,
                fisher=True,
                bias=False,
            ),
            "Minimum daily return (%)": series.min() * 100,
            "Maximum daily return (%)": series.max() * 100,
        }

    return pd.DataFrame(results).T


def main() -> None:
    returns = load_returns()
    summary = distribution_summary(returns)

    print("Return distribution analysis")
    print(f"Observations: {len(returns):,}")
    print(f"Assets: {len(returns.columns)}")
    print("\nStatistics:")
    print(summary.round(4))

    output_path = PROJECT_ROOT / "reports" / "distribution_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
