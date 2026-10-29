
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "simple_returns.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_returns() -> pd.DataFrame:
    returns = pd.read_csv(
        RETURNS_PATH,
        index_col="Date",
        parse_dates=True,
    )

    if returns.empty or returns.isna().any().any():
        raise ValueError("Returns are empty or contain missing values.")

    return returns


def average_pairwise_correlation(returns: pd.DataFrame) -> float:
    """Mean of the unique off-diagonal correlations."""
    corr = returns.corr().to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(upper))


def analyze_regimes(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Compare correlations in low- and high-SPY-volatility regimes."""
    # Trailing 21-session annualized realized volatility.
    spy_vol = returns["SPY"].rolling(21).std() * np.sqrt(252)

    # Ex-post threshold used for descriptive analysis only.
    threshold = spy_vol.quantile(0.80)
    high_vol = spy_vol > threshold

    valid = spy_vol.notna()
    low_returns = returns.loc[valid & ~high_vol]
    high_returns = returns.loc[valid & high_vol]

    summary = pd.DataFrame(
        {
            "Observations": [
                len(low_returns),
                len(high_returns),
            ],
            "Average pairwise correlation": [
                average_pairwise_correlation(low_returns),
                average_pairwise_correlation(high_returns),
            ],
            "SPY annualized volatility (%)": [
                spy_vol.loc[low_returns.index].mean() * 100,
                spy_vol.loc[high_returns.index].mean() * 100,
            ],
        },
        index=["Low volatility", "High volatility"],
    )

    return summary, spy_vol


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()
    summary, spy_vol = analyze_regimes(returns)

    print("Volatility regime analysis")
    print(f"Assets: {len(returns.columns)}")
    print(f"Threshold: 80th percentile of trailing SPY volatility")
    print(f"Threshold value: {spy_vol.quantile(0.80) * 100:.2f}% annualized")
    print("\nRegime comparison:")
    print(summary.round(4))

    summary.to_csv(REPORTS_DIR / "regime_summary.csv")

    # Plot annualized rolling volatility.
    rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100

    ax = rolling_vol.plot(figsize=(12, 6), linewidth=1)
    ax.set_title("21-Day Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility (%)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "rolling_volatility.png", dpi=160)
    plt.close()

    print(f"\nSaved: {REPORTS_DIR / 'regime_summary.csv'}")
    print(f"Saved: {REPORTS_DIR / 'rolling_volatility.png'}")


if __name__ == "__main__":
    main()
