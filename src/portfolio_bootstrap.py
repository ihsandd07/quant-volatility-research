"""Moving-block bootstrap for portfolio performance uncertainty."""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "reports" / "portfolio_daily_results.csv"
OUTPUT_CI = ROOT / "reports" / "portfolio_bootstrap_ci.csv"
OUTPUT_DIFF = ROOT / "reports" / "portfolio_bootstrap_sharpe_differences.csv"

TRADING_DAYS = 252
BLOCK_LENGTH = 10
N_BOOTSTRAP = 2000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def performance_metrics(returns, risk_free_returns):
    """Calculate performance metrics using excess returns for Sharpe."""
    returns = np.asarray(returns, dtype=float)
    risk_free_returns = np.asarray(risk_free_returns, dtype=float)

    if len(returns) != len(risk_free_returns):
        raise ValueError("Returns and risk-free series must have equal lengths.")

    equity = np.cumprod(1.0 + returns)
    daily_std = returns.std(ddof=1)
    excess_returns = returns - risk_free_returns
    excess_std = excess_returns.std(ddof=1)

    annualized_return = equity[-1] ** (TRADING_DAYS / len(returns)) - 1.0
    annualized_volatility = daily_std * np.sqrt(TRADING_DAYS)

    sharpe = (
        excess_returns.mean() / excess_std * np.sqrt(TRADING_DAYS)
        if excess_std > 0
        else np.nan
    )

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
    }


def moving_block_indices(n, block_length, rng):
    """Generate circular moving-block bootstrap indices."""
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)

    indices = np.concatenate([
        (start + np.arange(block_length)) % n
        for start in starts
    ])

    return indices[:n]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    daily = pd.read_csv(
        INPUT_PATH,
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )

    strategies = daily.columns.get_level_values(0).unique().tolist()

    returns = pd.DataFrame({
        strategy: daily[(strategy, "net_return")]
        for strategy in strategies
    })

    risk_free = daily[(strategies[0], "risk_free_return")]

    # Confirm the same Treasury series was saved for every strategy.
    for strategy in strategies[1:]:
        other = daily[(strategy, "risk_free_return")]
        if not np.allclose(
            risk_free.to_numpy(),
            other.to_numpy(),
            equal_nan=True,
        ):
            raise ValueError("Risk-free returns differ across strategies.")

    combined = pd.concat(
        [returns, risk_free.rename("risk_free_return")],
        axis=1,
    ).dropna()

    returns = combined[strategies]
    risk_free = combined["risk_free_return"]

    if len(returns) < BLOCK_LENGTH:
        raise ValueError("Not enough observations for the selected block length.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Portfolio returns contain non-finite values.")

    if not np.isfinite(risk_free.to_numpy()).all():
        raise ValueError("Risk-free returns contain non-finite values.")

    n = len(returns)
    rng = np.random.default_rng(RANDOM_SEED)

    metric_names = [
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
    ]

    bootstrap_values = {
        strategy: {metric: [] for metric in metric_names}
        for strategy in strategies
    }

    sharpe_differences = {
        f"{a}_minus_{b}": []
        for a, b in combinations(strategies, 2)
    }

    for _ in range(N_BOOTSTRAP):
        indices = moving_block_indices(n, BLOCK_LENGTH, rng)
        sample = returns.iloc[indices]
        sample_rf = risk_free.iloc[indices].to_numpy()

        sample_sharpes = {}

        for strategy in strategies:
            metrics = performance_metrics(
                sample[strategy].to_numpy(),
                sample_rf,
            )

            for metric in metric_names:
                bootstrap_values[strategy][metric].append(metrics[metric])

            sample_sharpes[strategy] = metrics["sharpe_ratio"]

        for a, b in combinations(strategies, 2):
            key = f"{a}_minus_{b}"
            sharpe_differences[key].append(
                sample_sharpes[a] - sample_sharpes[b]
            )

    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    lower_pct = 100.0 * alpha
    upper_pct = 100.0 * (1.0 - alpha)

    ci_rows = []

    for strategy in strategies:
        for metric in metric_names:
            values = np.asarray(bootstrap_values[strategy][metric])

            ci_rows.append({
                "strategy": strategy,
                "metric": metric,
                "bootstrap_mean": np.nanmean(values),
                "ci_lower": np.nanpercentile(values, lower_pct),
                "ci_upper": np.nanpercentile(values, upper_pct),
                "confidence_level": CONFIDENCE_LEVEL,
                "bootstrap_replications": N_BOOTSTRAP,
                "block_length": BLOCK_LENGTH,
            })

    ci_results = pd.DataFrame(ci_rows)
    ci_results.to_csv(OUTPUT_CI, index=False)

    difference_rows = []

    for comparison, values in sharpe_differences.items():
        values = np.asarray(values)
        lower = np.nanpercentile(values, lower_pct)
        upper = np.nanpercentile(values, upper_pct)

        difference_rows.append({
            "comparison": comparison,
            "mean_sharpe_difference": np.nanmean(values),
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_replications": N_BOOTSTRAP,
            "block_length": BLOCK_LENGTH,
            "includes_zero": lower <= 0 <= upper,
        })

    difference_results = pd.DataFrame(difference_rows)
    difference_results.to_csv(OUTPUT_DIFF, index=False)

    print(f"Observations: {n}")
    print(f"Strategies: {len(strategies)}")
    print(f"Bootstrap replications: {N_BOOTSTRAP}")
    print(f"Block length: {BLOCK_LENGTH} trading days")
    print("\n95% confidence intervals:")
    print(ci_results.to_string(index=False))
    print("\nPaired Sharpe-ratio differences:")
    print(difference_results.to_string(index=False))
    print(f"\nSaved: {OUTPUT_CI}")
    print(f"Saved: {OUTPUT_DIFF}")


if __name__ == "__main__":
    main()
