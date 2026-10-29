
"""Lookback-window sensitivity analysis for portfolio backtests."""

from pathlib import Path

import pandas as pd

from src.portfolio_backtest import (
    TRAIN_FRACTION,
    TRANSACTION_COST,
    load_returns,
    backtest,
    calculate_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

LOOKBACK_SCENARIOS = [126, 252, 504]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()
    split = int(len(returns) * TRAIN_FRACTION)
    test_start = returns.index[split]
    test_dates = returns.index[split:]

    print(f"Fixed test period: {test_start.date()} to {returns.index.max().date()}")
    print(f"Test observations: {len(test_dates)}")
    print(f"Transaction cost: {TRANSACTION_COST:.2%}")

    all_metrics = []

    for lookback in LOOKBACK_SCENARIOS:
        # Ensure every scenario has enough data before its first test date.
        available_history = int((returns.index < test_start).sum())
        if available_history < lookback:
            raise ValueError(
                f"{lookback}-day lookback requires {lookback} prior observations, "
                f"but only {available_history} are available."
            )

        print(f"\nRunning {lookback}-day lookback")

        daily_results, _ = backtest(
            returns,
            test_start,
            transaction_cost_rate=TRANSACTION_COST,
            lookback=lookback,
        )

        # Confirm every scenario uses the exact same out-of-sample dates.
        for strategy, frame in daily_results.items():
            if not frame.index.equals(test_dates):
                raise ValueError(
                    f"Test dates differ for {strategy} at lookback {lookback}."
                )

        metrics = calculate_metrics(daily_results)
        metrics.insert(0, "lookback_days", lookback)
        metrics.insert(1, "transaction_cost_rate", TRANSACTION_COST)
        all_metrics.append(metrics)

    results = pd.concat(all_metrics, ignore_index=True)

    output_path = REPORT_DIR / "lookback_sensitivity.csv"
    results.to_csv(output_path, index=False)

    columns = [
        "lookback_days",
        "strategy",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio_rf0",
        "maximum_drawdown",
        "total_transaction_cost",
    ]

    print("\nLookback sensitivity results:")
    print(
        results[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
