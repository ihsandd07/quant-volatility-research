
"""Transaction-cost sensitivity analysis for portfolio backtests."""

from pathlib import Path

import pandas as pd

from src.portfolio_backtest import (
    TRAIN_FRACTION,
    load_returns,
    backtest,
    calculate_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

# 5, 10, and 20 basis points per unit of absolute weight turnover.
COST_SCENARIOS = {
    "5_bps": 0.0005,
    "10_bps": 0.0010,
    "20_bps": 0.0020,
}


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()
    split = int(len(returns) * TRAIN_FRACTION)
    test_start = returns.index[split]

    all_metrics = []

    print(f"Test period: {test_start.date()} to {returns.index.max().date()}")
    print(f"Test observations: {len(returns.loc[test_start:])}")

    for scenario, cost_rate in COST_SCENARIOS.items():
        print(f"\nRunning {scenario} (cost rate = {cost_rate:.2%})")

        daily_results, _ = backtest(
            returns,
            test_start,
            transaction_cost_rate=cost_rate,
        )

        metrics = calculate_metrics(daily_results)
        metrics.insert(0, "cost_scenario", scenario)
        metrics.insert(1, "transaction_cost_rate", cost_rate)
        all_metrics.append(metrics)

    results = pd.concat(all_metrics, ignore_index=True)

    output_path = REPORT_DIR / "transaction_cost_sensitivity.csv"
    results.to_csv(output_path, index=False)

    display_columns = [
        "cost_scenario",
        "strategy",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "total_transaction_cost",
    ]

    print("\nTransaction-cost sensitivity results:")
    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
