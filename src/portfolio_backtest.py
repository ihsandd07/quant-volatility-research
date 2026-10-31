
"""Monthly portfolio backtest with forecast-based exposure control.

Strategies:
1. Equal weight
2. Global minimum variance
3. Maximum Sharpe (zero risk-free rate)
4. Historical-volatility-targeted minimum variance
5. Random-Forest-SPY-volatility-targeted equal weight

The SPY forecast strategy uses a one-step-ahead SPY variance forecast as
a market-risk signal. It does NOT directly forecast portfolio variance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "simple_returns.csv"
FORECAST_PATH = ROOT / "reports" / "walk_forward_test_forecasts.csv"
REPORT_DIR = ROOT / "reports"

TRADING_DAYS = 252
LOOKBACK = 252
TRAIN_FRACTION = 0.75
TRANSACTION_COST = 0.001  # 10 basis points per dollar traded
TARGET_VOL = 0.10          # 10% annualized
MAX_WEIGHT = 0.40          # maximum weight in any single ETF

ASSETS = ["SPY", "IWM", "EFA", "AGG", "GLD", "VNQ"]

STRATEGIES = [
    "volatility_target",
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "spy_vol_target",
]


def load_returns():
    """Load and validate daily simple returns."""
    returns = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
    ).sort_index()

    if returns.empty:
        raise ValueError("Return dataset is empty.")

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates found in return data.")

    if returns.isna().any().any():
        raise ValueError("Missing values found in return data.")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Non-finite values found in return data.")

    if not set(ASSETS).issubset(returns.columns):
        raise ValueError(f"Expected asset columns: {ASSETS}")

    return returns[ASSETS]


def load_rf_forecasts(test_dates):
    """Load Random Forest variance forecasts and align them to test dates."""
    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Missing forecast file: {FORECAST_PATH}\n"
            "Run the walk-forward forecasting pipeline first."
        )

    forecasts = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["Date"],
    )

    required_columns = {"Date", "random_forest"}
    if not required_columns.issubset(forecasts.columns):
        raise ValueError(
            f"Forecast file must contain {required_columns}; "
            f"found {forecasts.columns.tolist()}"
        )

    if forecasts["Date"].duplicated().any():
        raise ValueError("Duplicate dates found in forecast file.")

    forecasts = forecasts.set_index("Date").sort_index()
    aligned = forecasts["random_forest"].reindex(test_dates)

    if aligned.isna().any():
        missing_dates = aligned[aligned.isna()].index
        raise ValueError(
            "Missing Random Forest forecasts for test dates. "
            f"First missing dates: {list(missing_dates[:5])}"
        )

    if not np.isfinite(aligned.to_numpy()).all() or (aligned <= 0).any():
        raise ValueError("Forecast variances must be positive and finite.")

    return aligned


def optimize_weights(history, method, risk_free_rate=0.0):
    """Optimize fully invested, long-only weights using historical data."""
    n_assets = history.shape[1]
    equal_weights = np.full(n_assets, 1.0 / n_assets)

    mean_returns = history.mean().to_numpy()
    covariance = history.cov().to_numpy()
    covariance = covariance + np.eye(n_assets) * 1e-10

    def portfolio_variance(weights):
        return float(weights @ covariance @ weights)

    if method == "equal_weight":
        return equal_weights

    if method == "min_variance":
        objective = portfolio_variance

    elif method == "max_sharpe":

        def objective(weights):
            variance = portfolio_variance(weights)
            if variance <= 1e-16:
                return 1e6

            volatility = np.sqrt(variance)
            excess_return = float(weights @ mean_returns) - risk_free_rate
            return -excess_return / volatility

    else:
        raise ValueError(f"Unknown optimization method: {method}")

    result = minimize(
        objective,
        x0=equal_weights,
        method="SLSQP",
        bounds=[(0.0, MAX_WEIGHT)] * n_assets,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1.0,
            }
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(
            f"{method} optimization failed: {result.message}"
        )

    weights = np.asarray(result.x, dtype=float)

    if not np.isfinite(weights).all():
        raise RuntimeError(f"{method} produced non-finite weights.")

    weights = np.clip(weights, 0.0, MAX_WEIGHT)
    weights = weights / weights.sum()

    if np.max(weights) > MAX_WEIGHT + 1e-6:
        raise RuntimeError(f"{method} violated the maximum-weight constraint.")

    return weights
def get_weights(history, strategy, rf_variance=None):
    """Compute target weights using only information available at rebalance."""
    if strategy == "spy_vol_target":
        if (
            rf_variance is None
            or not np.isfinite(rf_variance)
            or rf_variance <= 0
        ):
            raise ValueError(
                "A positive, finite RF variance forecast is required."
            )

        # Equal-weight base portfolio, scaled using forecast SPY volatility.
        weights = optimize_weights(history, "equal_weight")
        forecast_spy_vol = np.sqrt(rf_variance * TRADING_DAYS)
        exposure = min(1.0, TARGET_VOL / forecast_spy_vol)

        return weights * exposure

    if strategy == "volatility_target":
        # Equal-weight base portfolio; estimate its own trailing volatility.
        weights = optimize_weights(history, "equal_weight")
        covariance = history.cov().to_numpy()

        estimated_variance = float(weights @ covariance @ weights)
        estimated_vol = np.sqrt(
            max(estimated_variance, 0.0) * TRADING_DAYS
        )

        exposure = (
            min(1.0, TARGET_VOL / estimated_vol)
            if estimated_vol > 0
            else 1.0
        )

        # Any uninvested fraction is held as cash earning 0%.
        return weights * exposure

    return optimize_weights(history, strategy)


def backtest(
    returns,
    test_start,
    transaction_cost_rate=TRANSACTION_COST,
    lookback=LOOKBACK,
):
    """Backtest monthly rebalancing with explicit transaction-cost accounting."""
    test_returns = returns.loc[test_start:].copy()

    if test_returns.empty:
        raise ValueError("Test period contains no observations.")

    rf_forecasts = load_rf_forecasts(test_returns.index)

    daily_results = {}
    weight_records = []

    for strategy in STRATEGIES:
        # Holdings are fractions of portfolio wealth at the start of each day.
        actual_weights = np.zeros(returns.shape[1])
        cash_weight = 1.0
        previous_month = None
        strategy_rows = []

        for date, realized_returns in test_returns.iterrows():
            current_month = (date.year, date.month)
            rebalance = current_month != previous_month

            asset_returns = realized_returns.to_numpy(dtype=float)
            transaction_cost = 0.0
            turnover = 0.0

            if rebalance:
                history = returns.loc[returns.index < date].tail(lookback)

                if len(history) < lookback:
                    raise ValueError(
                        f"Insufficient history before {date.date()}: "
                        f"{len(history)} observations."
                    )

                rf_variance = (
                    float(rf_forecasts.loc[date])
                    if strategy == "spy_vol_target"
                    else None
                )

                target_weights = get_weights(
                    history,
                    strategy,
                    rf_variance=rf_variance,
                )

                # Current holdings are measured before today's rebalance.
                # Turnover is the sum of absolute risky-weight changes.
                turnover = float(
                    np.abs(target_weights - actual_weights).sum()
                )
                transaction_cost = turnover * transaction_cost_rate

                # Costs are paid from portfolio wealth. Scale target holdings
                # so post-cost risky assets plus cash equal post-cost wealth.
                post_cost_wealth = 1.0 - transaction_cost
                if post_cost_wealth <= 0:
                    raise ValueError(
                        f"Nonpositive post-cost wealth on {date.date()}."
                    )

                actual_weights = target_weights.copy()
                cash_weight = 1.0 - float(target_weights.sum())

                # Store target allocations before applying today's market move.
                record = {
                    "date": date,
                    "strategy": strategy,
                    "sum_invested": float(target_weights.sum()),
                    "cash_weight": cash_weight,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                }
                record.update(dict(zip(ASSETS, target_weights)))

                if rf_variance is not None:
                    record["rf_forecast_spy_variance"] = rf_variance
                    record["rf_forecast_spy_volatility"] = np.sqrt(
                        max(rf_variance, 0.0) * TRADING_DAYS
                    )

                weight_records.append(record)
                previous_month = current_month

            # Portfolio return before trading costs: risky assets earn their
            # realized returns; cash earns zero.
            gross_return = float(actual_weights @ asset_returns)

            # Transaction costs are charged against pre-return portfolio wealth.
            net_return = (1.0 - transaction_cost) * (1.0 + gross_return) - 1.0

            strategy_rows.append({
                "date": date,
                "strategy": strategy,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "turnover": turnover,
                "risky_exposure": float(actual_weights.sum()),
                "cash_weight": float(cash_weight),
            })

            # Market returns change the value of each risky holding. Cash
            # retains its value. Normalize by post-market, post-cost wealth.
            post_cost_wealth = 1.0 - transaction_cost
            post_market_wealth = post_cost_wealth * (1.0 + gross_return)

            if post_market_wealth <= 0:
                raise ValueError(
                    f"Nonpositive portfolio value on {date.date()}."
                )

            risky_values = (
                actual_weights * post_cost_wealth * (1.0 + asset_returns)
            )
            cash_value = cash_weight * post_cost_wealth

            actual_weights = risky_values / post_market_wealth
            cash_weight = cash_value / post_market_wealth

        daily_results[strategy] = (
            pd.DataFrame(strategy_rows).set_index("date")
        )

    return daily_results, pd.DataFrame(weight_records)


def calculate_metrics(daily_results):
    """Calculate annualized performance statistics from net returns."""
    rows = []

    for strategy, frame in daily_results.items():
        net = frame["net_return"]
        equity = (1.0 + net).cumprod()
        running_peak = equity.cummax()
        drawdown = equity / running_peak - 1.0

        daily_std = net.std(ddof=1)
        annualized_vol = daily_std * np.sqrt(TRADING_DAYS)

        annualized_return = (
            equity.iloc[-1] ** (TRADING_DAYS / len(net)) - 1.0
        )

        sharpe = (
            net.mean() / daily_std * np.sqrt(TRADING_DAYS)
            if daily_std > 0
            else np.nan
        )

        rebalances = frame.loc[frame["turnover"] > 0, "turnover"]
        total_cost = frame["transaction_cost"].sum()

        rows.append({
            "strategy": strategy,
            "test_observations": len(net),
            "cumulative_return": equity.iloc[-1] - 1.0,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio_rf0": sharpe,
            "maximum_drawdown": drawdown.min(),
            "average_daily_transaction_cost": frame[
                "transaction_cost"
            ].mean(),
            "total_transaction_cost": total_cost,
            "average_monthly_turnover": (
                rebalances.mean() if len(rebalances) else 0.0
            ),
            "average_risky_exposure": frame["risky_exposure"].mean(),
            "average_cash_weight": frame["cash_weight"].mean(),
        })

    return pd.DataFrame(rows).sort_values(
        "sharpe_ratio_rf0",
        ascending=False,
    )


def save_outputs(daily_results, weights, metrics):
    """Save backtest outputs and charts."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        {
            strategy: frame[
                [
                    "gross_return",
                    "transaction_cost",
                    "net_return",
                    "turnover",
                    "risky_exposure",
                    "cash_weight",
                ]
            ]
            for strategy, frame in daily_results.items()
        },
        axis=1,
    )
    combined.to_csv(REPORT_DIR / "portfolio_daily_results.csv")

    weights.to_csv(
        REPORT_DIR / "portfolio_rebalance_weights.csv",
        index=False,
    )
    metrics.to_csv(
        REPORT_DIR / "portfolio_performance_metrics.csv",
        index=False,
    )

    equity_curves = pd.DataFrame({
        strategy: (frame["net_return"] + 1.0).cumprod()
        for strategy, frame in daily_results.items()
    })

    ax = equity_curves.plot(figsize=(12, 6))
    ax.set_title(
        "Out-of-Sample Portfolio Growth After Transaction Costs"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "portfolio_equity_curves.png",
        dpi=160,
    )
    plt.close()

    # Separate exposure chart makes the forecast-driven risk rule interpretable.
    exposure = pd.DataFrame({
        strategy: frame["risky_exposure"]
        for strategy, frame in daily_results.items()
    })

    ax = exposure.plot(figsize=(12, 5))
    ax.set_title("Daily Risky-Asset Exposure")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fraction of portfolio invested in risky assets")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "portfolio_exposure.png",
        dpi=160,
    )
    plt.close()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_returns()

    split = int(len(returns) * TRAIN_FRACTION)
    test_start = returns.index[split]

    print(f"Dataset: {returns.shape[0]} days, {returns.shape[1]} ETFs")
    print(
        f"Full period: {returns.index.min().date()} "
        f"to {returns.index.max().date()}"
    )
    print(
        f"Out-of-sample test: {test_start.date()} "
        f"to {returns.index.max().date()}"
    )
    print(f"Lookback: {LOOKBACK} trading days")
    print(f"Transaction cost: {TRANSACTION_COST:.2%} per dollar traded")
    print(f"Maximum asset weight: {MAX_WEIGHT:.0%}")
    print(f"Target volatility: {TARGET_VOL:.0%}")

    daily_results, weights = backtest(returns, test_start)
    metrics = calculate_metrics(daily_results)
    save_outputs(daily_results, weights, metrics)

    print("\nPerformance metrics:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved reports:")
    print("  reports/portfolio_daily_results.csv")
    print("  reports/portfolio_rebalance_weights.csv")
    print("  reports/portfolio_performance_metrics.csv")
    print("  reports/portfolio_equity_curves.png")
    print("  reports/portfolio_exposure.png")


if __name__ == "__main__":
    main()
