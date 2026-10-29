
import numpy as np
import pandas as pd
from pathlib import Path


REPORTS = Path("reports")
ASSETS = ["SPY", "IWM", "EFA", "AGG", "GLD", "VNQ"]
TOL = 1e-8


def load_reports():
    daily_path = REPORTS / "portfolio_daily_results.csv"
    weights_path = REPORTS / "portfolio_rebalance_weights.csv"
    metrics_path = REPORTS / "portfolio_performance_metrics.csv"

    for path in (daily_path, weights_path, metrics_path):
        assert path.exists(), f"Missing report: {path}"

    daily = pd.read_csv(
        daily_path, header=[0, 1], index_col=0
    )
    weights = pd.read_csv(weights_path)
    metrics = pd.read_csv(metrics_path).set_index("strategy")

    return daily, weights, metrics


def test_daily_returns_reconcile_with_saved_metrics():
    daily, _, metrics = load_reports()

    strategies = daily.columns.get_level_values(0).unique()

    for strategy in strategies:
        net = daily[(strategy, "net_return")].astype(float)
        recomputed = (1.0 + net).prod() - 1.0
        saved = float(metrics.loc[strategy, "cumulative_return"])

        assert np.isfinite(net).all()
        assert np.isclose(
            recomputed, saved, atol=TOL, rtol=0
        ), (
            f"{strategy}: recomputed cumulative return "
            f"{recomputed:.10f} != saved {saved:.10f}"
        )


def test_transaction_costs_are_valid():
    daily, _, _ = load_reports()
    strategies = daily.columns.get_level_values(0).unique()

    for strategy in strategies:
        gross = daily[(strategy, "gross_return")].astype(float)
        net = daily[(strategy, "net_return")].astype(float)
        costs = daily[(strategy, "transaction_cost")].astype(float)

        assert np.isfinite(gross).all()
        assert np.isfinite(net).all()
        assert np.isfinite(costs).all()
        assert (costs >= -TOL).all(), (
            f"{strategy}: negative transaction cost found"
        )
        assert (net <= gross + TOL).all(), (
            f"{strategy}: net return exceeds gross return"
        )


def test_daily_exposure_and_cash_sum_to_one():
    daily, _, _ = load_reports()
    strategies = daily.columns.get_level_values(0).unique()

    for strategy in strategies:
        risky = daily[(strategy, "risky_exposure")].astype(float)
        cash = daily[(strategy, "cash_weight")].astype(float)

        assert np.isfinite(risky).all()
        assert np.isfinite(cash).all()
        assert (risky >= -TOL).all(), (
            f"{strategy}: negative risky exposure found"
        )
        assert (cash >= -TOL).all(), (
            f"{strategy}: negative cash weight found"
        )
        assert np.allclose(
            risky + cash, 1.0, atol=TOL, rtol=0
        ), f"{strategy}: risky exposure + cash != 1"


def test_rebalance_weights_respect_constraints():
    _, weights, _ = load_reports()

    assert set(ASSETS).issubset(weights.columns), (
        "Rebalance report is missing asset columns"
    )

    asset_weights = weights[ASSETS].astype(float)
    sums = asset_weights.sum(axis=1)

    assert np.isfinite(asset_weights.to_numpy()).all()
    assert (asset_weights >= -TOL).all().all(), (
        "Negative asset weight found"
    )
    assert (asset_weights <= 0.40 + TOL).all().all(), (
        "Asset weight exceeds 40% cap"
    )
    assert (sums <= 1.0 + TOL).all(), (
        "Risky allocation exceeds 100%"
    )
    assert (sums >= -TOL).all(), (
        "Risky allocation is negative"
    )

    if "cash_weight" in weights.columns:
        cash = weights["cash_weight"].astype(float)
        assert (cash >= -TOL).all()
        assert np.allclose(
            sums + cash, 1.0, atol=TOL, rtol=0
        ), "Rebalance risky allocation + cash != 1"


def test_reports_cover_same_strategies():
    daily, weights, metrics = load_reports()

    daily_strategies = set(
        daily.columns.get_level_values(0).unique()
    )
    weight_strategies = set(weights["strategy"].unique())
    metric_strategies = set(metrics.index)

    assert daily_strategies == weight_strategies, (
        "Daily results and rebalance weights contain different strategies"
    )
    assert daily_strategies == metric_strategies, (
        "Daily results and performance metrics contain different strategies"
    )
