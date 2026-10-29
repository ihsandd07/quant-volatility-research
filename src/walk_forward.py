
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "simple_returns.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FRACTION = 0.60
VALIDATION_END_FRACTION = 0.75
REFIT_EVERY = 21
EWMA_LAMBDA = 0.94
RANDOM_STATE = 42
HAC_LAGS = 5


RF_CANDIDATES = [
    {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 10},
    {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10},
    {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 20},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 20},
]


def load_returns() -> pd.Series:
    data = pd.read_csv(RETURNS_PATH, index_col="Date", parse_dates=True)

    if "SPY" not in data.columns:
        raise ValueError("SPY column not found in returns file.")

    returns = data["SPY"].astype(float).sort_index()

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates found.")

    if returns.isna().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain invalid observations.")

    return returns


def build_dataset(returns: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Features dated t use returns through t-1; target is return[t] squared."""
    features = pd.DataFrame(index=returns.index)

    for lag in [1, 2, 3, 5, 10]:
        features[f"return_lag_{lag}"] = returns.shift(lag)
        features[f"squared_return_lag_{lag}"] = returns.shift(lag) ** 2

    past_returns = returns.shift(1)

    for window in [5, 21, 63]:
        features[f"rolling_variance_{window}"] = (
            past_returns.pow(2).rolling(window).mean()
        )
        features[f"rolling_volatility_{window}"] = (
            past_returns.rolling(window).std(ddof=1)
        )
        features[f"mean_absolute_return_{window}"] = (
            past_returns.abs().rolling(window).mean()
        )

    features = features.replace([np.inf, -np.inf], np.nan)
    dataset = features.copy()
    dataset["target"] = returns.pow(2)
    dataset = dataset.dropna()

    feature_columns = list(features.columns)
    return dataset, feature_columns


def qlike_loss(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    prediction = np.maximum(np.asarray(prediction, dtype=float), 1e-12)
    actual = np.asarray(actual, dtype=float)
    return np.log(prediction) + actual / prediction


def fit_garch(train_returns: pd.Series) -> tuple[float, float, float, float]:
    """Fit GARCH(1,1)-t and return parameters plus final fitted variance."""
    scaled = train_returns * 100

    fitted = arch_model(
        scaled,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    ).fit(disp="off")

    params = fitted.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])

    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("Invalid GARCH parameter estimates.")

    last_variance = float(fitted.conditional_volatility.iloc[-1] ** 2)
    last_return_sq = float(scaled.iloc[-1] ** 2)

    # One-step-ahead variance, in squared percentage-point units.
    next_variance = omega + alpha * last_return_sq + beta * last_variance

    return omega, alpha, beta, next_variance


def forecast_block(
    returns: pd.Series,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    dates: pd.DatetimeIndex,
    rf_params: dict,
) -> pd.DataFrame:
    """Forecast a block, refitting at its start using past data only."""
    rows = []

    for block_start in range(0, len(dates), REFIT_EVERY):
        block_dates = dates[block_start:block_start + REFIT_EVERY]
        first_date = block_dates[0]

        # Strictly historical information at the start of this block.
        history = returns.loc[returns.index < first_date]

        if len(history) < 100:
            raise ValueError("Insufficient history for walk-forward fitting.")

        # Fit GARCH on all returns available before this block.
        omega, alpha, beta, garch_variance = fit_garch(history)

        # Initialize EWMA from the available history, in percentage units.
        history_pct = history * 100
        ewma_variance = float(history_pct.iloc[0] ** 2)

        for value in history_pct.iloc[1:]:
            ewma_variance = (
                EWMA_LAMBDA * ewma_variance
                + (1 - EWMA_LAMBDA) * float(value ** 2)
            )

        # Fit Random Forest only on labeled rows dated before this block.
        rf_train = dataset.loc[dataset.index < first_date]
        rf_train = rf_train.dropna(subset=feature_columns + ["target"])

        rf = RandomForestRegressor(
            **rf_params,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(rf_train[feature_columns], rf_train["target"])

        for date in block_dates:
            prior = returns.loc[returns.index < date]
            if len(prior) < 63:
                raise ValueError("Insufficient history for variance features.")

            # All features at this date are based on past returns only.
            x = dataset.loc[[date], feature_columns]
            rf_prediction = float(rf.predict(x)[0])

            # Historical rolling variance, converted to decimal units.
            historical_pct_variance = float(
                (prior.iloc[-21:] * 100).var(ddof=1)
            )

            # Save predictions before updating with today's observed return.
            rows.append(
                {
                    "Date": date,
                    "realized_variance": float(returns.loc[date] ** 2),
                    "garch": garch_variance / 10000,
                    "ewma": ewma_variance / 10000,
                    "historical": historical_pct_variance / 10000,
                    "random_forest": max(rf_prediction, 1e-12),
                }
            )

            # Update volatility estimates only after today's forecast is saved.
            observed_pct_sq = float((returns.loc[date] * 100) ** 2)

            ewma_variance = (
                EWMA_LAMBDA * ewma_variance
                + (1 - EWMA_LAMBDA) * observed_pct_sq
            )

            garch_variance = (
                omega + alpha * observed_pct_sq + beta * garch_variance
            )

    return pd.DataFrame(rows).set_index("Date").sort_index()


def select_rf_parameters(
    returns: pd.Series,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    validation_dates: pd.DatetimeIndex,
) -> tuple[dict, pd.DataFrame]:
    """Select RF hyperparameters using validation QLIKE only."""
    results = []

    for candidate_number, params in enumerate(RF_CANDIDATES, start=1):
        print(
            f"Validation candidate {candidate_number}/{len(RF_CANDIDATES)}: "
            f"{params}"
        )

        forecasts = forecast_block(
            returns,
            dataset,
            feature_columns,
            validation_dates,
            params,
        )

        losses = qlike_loss(
            forecasts["realized_variance"].to_numpy(),
            forecasts["random_forest"].to_numpy(),
        )

        results.append(
            {
                **params,
                "validation_mean_QLIKE": float(losses.mean()),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        "validation_mean_QLIKE"
    ).reset_index(drop=True)

    best = results_df.iloc[0]
    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": (
            None if pd.isna(best["max_depth"]) else int(best["max_depth"])
        ),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }

    return best_params, results_df


def hac_loss_differential_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    max_lag: int = HAC_LAGS,
) -> dict[str, float]:
    """
    Two-sided HAC test of mean(loss_a - loss_b) = 0.

    Positive mean difference means model A has higher loss than model B.
    Uses a Newey-West long-run variance estimator and normal approximation.
    """
    differential = np.asarray(loss_a) - np.asarray(loss_b)
    differential = differential[np.isfinite(differential)]
    n = len(differential)

    if n < 10:
        raise ValueError("Too few observations for loss-differential testing.")

    mean_difference = float(differential.mean())
    centered = differential - mean_difference

    # Newey-West long-run variance with Bartlett weights.
    long_run_variance = float(np.dot(centered, centered) / n)

    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n
        )
        weight = 1 - lag / (max_lag + 1)
        long_run_variance += 2 * weight * covariance

    long_run_variance = max(long_run_variance, 0.0)
    standard_error = np.sqrt(long_run_variance / n)

    if standard_error == 0:
        statistic = 0.0 if mean_difference == 0 else np.sign(mean_difference) * np.inf
        p_value = 1.0 if mean_difference == 0 else 0.0
    else:
        statistic = mean_difference / standard_error
        p_value = float(2 * norm.sf(abs(statistic)))

    return {
        "mean_loss_difference_A_minus_B": mean_difference,
        "HAC_standard_error": float(standard_error),
        "z_statistic": float(statistic),
        "p_value": float(p_value),
        "observations": int(n),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()
    dataset, feature_columns = build_dataset(returns)

    # Split boundaries are defined from the complete chronological return index.
    n_returns = len(returns)
    train_end_position = int(n_returns * TRAIN_FRACTION)
    validation_end_position = int(n_returns * VALIDATION_END_FRACTION)

    validation_start = returns.index[train_end_position]
    test_start = returns.index[validation_end_position]

    validation_dates = dataset.index[
        (dataset.index >= validation_start) & (dataset.index < test_start)
    ]
    test_dates = dataset.index[dataset.index >= test_start]

    if len(validation_dates) == 0 or len(test_dates) == 0:
        raise ValueError("Validation or test period is empty.")

    print("Selecting Random Forest parameters using validation QLIKE...")
    best_params, validation_results = select_rf_parameters(
        returns,
        dataset,
        feature_columns,
        validation_dates,
    )

    print("\nValidation results:")
    print(validation_results.to_string(index=False))

    print("\nSelected Random Forest parameters:")
    print(best_params)

    print("\nRunning final walk-forward test...")
    test_forecasts = forecast_block(
        returns,
        dataset,
        feature_columns,
        test_dates,
        best_params,
    )

    model_names = ["garch", "ewma", "historical", "random_forest"]
    metric_rows = []
    loss_series = {}

    actual = test_forecasts["realized_variance"].to_numpy()

    for model_name in model_names:
        prediction = test_forecasts[model_name].to_numpy()
        losses = qlike_loss(actual, prediction)
        loss_series[model_name] = losses

        metric_rows.append(
            {
                "model": model_name,
                "Mean QLIKE": float(losses.mean()),
                "MSE": float(np.mean((actual - prediction) ** 2)),
            }
        )

    metrics = pd.DataFrame(metric_rows).set_index("model")
    metrics = metrics.sort_values("Mean QLIKE")


    # Pairwise tests use final-test QLIKE losses only.
    test_rows = []

    for model_a, model_b in combinations(model_names, 2):
        result = hac_loss_differential_test(
            loss_series[model_a],
            loss_series[model_b],
        )

        test_rows.append(
            {
                "model_A": model_a,
                "model_B": model_b,
                **result,
            }
        )

    tests = pd.DataFrame(test_rows)

    # Holm-Bonferroni adjustment across all pairwise comparisons.
    reject, adjusted_pvalues, _, _ = multipletests(
        tests["p_value"].to_numpy(),
        alpha=0.05,
        method="holm",
    )

    tests["holm_adjusted_p_value"] = adjusted_pvalues
    tests["significant_after_holm_5pct"] = reject


    # Save results and forecasts.
    validation_results.to_csv(
        REPORTS_DIR / "walk_forward_validation_results.csv",
        index=False,
    )
    metrics.to_csv(REPORTS_DIR / "walk_forward_test_metrics.csv")
    tests.to_csv(
        REPORTS_DIR / "walk_forward_qlike_hac_tests.csv",
        index=False,
    )
    test_forecasts.to_csv(
        REPORTS_DIR / "walk_forward_test_forecasts.csv",
        index_label="Date",
    )

    print("\nFinal walk-forward test")
    print(f"Validation begins: {validation_dates.min().date()}")
    print(f"Validation ends: {validation_dates.max().date()}")
    print(f"Test begins: {test_forecasts.index.min().date()}")
    print(f"Test ends: {test_forecasts.index.max().date()}")
    print(f"Test observations: {len(test_forecasts):,}")

    print("\nFinal test metrics (lower is better):")
    print(metrics.to_string(float_format=lambda x: f"{x:.9e}"))

    print("\nPairwise HAC tests of QLIKE loss differentials")
    print("Positive difference means model A has higher/worse loss.")
    print("Holm-adjusted p-values control family-wise error at 5%.")

    print(
        tests.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\nSaved reports:")
    print(REPORTS_DIR / "walk_forward_validation_results.csv")
    print(REPORTS_DIR / "walk_forward_test_metrics.csv")
    print(REPORTS_DIR / "walk_forward_qlike_hac_tests.csv")
    print(REPORTS_DIR / "walk_forward_test_forecasts.csv")


if __name__ == "__main__":
    main()
