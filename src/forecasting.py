
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "simple_returns.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FRACTION = 0.70
ROLLING_WINDOW = 21
EWMA_LAMBDA = 0.94


def load_returns() -> pd.Series:
    """Load and validate SPY daily simple returns."""
    data = pd.read_csv(
        RETURNS_PATH,
        index_col="Date",
        parse_dates=True,
    )

    if "SPY" not in data.columns:
        raise ValueError("SPY column not found in the returns file.")

    returns = data["SPY"].astype(float).sort_index()

    if returns.empty:
        raise ValueError("No SPY returns found.")

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates found in SPY returns.")

    if returns.isna().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("SPY returns contain invalid observations.")

    return returns


def forecast_variances(
    returns: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate chronological one-step-ahead variance forecasts.

    GARCH parameters are estimated on the training sample and then held
    fixed during the test period. GARCH and EWMA variances are updated
    using each observed return only after that day's forecast is recorded.

    Returns:
        forecasts: variance forecasts and squared-return proxy, in decimal units
        test_returns: simple returns for the test period
        garch_parameters: estimated GARCH parameters
    """
    split = int(len(returns) * TRAIN_FRACTION)

    if split <= ROLLING_WINDOW or split >= len(returns):
        raise ValueError("Insufficient observations for train/test split.")

    train = returns.iloc[:split]
    test = returns.iloc[split:]

    # Scale returns to percentage points for numerical stability.
    train_pct = train * 100
    all_pct = returns * 100

    # Fit GARCH(1,1) with Student-t innovations on training data only.
    model = arch_model(
        train_pct,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )
    fitted = model.fit(disp="off")
    garch_parameters = fitted.params.copy()

    omega = float(garch_parameters["omega"])
    alpha = float(garch_parameters["alpha[1]"])
    beta = float(garch_parameters["beta[1]"])

    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("Invalid GARCH parameters estimated.")

    # Check covariance-stationarity of the fitted GARCH variance process.
    if alpha + beta >= 1:
        raise ValueError(
            "Estimated GARCH parameters do not satisfy alpha + beta < 1."
        )

    # Initialize GARCH using the final training return and fitted variance.
    last_train_return_sq = float(train_pct.iloc[-1] ** 2)
    last_train_variance = float(
        fitted.conditional_volatility.iloc[-1] ** 2
    )

    garch_variance = (
        omega
        + alpha * last_train_return_sq
        + beta * last_train_variance
    )

    # Initialize EWMA by recursively processing training squared returns.
    ewma_variance = float(train_pct.iloc[0] ** 2)

    for return_value in train_pct.iloc[1:]:
        ewma_variance = (
            EWMA_LAMBDA * ewma_variance
            + (1 - EWMA_LAMBDA) * float(return_value ** 2)
        )

    rows = []

    for date in test.index:
        # Use only observations strictly before the forecast date.
        history = all_pct.loc[all_pct.index < date]

        if len(history) < ROLLING_WINDOW:
            raise ValueError("Insufficient history for historical variance.")

        historical_variance = float(
            history.iloc[-ROLLING_WINDOW:].var(ddof=1)
        )

        # Record today's forecast before observing today's return.
        rows.append(
            {
                "Date": date,
                "realized_variance": float(all_pct.loc[date] ** 2),
                "historical": historical_variance,
                "ewma": ewma_variance,
                "garch": garch_variance,
            }
        )

        # Update forecasts for the next day using today's observed return.
        observed_return_sq = float(all_pct.loc[date] ** 2)

        ewma_variance = (
            EWMA_LAMBDA * ewma_variance
            + (1 - EWMA_LAMBDA) * observed_return_sq
        )

        garch_variance = (
            omega
            + alpha * observed_return_sq
            + beta * garch_variance
        )

    forecasts = pd.DataFrame(rows).set_index("Date")

    # Convert squared percentage-point returns to squared decimal returns.
    variance_columns = [
        "realized_variance",
        "historical",
        "ewma",
        "garch",
    ]
    forecasts[variance_columns] /= 10000

    prediction_columns = ["historical", "ewma", "garch"]

    if not np.isfinite(forecasts.to_numpy()).all():
        raise ValueError("Forecast table contains invalid values.")

    if (forecasts[prediction_columns] <= 0).any().any():
        raise ValueError("Variance forecasts must be strictly positive.")

    return forecasts, test, garch_parameters


def evaluate_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Evaluate variance forecasts using mean QLIKE and MSE."""
    actual = forecasts["realized_variance"].to_numpy()

    if not np.isfinite(actual).all():
        raise ValueError("Realized variance proxy contains invalid values.")

    if (actual < 0).any():
        raise ValueError("Realized variance proxy cannot be negative.")

    results = {}

    for model_name in ["garch", "ewma", "historical"]:
        prediction = forecasts[model_name].to_numpy()

        if not np.isfinite(prediction).all() or (prediction <= 0).any():
            raise ValueError(
                f"{model_name} forecasts must be finite and positive."
            )

        # QLIKE can be negative when variances are expressed in decimal units.
        qlike = np.log(prediction) + actual / prediction
        mse = np.mean((actual - prediction) ** 2)

        results[model_name] = {
            "Mean QLIKE": float(np.mean(qlike)),
            "MSE": float(mse),
        }

    return pd.DataFrame(results).T.sort_values("Mean QLIKE")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()
    forecasts, test_returns, garch_parameters = forecast_variances(returns)
    metrics = evaluate_forecasts(forecasts)

    print("SPY one-step-ahead variance forecasting")
    print(f"Total observations: {len(returns):,}")
    print(f"Training observations: {len(returns) - len(test_returns):,}")
    print(f"Test observations: {len(test_returns):,}")
    print(f"Test begins: {test_returns.index.min().date()}")
    print(f"Test ends: {test_returns.index.max().date()}")

    print("\nGARCH(1,1) parameter estimates:")
    print(garch_parameters.to_string())

    print("\nForecast evaluation (lower is better):")
    print(metrics.to_string(float_format=lambda x: f"{x:.9e}"))

    forecasts_path = REPORTS_DIR / "spy_variance_forecasts.csv"
    metrics_path = REPORTS_DIR / "spy_forecast_metrics.csv"
    parameters_path = REPORTS_DIR / "spy_garch_parameters.csv"

    forecasts.to_csv(forecasts_path, index_label="Date")
    metrics.to_csv(metrics_path, index_label="model")
    garch_parameters.to_csv(parameters_path, header=["estimate"])

    print("\nSaved:")
    print(forecasts_path)
    print(metrics_path)
    print(parameters_path)


if __name__ == "__main__":
    main()
