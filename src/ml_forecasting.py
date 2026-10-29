
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETURNS_PATH = PROJECT_ROOT / "data" / "processed" / "simple_returns.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FRACTION = 0.70
RANDOM_STATE = 42


def load_returns() -> pd.Series:
    """Load and validate SPY daily simple returns."""
    data = pd.read_csv(
        RETURNS_PATH,
        index_col="Date",
        parse_dates=True,
    )

    if "SPY" not in data.columns:
        raise ValueError("SPY column not found in returns file.")

    returns = data["SPY"].astype(float).sort_index()

    if returns.empty:
        raise ValueError("No SPY returns found.")

    if returns.index.has_duplicates:
        raise ValueError("Duplicate dates found in SPY returns.")

    if returns.isna().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain invalid observations.")

    return returns


def build_features(returns: pd.Series) -> pd.DataFrame:
    """
    Build features available before each target day's return is observed.

    Each row dated t predicts squared return on date t.
    Features on date t use returns through date t-1 only.
    """
    past_returns = returns.shift(1)
    past_squared_returns = past_returns**2
    past_absolute_returns = past_returns.abs()

    features = pd.DataFrame(index=returns.index)

    # Lagged daily returns and squared returns.
    for lag in [1, 2, 3, 5, 10]:
        features[f"return_lag_{lag}"] = returns.shift(lag)
        features[f"squared_return_lag_{lag}"] = returns.shift(lag) ** 2

    # Rolling volatility, based only on observations before the target date.
    for window in [5, 21, 63]:
        features[f"rolling_variance_{window}"] = (
            past_squared_returns
            .rolling(window=window, min_periods=window)
            .mean()
        )

        features[f"rolling_volatility_{window}"] = (
            past_returns
            .rolling(window=window, min_periods=window)
            .std(ddof=1)
        )

        features[f"mean_absolute_return_{window}"] = (
            past_absolute_returns
            .rolling(window=window, min_periods=window)
            .mean()
        )

    # Target: next day's squared return, aligned with the feature date.
    target = returns**2

    dataset = features.copy()
    dataset["target_variance_proxy"] = target

    # Initial rows lack sufficient lagged history.
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

    return dataset


def evaluate_variance_forecast(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Calculate QLIKE and MSE for a variance forecast."""
    prediction = np.maximum(prediction, 1e-12)

    qlike = np.log(prediction) + actual / prediction
    mse = np.mean((actual - prediction) ** 2)

    return {
        "Mean QLIKE": float(np.mean(qlike)),
        "MSE": float(mse),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    returns = load_returns()
    dataset = build_features(returns)

    feature_columns = [
        column
        for column in dataset.columns
        if column != "target_variance_proxy"
    ]

    # Use a chronological split, never a random split.
    split = int(len(dataset) * TRAIN_FRACTION)

    train = dataset.iloc[:split]
    test = dataset.iloc[split:]

    X_train = train[feature_columns]
    y_train = train["target_variance_proxy"]

    X_test = test[feature_columns]
    y_test = test["target_variance_proxy"].to_numpy()

    # A robust nonlinear benchmark for volatility prediction.
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_test), 1e-12)

    metrics = evaluate_variance_forecast(y_test, predictions)

    forecasts = pd.DataFrame(
        {
            "realized_variance": y_test,
            "random_forest": predictions,
        },
        index=test.index,
    )
    forecasts.index.name = "Date"

    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    metrics_df = pd.DataFrame([metrics], index=["random_forest"])
    metrics_df.index.name = "model"

    forecasts_path = REPORTS_DIR / "spy_random_forest_forecasts.csv"
    metrics_path = REPORTS_DIR / "spy_random_forest_metrics.csv"
    importance_path = REPORTS_DIR / "spy_random_forest_feature_importance.csv"

    forecasts.to_csv(forecasts_path)
    metrics_df.to_csv(metrics_path)
    feature_importance.to_csv(importance_path, index=False)

    print("SPY Random Forest variance forecasting")
    print(f"Total usable observations: {len(dataset):,}")
    print(f"Training observations: {len(train):,}")
    print(f"Test observations: {len(test):,}")
    print(f"Test begins: {test.index.min().date()}")
    print(f"Test ends: {test.index.max().date()}")

    print("\nRandom Forest evaluation:")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.9e}"))

    print("\nTop 10 features:")
    print(feature_importance.head(10).to_string(index=False))

    print("\nSaved:")
    print(forecasts_path)
    print(metrics_path)
    print(importance_path)


if __name__ == "__main__":
    main()
