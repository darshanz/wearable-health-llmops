import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from healthllm.prompts import FEATURE_COLS

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None


# we alrady did the EDA and know the target column and features to use. so jsut hardcoding here.
TARGET_COL = "readiness"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def project_root():
    return Path(__file__).resolve().parents[2]


def default_data_path():
    # data directory is mounted to /data in the docker container 
    docker_path = Path("/data/pmdata/processed/pmdata_daily_features.csv")
    if docker_path.exists():
        return docker_path

    return project_root().parent / "data" / "pmdata" / "processed" / "pmdata_daily_features.csv"


def build_model():
    # Create simple pipeline with imputation and random forest regressor from Scikit learn
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]
    )

# load the data.
def load_training_data(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Training data must be a .parquet or .csv file")

    missing_cols = [col for col in [*FEATURE_COLS, TARGET_COL] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Training data is missing columns: {missing_cols}")

    model_df = df.dropna(subset=[TARGET_COL]).copy()
    if model_df.empty:
        raise ValueError(f"No rows with non-null {TARGET_COL} found")

    return model_df


def train_random_forest(df):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # train test split with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    metrics = {
        "target": TARGET_COL,
        "feature_columns": FEATURE_COLS,
        "num_rows": int(len(df)),
        "num_train_rows": int(len(X_train)),
        "num_test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_test, predictions)),
    }

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return model, metrics, feature_importance


def save_artifacts(model, metrics, feature_importance, model_path, metrics_path, importance_path):
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    importance_path = Path(importance_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLS,
        "target": TARGET_COL,
        "metrics": metrics,
    }

    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    feature_importance.to_csv(importance_path, index=False)


def log_mlflow_run(model, metrics, importance_path):
    if mlflow is None:
        # mlflow is installed from requirements.txt , just in case to be safe.
        print("MLflow is not installed. Skipping experiment logging.")
        return

    mlflow.set_experiment("wearable-readiness-rf")

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("target", TARGET_COL)
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLS))
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("num_rows", metrics["num_rows"])
        mlflow.log_param("num_train_rows", metrics["num_train_rows"])
        mlflow.log_param("num_test_rows", metrics["num_test_rows"])

        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2", metrics["r2"])

        mlflow.log_artifact(str(importance_path))
        mlflow.sklearn.log_model(model, "model")


def parse_args():
    root = project_root()

    parser = argparse.ArgumentParser(
        description="Train the Random Forest readiness baseline."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path(),
        help="Path to pmdata_daily_features.parquet or .csv.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root / "models" / "readiness_rf.pkl",
        help="Where to save the trained model artifact.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=root / "reports" / "rf_metrics.json",
        help="Where to save evaluation metrics.",
    )
    parser.add_argument(
        "--importance-path",
        type=Path,
        default=root / "reports" / "rf_feature_importance.csv",
        help="Where to save feature importances.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_training_data(args.data_path)
    model, metrics, feature_importance = train_random_forest(df)
    save_artifacts(
        model=model,
        metrics=metrics,
        feature_importance=feature_importance,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        importance_path=args.importance_path,
    )
    log_mlflow_run(
        model=model,
        metrics=metrics,
        importance_path=args.importance_path,
    )

    print(f"Loaded training data: {args.data_path}")
    print(f"Rows with readiness labels: {metrics['num_rows']}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Saved model: {args.model_path}")
    print(f"Saved metrics: {args.metrics_path}")
    print(f"Saved feature importances: {args.importance_path}")
    if mlflow is not None:
        print("Logged MLflow experiment: wearable-readiness-rf")


if __name__ == "__main__":
    main()
