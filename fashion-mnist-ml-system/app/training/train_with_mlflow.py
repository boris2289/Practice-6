from __future__ import annotations

import argparse
import json
import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from app.training.data import load_datasets, prepare_features
from app.training.pipeline import evaluate_model, save_artifacts, train_random_forest


def str_to_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def run_training(
    *,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
    experiment_name: str = "fashion-mnist-experiment",
    model_name: str = "fashion-mnist-random-forest",
    register_model: bool = True,
    tracking_uri: str | None = None,
) -> dict:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    train_df, test_df = load_datasets()
    x_train, y_train, x_test, y_test = prepare_features(train_df, test_df)

    with mlflow.start_run(run_name="random-forest-fashion-mnist") as run:
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "random_state": random_state,
                "n_jobs": n_jobs,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "n_features": int(x_train.shape[1]),
            }
        )

        model = train_random_forest(
            x_train,
            y_train,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        accuracy, f1_macro, report = evaluate_model(model, x_test, y_test)
        artifact_info = save_artifacts(model, accuracy, f1_macro, report)

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
            }
        )
        mlflow.log_artifact(artifact_info.model_path, artifact_path="model_artifacts")
        mlflow.log_artifact(artifact_info.class_names_path, artifact_path="model_artifacts")
        mlflow.log_artifact(artifact_info.metrics_path, artifact_path="metrics")
        mlflow.log_artifact(artifact_info.report_path, artifact_path="reports")

        input_example = x_test.iloc[:1]
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
        )

        model_version = None
        if register_model:
            registration = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=model_name,
            )
            model_version = registration.version

        result = {
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri(),
            "model_name": model_name,
            "model_version": model_version,
            "logged_model_uri": model_info.model_uri,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "artifacts": {
                "model_path": artifact_info.model_path,
                "class_names_path": artifact_info.class_names_path,
                "metrics_path": artifact_info.metrics_path,
                "report_path": artifact_info.report_path,
            },
        }

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FashionMNIST model and log to MLflow.")
    parser.add_argument("--n-estimators", type=int, default=int(os.getenv("N_ESTIMATORS", "200")))
    parser.add_argument("--random-state", type=int, default=int(os.getenv("RANDOM_STATE", "42")))
    parser.add_argument("--n-jobs", type=int, default=int(os.getenv("N_JOBS", "-1")))
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "fashion-mnist-experiment"),
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "fashion-mnist-random-forest"),
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        default=str_to_bool(os.getenv("REGISTER_MODEL", "true")),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_training(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        register_model=args.register_model,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
