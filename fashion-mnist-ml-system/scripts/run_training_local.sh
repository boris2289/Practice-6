#!/usr/bin/env bash
set -euo pipefail

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-fashion-mnist-experiment}"
export MODEL_NAME="${MODEL_NAME:-fashion-mnist-random-forest}"

python -m app.training.train_with_mlflow --register-model
