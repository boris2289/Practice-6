from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from app.core.constants import CLASS_NAMES
from app.core.paths import (
    ARTIFACTS_DIR,
    CLASSIFICATION_REPORT_PATH,
    CLASS_NAMES_PATH,
    METRICS_PATH,
    MODEL_PATH,
)


@dataclass
class TrainingResult:
    accuracy: float
    f1_macro: float
    report: dict
    model_path: str
    class_names_path: str
    metrics_path: str
    report_path: str


def train_random_forest(
    x_train,
    y_train,
    *,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test) -> tuple[float, float, dict]:
    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1_macro = float(f1_score(y_test, predictions, average="macro"))
    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return accuracy, f1_macro, report


def save_artifacts(model, accuracy: float, f1_macro: float, report: dict) -> TrainingResult:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    CLASS_NAMES_PATH.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps({"accuracy": accuracy, "f1_macro": f1_macro}, indent=2),
        encoding="utf-8",
    )
    CLASSIFICATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return TrainingResult(
        accuracy=accuracy,
        f1_macro=f1_macro,
        report=report,
        model_path=str(MODEL_PATH),
        class_names_path=str(CLASS_NAMES_PATH),
        metrics_path=str(METRICS_PATH),
        report_path=str(CLASSIFICATION_REPORT_PATH),
    )
