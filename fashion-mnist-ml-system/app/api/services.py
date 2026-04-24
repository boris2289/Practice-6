from __future__ import annotations

import json

import joblib
import numpy as np

from app.core.paths import CLASS_NAMES_PATH, MODEL_PATH


class ModelNotReadyError(RuntimeError):
    pass


class PredictorService:
    def __init__(self) -> None:
        self.model = None
        self.class_names: list[str] | None = None
        self.reload()

    def reload(self) -> None:
        if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
            self.model = None
            self.class_names = None
            return

        self.model = joblib.load(MODEL_PATH)
        self.class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.class_names is not None

    def predict(self, pixels: list[float]) -> dict:
        if not self.is_ready:
            raise ModelNotReadyError(
                "Model artifacts were not found. Train the model first to create artifacts/model.joblib and artifacts/class_names.json."
            )

        x = np.array(pixels, dtype=np.float32).reshape(1, -1)
        if x.max() > 1.0:
            x = x / 255.0

        pred_idx = int(self.model.predict(x)[0])
        return {
            "predicted_class_id": pred_idx,
            "predicted_class_name": self.class_names[pred_idx],
        }
