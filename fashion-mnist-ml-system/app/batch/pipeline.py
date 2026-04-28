from __future__ import annotations

from datetime import datetime, timezone

from app.api.services import PredictorService
from app.db.repository import (
    PIXEL_COLUMNS,
    fetch_input_rows_without_predictions,
    insert_predictions,
)


def run_batch_prediction(limit: int = 100, model_version: str | None = None) -> dict:
    predictor = PredictorService()
    if not predictor.is_ready:
        raise RuntimeError('Model artifacts are not ready. Train the model first.')

    rows = fetch_input_rows_without_predictions(limit=limit)
    if not rows:
        return {
            'rows_read': 0,
            'rows_written': 0,
            'message': 'No new rows found in input_data.',
        }

    prediction_rows = []
    for row in rows:
        pixels = [float(row[col]) for col in PIXEL_COLUMNS]
        result = predictor.predict(pixels)
        prediction_rows.append(
            {
                'input_data_id': row['id'],
                'prediction': result['predicted_class_id'],
                'prediction_class_name': result['predicted_class_name'],
                'prediction_timestamp': datetime.now(timezone.utc),
                'model_version': model_version,
            }
        )

    rows_written = insert_predictions(prediction_rows)
    return {
        'rows_read': len(rows),
        'rows_written': rows_written,
        'message': 'Batch prediction completed successfully.',
    }
