from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor, execute_values

from app.db.config import PostgresSettings, settings

PIXEL_COLUMNS = [f"pixel_{i}" for i in range(784)]


def get_connection(db_settings: PostgresSettings | None = None):
    db_settings = db_settings or settings
    return psycopg2.connect(db_settings.dsn)


def initialize_schema(db_settings: PostgresSettings | None = None) -> None:
    ddl = build_schema_sql()
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def build_schema_sql() -> str:
    pixel_definitions = ",\n    ".join([f"pixel_{i} INTEGER NOT NULL" for i in range(784)])
    return f"""
CREATE TABLE IF NOT EXISTS input_data (
    id BIGSERIAL PRIMARY KEY,
    source_label INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    {pixel_definitions}
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    input_data_id BIGINT NOT NULL REFERENCES input_data(id) ON DELETE CASCADE,
    prediction INTEGER NOT NULL,
    prediction_class_name TEXT,
    prediction_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    model_version TEXT,
    UNIQUE (input_data_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_input_data_id ON predictions(input_data_id);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_timestamp ON predictions(prediction_timestamp);
"""


def seed_input_data_from_dataframe(
    df: pd.DataFrame,
    limit: int | None = None,
    clear_existing: bool = False,
    db_settings: PostgresSettings | None = None,
) -> int:
    label_col = 'label' if 'label' in df.columns else 'y'
    if label_col not in df.columns:
        raise ValueError("The dataframe must contain a 'label' or 'y' column.")

    work_df = df.head(limit).copy() if limit else df.copy()
    rows: list[tuple[Any, ...]] = []
    columns = ['source_label', *PIXEL_COLUMNS]

    for _, row in work_df.iterrows():
        pixels = row.drop(labels=[label_col]).tolist()
        if len(pixels) != 784:
            raise ValueError('Each row must contain exactly 784 pixels.')
        rows.append((int(row[label_col]), *[int(v) for v in pixels]))

    if not rows:
        return 0

    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            if clear_existing:
                cur.execute('TRUNCATE TABLE predictions RESTART IDENTITY CASCADE;')
                cur.execute('TRUNCATE TABLE input_data RESTART IDENTITY CASCADE;')

            insert_query = sql.SQL('INSERT INTO input_data ({fields}) VALUES %s').format(
                fields=sql.SQL(', ').join(map(sql.Identifier, columns))
            )
            execute_values(cur, insert_query.as_string(cur), rows, page_size=500)
        conn.commit()

    return len(rows)


def fetch_input_rows_without_predictions(
    limit: int = 100,
    db_settings: PostgresSettings | None = None,
) -> list[dict[str, Any]]:
    pixel_select = ', '.join([f'i.pixel_{i}' for i in range(784)])
    query = f"""
        SELECT i.id, i.source_label, {pixel_select}
        FROM input_data i
        LEFT JOIN predictions p ON p.input_data_id = i.id
        WHERE p.input_data_id IS NULL
        ORDER BY i.id
        LIMIT %s
    """
    with get_connection(db_settings) as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
    return [dict(row) for row in rows]


def insert_predictions(
    predictions: list[dict[str, Any]],
    db_settings: PostgresSettings | None = None,
) -> int:
    if not predictions:
        return 0

    rows = [
        (
            item['input_data_id'],
            item['prediction'],
            item.get('prediction_class_name'),
            item.get('prediction_timestamp', datetime.utcnow()),
            item.get('model_version'),
        )
        for item in predictions
    ]

    query = """
        INSERT INTO predictions (
            input_data_id,
            prediction,
            prediction_class_name,
            prediction_timestamp,
            model_version
        ) VALUES %s
        ON CONFLICT (input_data_id) DO UPDATE SET
            prediction = EXCLUDED.prediction,
            prediction_class_name = EXCLUDED.prediction_class_name,
            prediction_timestamp = EXCLUDED.prediction_timestamp,
            model_version = EXCLUDED.model_version
    """
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows, page_size=500)
        conn.commit()
    return len(rows)


def get_table_counts(db_settings: PostgresSettings | None = None) -> dict[str, int]:
    with get_connection(db_settings) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT COUNT(*) FROM input_data;')
            input_count = cur.fetchone()[0]
            cur.execute('SELECT COUNT(*) FROM predictions;')
            prediction_count = cur.fetchone()[0]
    return {'input_data': input_count, 'predictions': prediction_count}


def fetch_recent_predictions(limit: int = 20, db_settings: PostgresSettings | None = None) -> pd.DataFrame:
    query = """
        SELECT
            p.id,
            p.input_data_id,
            i.source_label,
            p.prediction,
            p.prediction_class_name,
            p.model_version,
            p.prediction_timestamp
        FROM predictions p
        INNER JOIN input_data i ON i.id = p.input_data_id
        ORDER BY p.prediction_timestamp DESC, p.id DESC
        LIMIT %s
    """
    with get_connection(db_settings) as conn:
        return pd.read_sql(query, conn, params=(limit,))
