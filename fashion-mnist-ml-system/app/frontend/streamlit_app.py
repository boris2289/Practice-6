from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps

from app.batch.pipeline import run_batch_prediction
from app.db.config import settings
from app.db.repository import (
    fetch_recent_predictions,
    get_table_counts,
    initialize_schema,
    seed_input_data_from_dataframe,
)
from app.training.data import load_datasets

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
TEST_CSV_PATH = Path(os.getenv('TEST_CSV_PATH', '/app/data/fashion_test.csv'))
DEFAULT_SEED_LIMIT = int(os.getenv('DEFAULT_SEED_LIMIT', '100'))


@st.cache_data
def load_test_csv(path: str) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def call_prediction_api(pixels: list[float]) -> dict:
    response = requests.post(
        f'{API_BASE_URL}/predict',
        json={'pixels': pixels},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def preprocess_uploaded_image(uploaded_file, invert_colors: bool) -> tuple[np.ndarray, list[float]]:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    if invert_colors:
        image = ImageOps.invert(image)
    arr = np.array(image, dtype=np.float32)
    return arr, arr.flatten().tolist()


st.set_page_config(page_title='FashionMNIST Frontend', layout='wide')
st.title('FashionMNIST Demo + Batch Prediction')
st.write('Streamlit UI for API inference and local PostgreSQL batch prediction pipeline.')

with st.sidebar:
    st.subheader('API')
    st.code(API_BASE_URL)
    if st.button('Check API health'):
        try:
            health = requests.get(f'{API_BASE_URL}/health', timeout=10).json()
            st.success(str(health))
        except Exception as exc:
            st.error(f'API is unavailable: {exc}')

    st.subheader('PostgreSQL')
    st.code(settings.jdbc_url)
    st.caption(f'user={settings.user}')

api_tab, csv_tab, json_tab, db_tab = st.tabs(
    ['Upload image', 'Sample from test CSV', 'Paste JSON', 'PostgreSQL batch pipeline']
)

with api_tab:
    st.write('Upload an image, it will be converted to grayscale and resized to 28x28.')
    uploaded_file = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])
    invert_colors = st.checkbox('Invert colors before prediction', value=False)

    if uploaded_file is not None:
        arr, pixels = preprocess_uploaded_image(uploaded_file, invert_colors)
        st.image(arr.astype('uint8'), caption='Processed 28x28 image', width=280, clamp=True)
        if st.button('Predict uploaded image'):
            try:
                result = call_prediction_api(pixels)
                st.success(
                    f"Prediction: {result['predicted_class_name']} (class id = {result['predicted_class_id']})"
                )
                st.json(result)
            except Exception as exc:
                st.error(f'Prediction failed: {exc}')

with csv_tab:
    df = load_test_csv(str(TEST_CSV_PATH))
    if df is None:
        st.info(f'{TEST_CSV_PATH.name} was not found in {TEST_CSV_PATH.parent}.')
    else:
        if 'sample_row_index' not in st.session_state:
            st.session_state.sample_row_index = 0

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('Pick random row'):
                st.session_state.sample_row_index = random.randint(0, len(df) - 1)
        with col2:
            selected_index = st.number_input(
                'Row index from fashion_test.csv',
                min_value=0,
                max_value=max(len(df) - 1, 0),
                value=int(st.session_state.sample_row_index),
                step=1,
            )
        st.session_state.sample_row_index = int(selected_index)

        row = df.iloc[int(st.session_state.sample_row_index)]
        target_col = 'label' if 'label' in df.columns else 'y'
        true_label = int(row[target_col])
        pixels = row.drop(labels=[target_col]).to_numpy(dtype=np.float32)
        image = pixels.reshape(28, 28)

        st.image(image.astype('uint8'), caption=f'True label id = {true_label}', width=280, clamp=True)
        if st.button('Predict sample row'):
            try:
                result = call_prediction_api(pixels.tolist())
                st.success(
                    f"Prediction: {result['predicted_class_name']} (class id = {result['predicted_class_id']})"
                )
                st.write(f'True label id from CSV: {true_label}')
                st.json(result)
            except Exception as exc:
                st.error(f'Prediction failed: {exc}')

with json_tab:
    st.write('Paste JSON payload with exactly 784 values in `pixels`.')
    sample_payload = {'pixels': [0] * 784}
    json_text = st.text_area(
        'JSON payload',
        value=json.dumps(sample_payload, ensure_ascii=False, indent=2),
        height=320,
    )

    if st.button('Predict from JSON'):
        try:
            payload = json.loads(json_text)
            if 'pixels' not in payload:
                st.error("JSON must contain the key 'pixels'.")
            elif not isinstance(payload['pixels'], list):
                st.error("'pixels' must be a list.")
            elif len(payload['pixels']) != 784:
                st.error("'pixels' must contain exactly 784 values.")
            else:
                result = call_prediction_api(payload['pixels'])
                st.success(
                    f"Prediction: {result['predicted_class_name']} (class id = {result['predicted_class_id']})"
                )
                st.json(result)
        except json.JSONDecodeError:
            st.error('Invalid JSON format.')
        except Exception as exc:
            st.error(f'Prediction failed: {exc}')

with db_tab:
    st.subheader('Local PostgreSQL batch prediction pipeline')
    st.caption('This matches the task: read input_data from DB, run model, write predictions back, and support scheduling.')

    left, right = st.columns([1, 1])
    with left:
        seed_limit = st.number_input('Rows to insert into input_data', min_value=1, max_value=10000, value=DEFAULT_SEED_LIMIT)
        clear_existing = st.checkbox('Clear existing input_data and predictions before insert', value=False)
        if st.button('1) Initialize DB tables'):
            try:
                initialize_schema()
                st.success('Tables input_data and predictions are ready.')
            except Exception as exc:
                st.error(f'Initialization failed: {exc}')

        if st.button('2) Insert sample rows from fashion_test.csv into PostgreSQL'):
            try:
                initialize_schema()
                _, test_df = load_datasets()
                inserted = seed_input_data_from_dataframe(test_df, limit=int(seed_limit), clear_existing=clear_existing)
                st.success(f'Inserted {inserted} rows into input_data.')
            except Exception as exc:
                st.error(f'Insert failed: {exc}')

        batch_limit = st.number_input('Batch size for prediction run', min_value=1, max_value=10000, value=100)
        model_version = st.text_input('Model version label to store', value='local-rf-v1')
        if st.button('3) Run batch prediction now'):
            try:
                result = run_batch_prediction(limit=int(batch_limit), model_version=model_version)
                st.success('Batch prediction completed.')
                st.json(result)
            except Exception as exc:
                st.error(f'Batch prediction failed: {exc}')

    with right:
        if st.button('Refresh DB stats'):
            st.cache_data.clear()
        try:
            counts = get_table_counts()
            st.metric('Rows in input_data', counts['input_data'])
            st.metric('Rows in predictions', counts['predictions'])
            recent = fetch_recent_predictions(limit=20)
            if not recent.empty:
                st.dataframe(recent, use_container_width=True)
            else:
                st.info('No predictions yet.')
        except Exception as exc:
            st.warning(f'Could not read DB stats yet: {exc}')

    st.markdown('''
**Scheduler commands**

Run one batch manually:
```bash
python -m app.batch.run_batch_prediction --limit 100 --model-version local-rf-v1
```

Run continuously every 5 minutes:
```bash
BATCH_INTERVAL_SECONDS=300 python -m app.batch.scheduler
```
''')
