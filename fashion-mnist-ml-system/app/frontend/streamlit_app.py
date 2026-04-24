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

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
TEST_CSV_PATH = Path(os.getenv("TEST_CSV_PATH", "/app/data/fashion_test.csv"))


def call_prediction_api(pixels: list[float]) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"pixels": pixels},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def preprocess_uploaded_image(uploaded_file, invert_colors: bool) -> tuple[np.ndarray, list[float]]:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    if invert_colors:
        image = ImageOps.invert(image)
    arr = np.array(image, dtype=np.float32)
    return arr, arr.flatten().tolist()


@st.cache_data
def load_test_csv(path: str) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


st.set_page_config(page_title="FashionMNIST Frontend", layout="centered")
st.title("FashionMNIST Demo")
st.write("Simple Streamlit frontend for the FastAPI prediction service.")

with st.sidebar:
    st.subheader("Connection")
    st.code(API_BASE_URL)
    st.write(f"CSV path: `{TEST_CSV_PATH}`")

    if st.button("Check API health"):
        try:
            health = requests.get(f"{API_BASE_URL}/health", timeout=10).json()
            st.success(f"API status: {health}")
        except Exception as exc:
            st.error(f"API is unavailable: {exc}")

tab_upload, tab_sample, tab_json = st.tabs(
    ["Upload image", "Sample from test CSV", "Paste JSON"]
)

with tab_upload:
    st.write("Upload an image, it will be converted to grayscale and resized to 28x28.")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    invert_colors = st.checkbox("Invert colors before prediction", value=False)

    if uploaded_file is not None:
        arr, pixels = preprocess_uploaded_image(uploaded_file, invert_colors)
        st.image(arr.astype("uint8"), caption="Processed 28x28 image", width=280, clamp=True)

        if st.button("Predict uploaded image"):
            try:
                result = call_prediction_api(pixels)
                st.success(
                    f"Prediction: {result['predicted_class_name']} "
                    f"(class id = {result['predicted_class_id']})"
                )
                st.json(result)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab_sample:
    df = load_test_csv(str(TEST_CSV_PATH))

    if df is None:
        st.info(f"{TEST_CSV_PATH.name} was not found in {TEST_CSV_PATH.parent}.")
    else:
        if "label" not in df.columns:
            st.error("CSV must contain a 'label' column.")
        else:
            if "sample_row_index" not in st.session_state:
                st.session_state.sample_row_index = 0

            if st.button("Pick random row"):
                st.session_state.sample_row_index = random.randint(0, len(df) - 1)

            selected_index = st.number_input(
                "Row index from fashion_test.csv",
                min_value=0,
                max_value=max(len(df) - 1, 0),
                value=int(st.session_state.sample_row_index),
                step=1,
            )
            st.session_state.sample_row_index = int(selected_index)

            row = df.iloc[int(st.session_state.sample_row_index)]
            true_label = int(row["label"])
            pixels = row.drop(labels=["label"]).to_numpy(dtype=np.float32)
            image = pixels.reshape(28, 28)

            st.image(image.astype("uint8"), caption=f"True label id = {true_label}", width=280, clamp=True)

            if st.button("Predict sample row"):
                try:
                    result = call_prediction_api(pixels.tolist())
                    st.success(
                        f"Prediction: {result['predicted_class_name']} "
                        f"(class id = {result['predicted_class_id']})"
                    )
                    st.write(f"True label id from CSV: {true_label}")
                    st.json(result)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

with tab_json:
    st.write("Paste JSON payload with exactly 784 values in `pixels`.")

    sample_payload = {"pixels": [0] * 784}
    json_text = st.text_area(
        "JSON payload",
        value=json.dumps(sample_payload, ensure_ascii=False, indent=2),
        height=320,
    )

    if st.button("Predict from JSON"):
        try:
            payload = json.loads(json_text)

            if "pixels" not in payload:
                st.error("JSON must contain the key `pixels`.")
            elif not isinstance(payload["pixels"], list):
                st.error("`pixels` must be a list.")
            elif len(payload["pixels"]) != 784:
                st.error("`pixels` must contain exactly 784 values.")
            else:
                result = call_prediction_api(payload["pixels"])
                st.success(
                    f"Prediction: {result['predicted_class_name']} "
                    f"(class id = {result['predicted_class_id']})"
                )
                st.json(result)

        except json.JSONDecodeError:
            st.error("Invalid JSON format.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")