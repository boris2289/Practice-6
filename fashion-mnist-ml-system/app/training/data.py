from pathlib import Path

import pandas as pd

from app.core.paths import DATA_DIR, TEST_CSV_PATH, TRAIN_CSV_PATH

TRAIN_URL = "https://raw.githubusercontent.com/ymattu/fashion-mnist-csv/refs/heads/master/fashion_train.csv"
TEST_URL = "https://raw.githubusercontent.com/ymattu/fashion-mnist-csv/refs/heads/master/fashion_test.csv"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_or_download_csv(local_path: Path, url: str) -> pd.DataFrame:
    ensure_data_dir()
    if local_path.exists():
        return pd.read_csv(local_path)
    df = pd.read_csv(url)
    df.to_csv(local_path, index=False)
    return df


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_or_download_csv(TRAIN_CSV_PATH, TRAIN_URL)
    test_df = load_or_download_csv(TEST_CSV_PATH, TEST_URL)
    return train_df, test_df


def prepare_features(train_df, test_df):

    target_col = "label"

    x_train = train_df.drop(columns=[target_col]).astype("float32") / 255.0
    y_train = train_df[target_col].astype("int64")

    x_test = test_df.drop(columns=[target_col]).astype("float32") / 255.0
    y_test = test_df[target_col].astype("int64")

    return x_train, y_train, x_test, y_test
