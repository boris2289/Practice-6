from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

TRAIN_CSV_PATH = DATA_DIR / "fashion_train.csv"
TEST_CSV_PATH = DATA_DIR / "fashion_test.csv"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / "classification_report.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
