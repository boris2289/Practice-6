# FashionMNIST ML System

This project is a cleaned-up continuation of Practical Task 6.

What was added compared with the notebook-only version:
- FastAPI was moved into regular Python modules.
- Docker and docker-compose were moved out of the notebook.
- A simple Streamlit frontend was added.
- MLflow experiment tracking and model registration were added.

## Project structure

```text
fashion-mnist-ml-system/
├── app/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── services.py
│   ├── core/
│   │   ├── constants.py
│   │   └── paths.py
│   ├── frontend/
│   │   └── streamlit_app.py
│   └── training/
│       ├── data.py
│       ├── pipeline.py
│       └── train_with_mlflow.py
├── artifacts/
├── data/
├── mlflow/
├── mlflow_artifacts/
├── notebooks/
├── requirements/
├── Dockerfile.api
├── Dockerfile.frontend
├── Dockerfile.mlflow
├── Dockerfile.train
└── docker-compose.yml
```

## What each part does

### FastAPI
- `GET /` - confirms that the API is running
- `GET /health` - healthcheck
- `POST /predict` - accepts 784 pixels and returns the predicted FashionMNIST class

### Streamlit frontend
- upload an image and send it to the API;
- or use a sample row from `fashion_test.csv`.

### MLflow
The training script:
- creates an experiment;
- logs model parameters;
- logs metrics like accuracy and macro F1;
- logs saved artifacts;
- registers the model in the MLflow Model Registry.

## Local run

### 1. Install dependencies
Use separate environments or install what you need:
```bash
pip install -r requirements/api.txt
pip install -r requirements/frontend.txt
pip install -r requirements/train.txt
```

### 2. Start MLflow locally
```bash
pip install -r requirements/mlflow.txt
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts
```

### 3. Train and log the model
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m app.training.train_with_mlflow --register-model
```

After training, these files appear in `artifacts/`:
- `model.joblib`
- `class_names.json`
- `metrics.json`
- `classification_report.json`

### 4. Run the API
```bash
uvicorn app.api.main:app --reload
```

### 5. Run Streamlit
```bash
export API_BASE_URL=http://localhost:8000
streamlit run app/frontend/streamlit_app.py
```

## Docker Compose run

### Start services
```bash
docker compose up --build mlflow api frontend
```

### Train the model inside Docker
```bash
docker compose --profile train run --rm trainer
```

### Open in browser
- FastAPI docs: `http://localhost:8000/docs`
- Streamlit frontend: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

## Notes
- The API expects `artifacts/model.joblib` and `artifacts/class_names.json`.
- If they do not exist yet, `POST /predict` returns a helpful 503 error.
- The training code automatically downloads `fashion_train.csv` and `fashion_test.csv` into `data/` if they are missing.
- The original notebook can stay as the experimentation layer, but deployment code is now separated into files.
