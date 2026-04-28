# FashionMNIST ML System

This project now covers both:
- Practical Task 6: API + frontend + MLflow.
- Practical Task 7: batch prediction pipeline with PostgreSQL input and prediction storage. The batch task requires reading input rows from a database, generating predictions, writing results back, and running automatically on a schedule.

## Project structure

```text
fashion-mnist-ml-system/
├── app/
│   ├── api/
│   ├── batch/
│   ├── core/
│   ├── db/
│   ├── frontend/
│   └── training/
├── artifacts/
├── data/
├── mlflow/
├── mlflow_artifacts/
├── notebooks/
├── requirements/
├── scripts/
├── sql/
├── Dockerfile.api
├── Dockerfile.frontend
├── Dockerfile.mlflow
├── Dockerfile.train
└── docker-compose.yml
```

## Main features

### FastAPI
- `GET /` - confirms the API is running
- `GET /health` - healthcheck
- `POST /predict` - accepts exactly 784 pixels and returns the predicted FashionMNIST class

### Streamlit frontend
- Upload image inference
- Sample inference from `fashion_test.csv`
- Raw JSON inference with `pixels`
- PostgreSQL tab that can:
  - create DB tables
  - insert sample rows into `input_data`
  - run batch prediction
  - display row counts and recent predictions

### MLflow
The training script:
- creates an experiment
- logs model parameters
- logs metrics like accuracy and macro F1
- logs artifacts
- registers the model in the MLflow Model Registry

### Batch prediction pipeline
- Reads rows from `input_data`
- Loads the trained model from `artifacts/model.joblib`
- Writes predictions to `predictions`
- Supports one-off execution and scheduled execution

## PostgreSQL schema

Tables created by the app:
- `input_data`
  - `id`
  - `source_label`
  - `pixel_0 ... pixel_783`
- `predictions`
  - `id`
  - `input_data_id`
  - `prediction`
  - `prediction_class_name`
  - `prediction_timestamp`
  - `model_version`

DDL is also stored in `sql/init_batch_prediction_tables.sql`.

## Local run

### 1. Install dependencies
```bash
pip install -r requirements/api.txt
pip install -r requirements/frontend.txt
pip install -r requirements/train.txt
pip install -r requirements/mlflow.txt
```

### 2. Start MLflow locally
```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts
```

### 3. Train and register the model
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m app.training.train_with_mlflow --register-model
```

### 4. Run API
```bash
uvicorn app.api.main:app --reload
```

### 5. Run Streamlit
For local PostgreSQL with the credentials provided by the user:
```bash
export API_BASE_URL=http://localhost:8000
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=postgres
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
streamlit run app/frontend/streamlit_app.py
```

### 6. Seed PostgreSQL from CSV
```bash
python scripts/seed_postgres_from_csv.py --limit 100 --clear-existing
```

### 7. Run batch prediction once
```bash
python -m app.batch.run_batch_prediction --limit 100 --model-version local-rf-v1
```

### 8. Run scheduler every 5 minutes
```bash
BATCH_INTERVAL_SECONDS=300 python -m app.batch.scheduler
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

### Optional scheduler service
This service reads and writes to the host PostgreSQL through `host.docker.internal`:
```bash
docker compose --profile batch up --build batch_scheduler
```

### Open in browser
- FastAPI docs: `http://localhost:8000/docs`
- Streamlit frontend: `http://localhost:8501`
- MLflow UI: `http://localhost:5001`

## PostgreSQL connection values used in Streamlit
- host: `localhost` for local Python run, `host.docker.internal` for Dockerized Streamlit
- port: `5432`
- database: `postgres`
- user: `postgres`
- password: `postgres`
- JDBC URL: `jdbc:postgresql://localhost:5432/postgres`

## Notes
- The API expects `artifacts/model.joblib` and `artifacts/class_names.json`.
- If they do not exist yet, `POST /predict` returns a helpful 503 error.
- The batch pipeline uses only rows that do not yet have a prediction.
- The original notebook remains available in `notebooks/` for experimentation.
