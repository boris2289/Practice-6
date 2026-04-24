from fastapi import FastAPI, HTTPException

from app.api.schemas import PredictionRequest, PredictionResponse
from app.api.services import ModelNotReadyError, PredictorService

app = FastAPI(title="FashionMNIST API", version="2.0.0")
predictor = PredictorService()


@app.get("/")
def root() -> dict:
    return {
        "message": "FashionMNIST API is running",
        "model_loaded": predictor.is_ready,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": predictor.is_ready,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**predictor.predict(request.pixels))
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
