from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    pixels: list[float] = Field(..., description="Flattened 28x28 image with exactly 784 values")

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, value: list[float]) -> list[float]:
        if len(value) != 784:
            raise ValueError("Input must contain exactly 784 pixel values.")
        return value


class PredictionResponse(BaseModel):
    predicted_class_id: int
    predicted_class_name: str
