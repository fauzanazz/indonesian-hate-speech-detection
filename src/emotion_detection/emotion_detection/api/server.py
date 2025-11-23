"""FastAPI server exposing emotion detection predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from ..config import EmotionSettings
from ..models import get_model_cls, list_models
from ..models.base import BaseEmotionModel
from ..preprocess import clean_corpus


class PredictItem(BaseModel):
    text: str
    label: str
    scores: Optional[Dict[str, float]] = None


class PredictRequest(BaseModel):
    text: Optional[str] = Field(None, description="Single text input")
    texts: Optional[List[str]] = Field(None, description="Batch text input")

    @model_validator(mode="after")
    def validate_payload(self) -> "PredictRequest":
        if not self.text and not self.texts:
            raise ValueError("Provide either 'text' or 'texts'.")
        return self


class PredictResponse(BaseModel):
    predictions: List[PredictItem]


@dataclass
class AppState:
    model_name: str
    artefact_dir: Path
    settings: EmotionSettings
    model: Optional[BaseEmotionModel] = None


def create_app(
    *,
    model_name: str,
    artefact_dir: Path,
    settings: Optional[EmotionSettings] = None,
) -> FastAPI:
    state = AppState(
        model_name=model_name,
        artefact_dir=artefact_dir,
        settings=settings or EmotionSettings(),
    )

    app = FastAPI(title="Emotion Detection API", version="1.0.0")

    @app.on_event("startup")
    def _load_model() -> None:  # pragma: no cover - exercised via integration
        try:
            cls = get_model_cls(state.model_name)
        except KeyError as exc:  # pragma: no cover - surfacing to startup logs
            raise RuntimeError(str(exc)) from exc
        state.model = cls.load(state.artefact_dir)

    def get_model() -> BaseEmotionModel:
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        return state.model

    def prepare_texts(texts: List[str]) -> List[str]:
        if state.model_name == "indobert":
            return texts
        return clean_corpus(texts)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "model": state.model_name}

    @app.get("/models")
    def models() -> Dict[str, object]:
        return {
            "loaded": state.model_name,
            "available": list_models(),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        payload: PredictRequest,
        model: BaseEmotionModel = Depends(get_model),
    ) -> PredictResponse:
        if payload.texts:
            raw_texts = payload.texts
        else:
            raw_texts = [payload.text or ""]

        processed = prepare_texts(raw_texts)
        predictions = model.predict(processed)
        try:
            probabilities = model.predict_proba(processed)
        except AttributeError:
            probabilities = [None] * len(predictions)

        items = [
            PredictItem(text=raw, label=pred, scores=proba)
            for raw, pred, proba in zip(raw_texts, predictions, probabilities)
        ]

        return PredictResponse(predictions=items)

    return app


