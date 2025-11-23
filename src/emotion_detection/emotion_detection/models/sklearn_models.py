"""Scikit-learn based emotion detection models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from ..registry import register_model
from .base import BaseEmotionModel


class SklearnPipelineModel(BaseEmotionModel):
    """Base class wrapping a TF-IDF + classifier pipeline."""

    def __init__(
        self,
        *,
        max_features: Optional[int] = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.pipeline: Optional[Pipeline] = None

    # --- Abstract hooks -------------------------------------------------
    def _build_classifier(self):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    # --- BaseEmotionModel API -------------------------------------------
    def fit(self, texts: Iterable[str], labels: Iterable[str]) -> None:
        y = self.label_encoder.fit_transform(list(labels))

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )

        classifier = self._build_classifier()
        self.pipeline = Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", classifier),
            ]
        )

        self.pipeline.fit(list(texts), y)

    def predict(self, texts: Iterable[str]) -> List[str]:
        pipeline = self._ensure_pipeline()
        y_pred = pipeline.predict(list(texts))
        return self.label_encoder.inverse_transform(y_pred).tolist()

    def predict_proba(self, texts: Iterable[str]) -> List[dict[str, float]]:
        pipeline = self._ensure_pipeline()
        if not hasattr(pipeline, "predict_proba"):
            raise AttributeError("Underlying model does not support predict_proba")

        probs = pipeline.predict_proba(list(texts))
        classes = self.label_encoder.classes_.tolist()
        return [dict(zip(classes, row.tolist())) for row in probs]

    def save(self, path: Path) -> None:
        pipeline = self._ensure_pipeline()
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path / "model.joblib")
        joblib.dump(self.label_encoder, path / "label_encoder.joblib")

    @classmethod
    def load(cls, path: Path) -> "SklearnPipelineModel":
        pipeline: Pipeline = joblib.load(path / "model.joblib")
        label_encoder: LabelEncoder = joblib.load(path / "label_encoder.joblib")

        instance = cls()
        instance.pipeline = pipeline
        instance.label_encoder = label_encoder
        return instance

    # --- Helpers --------------------------------------------------------
    def _ensure_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self.pipeline


@register_model("logreg")
class LogisticRegressionModel(SklearnPipelineModel):
    """TF-IDF + Logistic Regression."""

    def __init__(
        self,
        *,
        max_features: Optional[int] = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        random_state: int = 42,
        c: float = 1.0,
    ) -> None:
        super().__init__(
            max_features=max_features,
            ngram_range=ngram_range,
            random_state=random_state,
        )
        self.c = c

    def _build_classifier(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.c,
            max_iter=2000,
            random_state=self.random_state,
        )


@register_model("svm")
class LinearSVMModel(SklearnPipelineModel):
    """TF-IDF + linear SVM with probability calibration."""

    def __init__(
        self,
        *,
        max_features: Optional[int] = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        random_state: int = 42,
        c: float = 1.0,
    ) -> None:
        super().__init__(
            max_features=max_features,
            ngram_range=ngram_range,
            random_state=random_state,
        )
        self.c = c

    def _build_classifier(self) -> CalibratedClassifierCV:
        base = LinearSVC(C=self.c, random_state=self.random_state)
        return CalibratedClassifierCV(estimator=base, cv=3)


@register_model("nb")
class MultinomialNBModel(SklearnPipelineModel):
    """TF-IDF + Multinomial Naive Bayes."""

    def __init__(
        self,
        *,
        max_features: Optional[int] = 20000,
        ngram_range: tuple[int, int] = (1, 2),
        alpha: float = 1.0,
    ) -> None:
        super().__init__(
            max_features=max_features,
            ngram_range=ngram_range,
            random_state=0,
        )
        self.alpha = alpha

    def _build_classifier(self) -> MultinomialNB:
        return MultinomialNB(alpha=self.alpha)

