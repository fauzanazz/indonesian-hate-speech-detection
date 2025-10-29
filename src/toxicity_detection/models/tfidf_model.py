import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import numpy.typing as npt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from toxicity_detection.data.preprocessor import IndonesianTextPreprocessor
from toxicity_detection.models.base import BaseModel
from toxicity_detection.utils.logging import setup_logger
from toxicity_detection.utils.metrics import calculate_ece

logger = setup_logger(__name__)


class TFIDFModel(BaseModel):
    def __init__(
        self,
        name: str = "tfidf_lr",
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 1.0,
        max_iter: int = 1000,
        use_calibration: bool = True,
        calibration_method: str = "isotonic",
        **preprocessor_kwargs: Any,
    ) -> None:
        super().__init__(name)

        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.max_iter = max_iter
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.preprocessor = IndonesianTextPreprocessor(**preprocessor_kwargs)

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.9,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w+",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,  
        )

        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced", 
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        )

        self.pipeline: Optional[Pipeline | CalibratedClassifierCV] = None

        logger.info(
            "TF-IDF model initialized",
            max_features=max_features,
            ngram_range=ngram_range,
            C=C,
            use_calibration=use_calibration,
        )

    def train(
        self,
        X_train: List[str],
        y_train: npt.NDArray[Any],
        X_val: Optional[List[str]] = None,
        y_val: Optional[npt.NDArray[Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        logger.info("Training TF-IDF model", n_samples=len(X_train))
        start_time = time.time()
        X_train_processed = self.preprocessor.preprocess_batch(X_train)
        base_pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            ("classifier", self.classifier),
        ])

        if self.use_calibration and X_val is not None and y_val is not None:
            logger.info("Training with calibration", method=self.calibration_method)

            base_pipeline.fit(X_train_processed, y_train)
            X_val_processed = self.preprocessor.preprocess_batch(X_val)
            self.pipeline = CalibratedClassifierCV(
                base_pipeline,
                method=self.calibration_method,
                cv="prefit",
            )
            self.pipeline.fit(X_val_processed, y_val)
            self.is_trained = True 
            y_val_proba = self.predict_proba(X_val)
            ece = calculate_ece(y_val, y_val_proba)
            logger.info("Calibration complete", ece=f"{ece:.4f}")

        else:
            base_pipeline.fit(X_train_processed, y_train)
            self.pipeline = base_pipeline
            self.is_trained = True 
            ece = None
        training_time = time.time() - start_time

        logger.info(
            "Training complete",
            training_time=f"{training_time:.2f}s",
            n_features=len(self.vectorizer.get_feature_names_out()),
        )

        metrics = {
            "training_time": training_time,
            "n_features": len(self.vectorizer.get_feature_names_out()),
            "calibration_ece": ece,
        }

        return metrics

    def predict(self, texts: List[str]) -> npt.NDArray[Any]:
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model must be trained before prediction")

        X_processed = self.preprocessor.preprocess_batch(texts)
        return self.pipeline.predict(X_processed)

    def predict_proba(self, texts: List[str]) -> npt.NDArray[Any]:
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model must be trained before prediction")

        X_processed = self.preprocessor.preprocess_batch(texts)
        proba = self.pipeline.predict_proba(X_processed)
        return proba[:, 1]  

    def get_feature_importance(self, top_k: int = 20) -> Dict[str, List[tuple[str, float]]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        feature_names = self.vectorizer.get_feature_names_out()

        if isinstance(self.pipeline, CalibratedClassifierCV):
            coef = self.pipeline.calibrated_classifiers_[0].estimator.named_steps["classifier"].coef_[0]
        else:
            coef = self.pipeline.named_steps["classifier"].coef_[0]

        top_toxic_idx = np.argsort(coef)[-top_k:][::-1]
        top_non_toxic_idx = np.argsort(coef)[:top_k]

        return {
            "toxic": [(feature_names[i], float(coef[i])) for i in top_toxic_idx],
            "non_toxic": [(feature_names[i], float(coef[i])) for i in top_non_toxic_idx],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / f"{self.name}_pipeline.joblib"
        joblib.dump(self.pipeline, model_path)
        preprocessor_path = path / f"{self.name}_preprocessor.joblib"
        joblib.dump(self.preprocessor, preprocessor_path)

        metadata = {
            "name": self.name,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "C": self.C,
            "is_trained": self.is_trained,
        }
        metadata_path = path / f"{self.name}_metadata.joblib"
        joblib.dump(metadata, metadata_path)

        logger.info("Model saved", path=str(path))

    def load(self, path: Path) -> None:
        path = Path(path)
        model_path = path / f"{self.name}_pipeline.joblib"
        self.pipeline = joblib.load(model_path)
        preprocessor_path = path / f"{self.name}_preprocessor.joblib"
        self.preprocessor = joblib.load(preprocessor_path)
        metadata_path = path / f"{self.name}_metadata.joblib"
        metadata = joblib.load(metadata_path)
        self.name = metadata["name"]
        self.is_trained = metadata["is_trained"]
        logger.info("Model loaded", path=str(path))

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update({
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "C": self.C,
            "use_calibration": self.use_calibration,
            "calibration_method": self.calibration_method,
        })
        return params
