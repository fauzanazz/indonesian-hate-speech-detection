import numpy as np
import pytest
from toxicity_detection.models.tfidf_model import TFIDFModel


class TestTFIDFModel:
    @pytest.fixture
    def sample_data(self) -> tuple[list[str], np.ndarray]:
        X = [
            "ini teks yang baik dan positif",
            "ini teks yang buruk dan negatif",
            "teks netral biasa saja",
            "teks sangat buruk toxic",
        ]
        y = np.array([0, 1, 0, 1])
        return X, y

    def test_model_training(self, sample_data: tuple[list[str], np.ndarray]) -> None:
        X, y = sample_data
        model = TFIDFModel(max_features=100)
        metrics = model.train(X, y)
        assert model.is_trained
        assert "training_time" in metrics
        assert model.pipeline is not None

    def test_model_prediction(self, sample_data: tuple[list[str], np.ndarray]) -> None:
        X, y = sample_data
        model = TFIDFModel(max_features=100)
        model.train(X, y)
        test_texts = ["teks yang sangat baik", "teks yang sangat buruk"]
        predictions = model.predict(test_texts)

        assert len(predictions) == 2
        assert all(p in [0, 1] for p in predictions)

    def test_model_predict_proba(self, sample_data: tuple[list[str], np.ndarray]) -> None:
        X, y = sample_data
        model = TFIDFModel(max_features=100)
        model.train(X, y)
        test_texts = ["teks yang baik"]
        probas = model.predict_proba(test_texts)
        assert len(probas) == 1
        assert 0 <= probas[0] <= 1

    def test_model_not_trained_error(self) -> None:
        model = TFIDFModel()
        with pytest.raises(ValueError, match="must be trained"):
            model.predict(["test"])


def test_model_reproducibility() -> None:
    X = ["good text", "bad toxic text", "neutral text"] * 10
    y = np.array([0, 1, 0] * 10)
    model1 = TFIDFModel(max_features=50)
    model1.train(X, y)
    model2 = TFIDFModel(max_features=50)
    model2.train(X, y)
    test_texts = ["test text"]
    pred1 = model1.predict_proba(test_texts)
    pred2 = model2.predict_proba(test_texts)
    assert np.allclose(pred1, pred2)
