import pytest
from toxicity_detection.data.preprocessor import IndonesianTextPreprocessor, extract_lexical_features


class TestIndonesianTextPreprocessor:
    def test_basic_preprocessing(self) -> None:
        preprocessor = IndonesianTextPreprocessor()
        text = "@USER Ini CONTOH text dengan URL http://example.com"
        result = preprocessor.preprocess(text)

        assert "@USER" not in result
        assert "http://example.com" not in result
        assert result.islower()

    def test_remove_urls(self) -> None:
        preprocessor = IndonesianTextPreprocessor(remove_urls=True)
        text = "Check this out: http://example.com and www.test.com"
        result = preprocessor.preprocess(text)
        assert "http://example.com" not in result
        assert "www.test.com" not in result

    def test_remove_mentions(self) -> None:
        preprocessor = IndonesianTextPreprocessor(remove_mentions=True)
        text = "@user1 halo @user2 bagaimana"
        result = preprocessor.preprocess(text)
        assert "@user1" not in result
        assert "@user2" not in result
        assert "halo" in result

    def test_batch_preprocessing(self) -> None:
        preprocessor = IndonesianTextPreprocessor()
        texts = [
            "Text PERTAMA",
            "Text KEDUA dengan @mention",
            "Text KETIGA dengan http://url.com",
        ]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == 3
        assert all(r.islower() for r in results)


def test_extract_lexical_features() -> None:
    text = "Ini adalah TEXT dengan 3 kalimat! Satu? Dua!"
    features = extract_lexical_features(text)
    assert features["n_chars"] > 0
    assert features["n_words"] > 0
    assert features["n_exclamation"] == 2
    assert features["n_question"] == 1
    assert features["uppercase_ratio"] > 0
