import re
from typing import List, Optional

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

from toxicity_detection.utils.logging import setup_logger

logger = setup_logger(__name__)


class IndonesianTextPreprocessor:
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        use_stemming: bool = False,
        min_length: int = 2,
    ) -> None:

        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.use_stemming = use_stemming
        self.min_length = min_length

        self.stemmer = None
        if self.use_stemming:
            if not SASTRAWI_AVAILABLE:
                logger.warning("Sastrawi not available, stemming will be disabled")
                self.use_stemming = False
            else:
                factory = StemmerFactory()
                self.stemmer = factory.create_stemmer()
                logger.info("Indonesian stemmer initialized")

        self.stopwords = {
            "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah",
            "ini", "itu", "atau", "dalam", "juga", "akan", "ada", "oleh", "tidak",
            "si", "saya", "anda", "kamu", "dia", "kami", "kita", "mereka",
            "ya", "sudah", "bisa", "mau", "harus", "lebih", "sangat", "saja",
        }

    def clean_text(self, text: str) -> str:
        if self.remove_urls:
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )
            text = re.sub(r"www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", "", text)

        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        text = re.sub(r"\s+", " ", text)

        text = text.strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        tokens = text.split()
        tokens = [t for t in tokens if len(t) >= self.min_length]
        return tokens

    def preprocess(self, text: str, remove_stopwords: bool = False) -> str:
        text = self.clean_text(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation if specified
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        # Tokenize
        tokens = self.tokenize(text)

        # Remove stopwords
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Apply stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        # Reconstruct text
        return " ".join(tokens)

    def preprocess_batch(
        self,
        texts: List[str],
        remove_stopwords: bool = False,
    ) -> List[str]:
        return [self.preprocess(text, remove_stopwords) for text in texts]


def extract_lexical_features(text: str) -> dict[str, int | float]:
    # Character-level features
    n_chars = len(text)
    n_uppercase = sum(1 for c in text if c.isupper())
    n_punctuation = sum(1 for c in text if c in "!?.,;:")
    n_exclamation = text.count("!")
    n_question = text.count("?")

    # Word-level features
    words = text.split()
    n_words = len(words)
    avg_word_length = sum(len(w) for w in words) / n_words if n_words > 0 else 0

    # Special patterns
    n_mentions = len(re.findall(r"@\w+", text))
    n_urls = len(re.findall(r"http[s]?://\S+", text))
    n_repeated_chars = len(re.findall(r"(.)\1{2,}", text))  # e.g., "aaa", "!!!"

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_uppercase": n_uppercase,
        "n_punctuation": n_punctuation,
        "n_exclamation": n_exclamation,
        "n_question": n_question,
        "avg_word_length": avg_word_length,
        "n_mentions": n_mentions,
        "n_urls": n_urls,
        "n_repeated_chars": n_repeated_chars,
        "uppercase_ratio": n_uppercase / n_chars if n_chars > 0 else 0,
        "punctuation_ratio": n_punctuation / n_chars if n_chars > 0 else 0,
    }
