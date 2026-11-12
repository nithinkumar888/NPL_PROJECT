"""Text Preprocessing Module for Telangana Organic Farming Advisory System.

Performs:
- Lowercasing (unicode-aware)
- Tokenization (supporting English, Telugu scripts, numbers, and key agricultural units like percentages)
- Lemmatization (NLTK WordNetLemmatizer with part-of-speech awareness)
"""

import re
import unicodedata
from typing import List, Union
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


def ensure_nltk_resources():
    """Ensure required NLTK datasets are downloaded silently."""
    required = ["punkt", "wordnet", "omw-1.4"]
    for resource in required:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        except Exception:
            pass


ensure_nltk_resources()


class TextPreprocessor:
    """Preprocesses text for NLP matching and embeddings."""

    # Matches English words, Telugu Unicode range (\u0C00-\u0C7F), numbers, and percentage signs
    TOKEN_PATTERN = re.compile(r"[\w\u0C00-\u0C7F%]+", re.UNICODE)

    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def lowercase(self, text: str) -> str:
        """Lowercase string while preserving unicode normalization."""
        if not text or not isinstance(text, str):
            return ""
        # NFKC normalization for consistent unicode representation
        text = unicodedata.normalize("NFKC", text)
        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words/tokens while supporting Telugu script and units."""
        if not text or not isinstance(text, str):
            return []
        lowered = self.lowercase(text)
        tokens = self.TOKEN_PATTERN.findall(lowered)
        return tokens

    def lemmatize_token(self, token: str) -> str:
        """Lemmatize an individual token if lemmatizer is available and token is ASCII/English."""
        if not self.lemmatizer or not token:
            return token
        # If token has Indic/Telugu characters, return as-is
        if any(ord(char) > 127 for char in token):
            return token
        # Lemmatize as noun, then verb if unchanged
        lemma = self.lemmatizer.lemmatize(token, pos=wordnet.NOUN)
        if lemma == token:
            lemma = self.lemmatizer.lemmatize(token, pos=wordnet.VERB)
        return lemma

    def preprocess(self, text: str, return_tokens: bool = False) -> Union[str, List[str]]:
        """Preprocess text: lowercase -> tokenize -> lemmatize."""
        if not text or not isinstance(text, str):
            return [] if return_tokens else ""

        tokens = self.tokenize(text)
        lemmatized_tokens = [self.lemmatize_token(t) for t in tokens]

        if return_tokens:
            return lemmatized_tokens
        return " ".join(lemmatized_tokens)


# Global helper instance
_preprocessor = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """Convenience helper to preprocess text returning a normalized string."""
    return _preprocessor.preprocess(text, return_tokens=False)


def tokenize_and_lemmatize(text: str) -> List[str]:
    """Convenience helper returning list of lemmatized tokens."""
    return _preprocessor.preprocess(text, return_tokens=True)
