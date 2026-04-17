"""Text preprocessing: tokenise, remove stopwords, stem."""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ensure NLTK data is available (download only once)
for _resource in ("stopwords", "punkt_tab"):
    try:
        nltk.data.find(
            f"corpora/{_resource}"
            if _resource == "stopwords"
            else f"tokenizers/{_resource}"
        )
    except LookupError:
        nltk.download(_resource, quiet=True)

_STEMMER = PorterStemmer()
_STOP_WORDS: set[str] = set(stopwords.words("english"))

# regex to split on non-alphanumeric characters
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out common English stopwords."""
    return [t for t in tokens if t not in _STOP_WORDS]


def stem(tokens: list[str]) -> list[str]:
    """Apply Porter stemming to each token."""
    return [_STEMMER.stem(t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """Full pipeline: tokenise → remove stopwords → stem."""
    return stem(remove_stopwords(tokenize(text)))


if __name__ == "__main__":
    sample = "Geological formations include sedimentary rocks and metamorphic minerals."
    tokens = preprocess(sample)
    print(f"Input:  {sample}")
    print(f"Tokens: {tokens}")
