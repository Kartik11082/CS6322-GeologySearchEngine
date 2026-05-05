"""Text preprocessing: tokenise, remove stopwords, stem."""

import json
import re
import time
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    import Stemmer  # type: ignore
except ModuleNotFoundError:
    Stemmer = None  # type: ignore

def _tokenizer_debug_fs() -> dict:
    row: dict = {"nltk_data_paths_used": []}
    for root in nltk.data.path[:3]:
        p = Path(root) / "tokenizers"
        row["nltk_data_paths_used"].append(str(root))
        if not p.is_dir():
            continue
        for name in ("punkt", "punkt_tab"):
            sub = p / name
            if not sub.exists():
                continue
            kids = sorted(x.name for x in sub.iterdir()) if sub.is_dir() else []
            row[f"{name}_children_head"] = kids[:40]
            if name == "punkt":
                row["punkt_PY3_tab_exists"] = (sub / "PY3_tab").exists()
                row["punkt_PICKLE_exists"] = (sub / "PY3").exists()
    return row


# ensure NLTK data is available (download only once)
for _resource in ("stopwords", "punkt_tab"):
    _path_key = (
        f"corpora/{_resource}"
        if _resource == "stopwords"
        else f"tokenizers/{_resource}"
    )
    try:
        nltk.data.find(_path_key)

    except LookupError:

        nltk.download(_resource, quiet=True)
    except OSError as exc:

        nltk.download(_resource, quiet=True)


_PY_STEMMER = Stemmer.Stemmer("english") if Stemmer is not None else None
_NLTK_STEMMER = PorterStemmer()
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
    """Apply Porter stemming to each token.

    Prefer PyStemmer when installed for speed; fall back to NLTK otherwise.
    """
    if _PY_STEMMER is not None:
        return _PY_STEMMER.stemWords(tokens)
    return [_NLTK_STEMMER.stem(token) for token in tokens]


def preprocess(text: str) -> list[str]:
    """Full pipeline: tokenise → remove stopwords → stem."""
    return stem(remove_stopwords(tokenize(text)))


if __name__ == "__main__":
    sample = "Geological formations include sedimentary rocks and metamorphic minerals."
    tokens = preprocess(sample)
    print(f"Input:  {sample}")
    print(f"Tokens: {tokens}")
