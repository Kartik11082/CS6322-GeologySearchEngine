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

# #region agent log
def _agent_dbg(payload: dict) -> None:
    payload.setdefault("sessionId", "012961")
    payload.setdefault("timestamp", int(time.time() * 1000))
    payload.setdefault("runId", "pre-import")
    _p = Path(
        "/Users/uddeshsingh/Documents/Spring26/IR/Project/CS6322-GeologySearchEngine/.cursor/debug-012961.log"
    )
    with _p.open("a", encoding="utf-8") as _f:
        _f.write(json.dumps(payload, default=str) + "\n")


# #endregion


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


# Hypotheses: H1=find raises OSError not caught→no download | H2=PY3_tab missing partial tree
# H3=download skipped | H4=stale/conflicting tokenizer dirs | H5=nltk.path resolution
_agent_dbg(
    {
        "location": "preprocessor.py:pre-loop",
        "message": "nltk bootstrap paths",
        "hypothesisId": "H4,H5",
        "data": {
            "version": getattr(nltk, "__version__", ""),
            "data_path_order": list(nltk.data.path),
            "tokenizer_probe": _tokenizer_debug_fs(),
        },
    }
)

# ensure NLTK data is available (download only once)
for _resource in ("stopwords", "punkt_tab"):
    _path_key = (
        f"corpora/{_resource}"
        if _resource == "stopwords"
        else f"tokenizers/{_resource}"
    )
    _agent_dbg(
        {
            "location": "preprocessor.py:loop",
            "message": "before find",
            "hypothesisId": "H2,H4",
            "data": {"resource": _resource, "path_key": _path_key},
        }
    )
    try:
        nltk.data.find(_path_key)
        _agent_dbg(
            {
                "location": "preprocessor.py:loop",
                "message": "find succeeded",
                "hypothesisId": "H3",
                "data": {"resource": _resource},
            }
        )
    except LookupError:
        _agent_dbg(
            {
                "location": "preprocessor.py:loop",
                "message": "LookupError triggering download",
                "hypothesisId": "H3",
                "data": {"resource": _resource},
            }
        )
        nltk.download(_resource, quiet=True)
    except OSError as exc:
        _agent_dbg(
            {
                "location": "preprocessor.py:loop",
                "message": "find raised OSError; download fallback",
                "hypothesisId": "H1,H3",
                "data": {
                    "resource": _resource,
                    "path_key": _path_key,
                    "exc_type": type(exc).__name__,
                    "exc_repr": repr(exc),
                    **(_tokenizer_debug_fs()),
                },
            }
        )
        nltk.download(_resource, quiet=True)
        _agent_dbg(
            {
                "location": "preprocessor.py:loop",
                "message": "post-OSError download attempted",
                "hypothesisId": "H1",
                "data": {"resource": _resource},
            }
        )

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
