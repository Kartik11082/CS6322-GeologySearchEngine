from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .config import ServiceConfig
from .utils import batched


@dataclass(slots=True)
class ProjectionArtifacts:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    normalizer: Normalizer

    def transform(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for chunk in batched(texts, batch_size):
            matrix = self.vectorizer.transform(chunk)
            dense = self.svd.transform(matrix)
            normalized = self.normalizer.transform(dense).astype(np.float32)
            chunks.append(normalized)
        if not chunks:
            width = getattr(self.svd, "n_components", 0)
            return np.zeros((0, width), dtype=np.float32)
        return np.vstack(chunks)

    def transform_query(self, query: str) -> np.ndarray:
        return self.transform([query], batch_size=1)[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "ProjectionArtifacts":
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, ProjectionArtifacts):
            raise TypeError(f"Unexpected projection artifact type: {type(obj)!r}")
        return obj


def fit_projection(sample_texts: Sequence[str], cfg: ServiceConfig) -> ProjectionArtifacts:
    vectorizer = TfidfVectorizer(
        max_features=cfg.tfidf_max_features,
        min_df=cfg.tfidf_min_df,
        max_df=cfg.tfidf_max_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform(sample_texts)
    n_components = min(cfg.svd_components, max(2, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=cfg.random_seed)
    dense = svd.fit_transform(tfidf)
    normalizer = Normalizer(copy=False)
    normalizer.fit(dense)
    return ProjectionArtifacts(vectorizer=vectorizer, svd=svd, normalizer=normalizer)
