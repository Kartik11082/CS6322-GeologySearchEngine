"""Pure stdlib cosine for scalar clustering document-frequency vectors (no NLTK)."""

from __future__ import annotations

import math
from typing import Mapping


def association_cosine_doc_frequency(
    u_postings: Mapping[str, float | int],
    v_postings: Mapping[str, float | int],
    local_doc_ids: set[str],
) -> float:
    """
    Cosine similarity between vectors s_u[d] = f(u,d), d ∈ D_l.

    Same numerator Σ_d f(u,d)f(v,d) as association C(u,v) in lecture notes / report.
    """
    dot = 0.0
    nu_sq = 0.0
    nv_sq = 0.0
    for d in local_doc_ids:
        fu = float(u_postings.get(d, 0) or 0)
        fv = float(v_postings.get(d, 0) or 0)
        dot += fu * fv
        nu_sq += fu * fu
        nv_sq += fv * fv
    if dot <= 0:
        return 0.0
    if nu_sq <= 0 or nv_sq <= 0:
        return 0.0
    den = math.sqrt(nu_sq * nv_sq)
    return dot / den if den > 0 else 0.0
