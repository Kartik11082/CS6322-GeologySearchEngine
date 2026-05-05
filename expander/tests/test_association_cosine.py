"""Unit tests for scalar clustering cosine over local document-frequency vectors."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "expander"))

from scalar_association_math import association_cosine_doc_frequency  # noqa: E402


class TestAssociationCosineDocFrequency(unittest.TestCase):
    def test_perpendicular_vectors(self):
        dl = {"d1", "d2"}
        u = {"d1": 1}
        v = {"d2": 1}
        self.assertAlmostEqual(
            association_cosine_doc_frequency(u, v, dl), 0.0, places=9
        )

    def test_identical_direction(self):
        dl = {"d1"}
        u = {"d1": 3.0}
        v = {"d1": 4.0}
        self.assertAlmostEqual(
            association_cosine_doc_frequency(u, v, dl), 1.0, places=9
        )

    def test_known_angle(self):
        dl = {"d1", "d2"}
        u = {"d1": 1.0, "d2": 0.0}
        v = {"d1": 0.0, "d2": 1.0}
        self.assertAlmostEqual(
            association_cosine_doc_frequency(u, v, dl), 0.0, places=9
        )
        # (1,1) vs (2,2) normalized
        a = {"d1": 1.0, "d2": 1.0}
        b = {"d1": 2.0, "d2": 2.0}
        self.assertAlmostEqual(
            association_cosine_doc_frequency(a, b, dl), 1.0, places=9
        )


if __name__ == "__main__":
    unittest.main()
