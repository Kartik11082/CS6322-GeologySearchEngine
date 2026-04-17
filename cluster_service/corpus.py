from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

try:
    from langdetect import DetectorFactory, LangDetectException, detect
except ImportError:  # pragma: no cover - exercised in this environment
    DetectorFactory = None
    LangDetectException = Exception
    detect = None

from .config import ServiceConfig
from .utils import normalize_url

if DetectorFactory is not None:
    DetectorFactory.seed = 0

_SPACE_RE = re.compile(r"\s+")
_ASCII_RE = re.compile(r"[A-Za-z]")
_COMMON_ENGLISH = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "geology",
    "earth",
    "rock",
    "volcano",
}


@dataclass(slots=True)
class CorpusRecord:
    normalized_url: str
    url: str
    title: str
    text: str
    domain: str
    content_type: str
    depth: int
    status: int

    @property
    def clustering_text(self) -> str:
        title = self.title.strip()
        if title:
            return f"{title} {title} {title} {self.text}"
        return self.text


@dataclass(slots=True)
class CorpusBundle:
    records: list[CorpusRecord]
    stats: dict
    graph_stats: dict


def _clean_text(text: str, max_chars: int) -> str:
    compact = _SPACE_RE.sub(" ", (text or "").strip())
    return compact[:max_chars].strip()


def _looks_low_value(
    normalized_url: str, title: str, text: str, cfg: ServiceConfig
) -> bool:
    low_url = normalized_url.lower()
    low_title = (title or "").lower()
    low_text = (text or "").lower()

    if any(token in low_url for token in cfg.blocked_url_substrings):
        return True
    if any(token.lower() in low_title for token in cfg.blocked_title_substrings):
        return True
    if any(token.lower() in low_text for token in cfg.blocked_text_substrings):
        return True
    return False


def _is_english(text: str) -> bool:
    sample = text[:1000].strip()
    if not sample:
        return False
    if detect is None:
        letters = _ASCII_RE.findall(sample)
        if len(letters) < max(20, len(sample) // 10):
            return False
        lower = sample.lower()
        matches = sum(1 for token in _COMMON_ENGLISH if token in lower)
        return matches >= 2
    try:
        return detect(sample) == "en"
    except LangDetectException:
        return False


def _graph_stats(graph_path: Path) -> dict:
    edge_count = 0
    nodes: set[str] = set()
    with open(graph_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            src = normalize_url(row.get("src_url", ""))
            dst = normalize_url(row.get("dst_url", ""))
            if src:
                nodes.add(src)
            if dst:
                nodes.add(dst)
            edge_count += 1
    return {"edges": edge_count, "graph_nodes": len(nodes)}


def load_corpus(
    pages_path: Path, graph_path: Path, cfg: ServiceConfig | None = None
) -> CorpusBundle:
    cfg = cfg or ServiceConfig()
    stats = Counter()
    domains = Counter()
    seen_urls: set[str] = set()
    seen_text_hashes: set[str] = set()
    records: list[CorpusRecord] = []

    with open(pages_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["lines_read"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue

            status = int(obj.get("status") or 0)
            if status != 200:
                stats["filtered_non_200"] += 1
                continue

            content_type = (obj.get("content_type") or "").split(";")[0].strip().lower()
            if content_type and content_type != "text/html":
                stats["filtered_non_html"] += 1
                continue

            raw_url = obj.get("url", "")
            normalized_url = normalize_url(raw_url)
            if not normalized_url:
                stats["filtered_missing_url"] += 1
                continue
            if normalized_url in seen_urls:
                stats["filtered_duplicate_url"] += 1
                continue

            title = _clean_text(obj.get("title", ""), 512)
            text = _clean_text(obj.get("text", ""), cfg.max_text_chars)
            if len(text) < cfg.min_chars:
                stats["filtered_short_text"] += 1
                continue
            if _looks_low_value(normalized_url, title, text, cfg):
                stats["filtered_low_value"] += 1
                continue
            if not _is_english(f"{title} {text}"):
                stats["filtered_non_english"] += 1
                continue

            text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if text_hash in seen_text_hashes:
                stats["filtered_duplicate_text"] += 1
                continue

            seen_urls.add(normalized_url)
            seen_text_hashes.add(text_hash)
            domain = urlsplit(normalized_url).netloc.lower()
            domains[domain] += 1
            records.append(
                CorpusRecord(
                    normalized_url=normalized_url,
                    url=raw_url,
                    title=title,
                    text=text,
                    domain=domain,
                    content_type=content_type or "text/html",
                    depth=int(obj.get("depth") or 0),
                    status=status,
                )
            )
            stats["kept"] += 1

    graph_stats = _graph_stats(graph_path)
    summary = {
        "records_kept": stats["kept"],
        "records_seen": stats["lines_read"],
        "unique_domains": len(domains),
        "top_domains": domains.most_common(15),
        "filters": {k: v for k, v in stats.items() if k != "kept"},
    }
    summary.update(graph_stats)
    return CorpusBundle(records=records, stats=summary, graph_stats=graph_stats)


def select_balanced_sample(records: list[CorpusRecord], cfg: ServiceConfig) -> list[int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.domain].append(index)

    rng = random.Random(cfg.random_seed)
    for indices in grouped.values():
        rng.shuffle(indices)

    capped = {
        domain: indices[: cfg.domain_cap_per_sample]
        for domain, indices in grouped.items()
        if indices
    }

    ordered_domains = list(capped.keys())
    rng.shuffle(ordered_domains)
    selected: list[int] = []
    offsets = {domain: 0 for domain in ordered_domains}

    while len(selected) < min(cfg.training_sample_cap, len(records)):
        progressed = False
        for domain in ordered_domains:
            indices = capped[domain]
            offset = offsets[domain]
            if offset >= len(indices):
                continue
            selected.append(indices[offset])
            offsets[domain] += 1
            progressed = True
            if len(selected) >= cfg.training_sample_cap:
                break
        if not progressed:
            break

    return selected


def benchmark_queries(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def judged_query_ids(queries: Iterable[dict]) -> set[str]:
    return {
        str(item["query_id"])
        for item in queries
        if bool(item.get("judged_subset"))
    }
