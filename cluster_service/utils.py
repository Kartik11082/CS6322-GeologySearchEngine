from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""

    split = urlsplit(value)
    scheme = (split.scheme or "https").lower()
    netloc = split.netloc.lower()
    path = split.path.rstrip("/")

    kept_query = []
    for key, val in parse_qsl(split.query, keep_blank_values=True):
        lk = key.lower()
        if lk.startswith("utm_"):
            continue
        if lk in {"replytocom", "currency", "return_to"}:
            continue
        kept_query.append((key, val))

    query = urlencode(kept_query, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def batched(items: Sequence[Any] | Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def nested_get(payload: Any, path: Sequence[str]) -> Any:
    current = payload
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
