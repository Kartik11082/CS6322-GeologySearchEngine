import posixpath
import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from lxml import html as lxml_html

from crawler.config import Config

_BINARY_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".tar",
    ".gz", ".mp3", ".mp4", ".avi", ".css", ".js", ".json", ".xml",
)


class Parser:
    """HTML parser and quality filter for crawler_v2."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def _normalize_url(self, url: str) -> str:
        try:
            parsed = urlsplit(url)
        except ValueError:
            return ""

        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            return ""

        try:
            host = (parsed.hostname or "").lower()
            port = parsed.port
        except ValueError:
            return ""
        if not host:
            return ""
        use_port = bool(
            port
            and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443))
        )
        netloc = f"{host}:{port}" if use_port else host

        path = parsed.path or "/"
        path = re.sub(r"/{2,}", "/", path)
        path = posixpath.normpath(path)
        if not path.startswith("/"):
            path = "/" + path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        query_pairs.sort(key=lambda kv: (kv[0], kv[1]))
        query = urlencode(query_pairs, doseq=True)

        return urlunsplit((scheme, netloc, path, query, ""))

    def _is_link_allowed(self, normalized_url: str) -> bool:
        lowered = normalized_url.lower()
        try:
            parsed = urlsplit(lowered)
        except ValueError:
            return False

        if parsed.scheme not in {"http", "https"}:
            return False

        path = parsed.path or ""
        if path.endswith(_BINARY_EXTENSIONS):
            return False

        if any(pattern in lowered for pattern in self.config.JUNK_URL_PATTERNS):
            return False

        if not any(keyword in lowered for keyword in self.config.FOCUS_KEYWORDS):
            return False

        return True

    def parse(self, html: str, base_url: str) -> tuple[list[str], str, str]:
        tree = lxml_html.fromstring(html or "")

        title_nodes = tree.xpath("//title/text()")
        title = title_nodes[0].strip() if title_nodes else ""

        links: list[str] = []
        seen_links: set[str] = set()
        for href in tree.xpath("//a/@href"):
            absolute = urljoin(base_url, href)
            normalized = self._normalize_url(absolute)
            if not normalized:
                continue
            if not self._is_link_allowed(normalized):
                continue
            if normalized in seen_links:
                continue
            seen_links.add(normalized)
            links.append(normalized)
            if len(links) >= 80:
                break

        removable = tree.xpath(
            "//script|//style|//noscript|//header|//footer|//nav|//aside|//form|//iframe"
        )
        removable += tree.xpath(
            "//*[contains(concat(' ', normalize-space(@class), ' '), ' sidebar ') "
            "or contains(concat(' ', normalize-space(@class), ' '), ' menu ') "
            "or contains(concat(' ', normalize-space(@class), ' '), ' advertisement ')]"
        )
        for el in removable:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

        text = tree.text_content()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 50_000:
            text = text[:50_000]

        lowered_text = text.lower()
        if not any(keyword in lowered_text for keyword in self.config.CONTENT_KEYWORDS):
            return ([], "", "")

        return (links, text, title)
