"""Company website fetcher: ``httpx`` for transport, ``selectolax`` for parsing.

Fetches the homepage, ``/about`` and ``/careers`` and turns each page into an
:class:`~src.api.schemas.Evidence` item carrying the title, meta description,
headings and a bounded amount of visible text. The careers page additionally
yields a hiring-signal count so the Profile Builder can emit a ``hiring``
signal without an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import httpx
from loguru import logger
from selectolax.parser import HTMLParser
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.api.schemas import Evidence

PAGE_PATHS: tuple[str, ...] = ("", "/about", "/careers")

_NOISE_TAGS = ("script", "style", "noscript", "svg", "nav", "footer", "iframe")
_ROLE_WORDS = re.compile(
    r"\b(engineer|developer|manager|designer|scientist|analyst|lead|head of|director|"
    r"account executive|sales|marketing|recruiter|founder|intern)\b",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")


@dataclass
class PageExtract:
    """Structured extraction of one HTML page.

    Attributes:
        url: Fetched URL.
        title: ``<title>`` text.
        description: ``<meta name="description">`` content.
        headings: ``h1``/``h2`` texts in document order.
        text: Visible text, whitespace-normalised and truncated.
        job_count: Number of list items/links that look like job openings.
    """

    url: str
    title: str = ""
    description: str = ""
    headings: list[str] = field(default_factory=list)
    text: str = ""
    job_count: int = 0


def normalise_website(website: str) -> str:
    """Turn a bare domain or URL into an ``https://host`` base URL.

    Args:
        website: ``example.com``, ``www.example.com/x`` or ``http://example.com``.

    Returns:
        Scheme + host without path or trailing slash.

    Raises:
        ValueError: If no host can be extracted.
    """
    candidate = website.strip()
    if not candidate:
        raise ValueError("empty website")
    if "://" not in candidate:
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    if not parsed.netloc:
        raise ValueError(f"cannot parse website {website!r}")
    scheme = parsed.scheme if parsed.scheme in ("http", "https") else "https"
    return f"{scheme}://{parsed.netloc}"


def parse_html(html: str, url: str, *, max_chars: int = 1500) -> PageExtract:
    """Extract title, description, headings, visible text and job count.

    Args:
        html: Raw HTML document.
        url: URL the document came from (stored on the result).
        max_chars: Cap on the length of ``text``.

    Returns:
        The page extraction.
    """
    tree = HTMLParser(html)
    title_node = tree.css_first("title")
    title = _WS.sub(" ", title_node.text(strip=True)) if title_node else ""

    description = ""
    for meta in tree.css("meta"):
        name = (meta.attributes.get("name") or meta.attributes.get("property") or "").lower()
        if name in ("description", "og:description"):
            description = (meta.attributes.get("content") or "").strip()
            if description:
                break

    headings = [_WS.sub(" ", h.text(strip=True)) for h in tree.css("h1, h2") if h.text(strip=True)][
        :12
    ]

    job_count = 0
    for node in tree.css("li, a, h3"):
        txt = node.text(strip=True)
        if 4 <= len(txt) <= 90 and _ROLE_WORDS.search(txt):
            job_count += 1

    for tag in _NOISE_TAGS:
        for node in tree.css(tag):
            node.decompose()
    body = tree.body if tree.body is not None else tree.root
    raw_text = body.text(separator=" ", strip=True) if body is not None else ""
    text = _WS.sub(" ", raw_text)[:max_chars]

    return PageExtract(
        url=url,
        title=title,
        description=description,
        headings=headings,
        text=text,
        job_count=job_count,
    )


def extract_to_evidence(page: PageExtract, path: str) -> Evidence:
    """Convert a :class:`PageExtract` into the common evidence shape.

    Args:
        page: Parsed page.
        path: Path that was fetched (``""``, ``/about`` or ``/careers``).

    Returns:
        Evidence with ``source="website"``.
    """
    label = {"": "homepage", "/about": "about page", "/careers": "careers page"}.get(path, path)
    parts: list[str] = []
    if page.description:
        parts.append(page.description)
    if page.headings:
        parts.append("Headings: " + " | ".join(page.headings[:6]))
    if path == "/careers":
        parts.append(f"Open roles detected: {page.job_count}")
    if page.text:
        parts.append(page.text[:600])
    return Evidence(
        title=page.title or f"{label} of {urlparse(page.url).netloc}",
        url=page.url,
        content=" ".join(parts),
        score=0.75 if path == "" else 0.7,
        source="website",
    )


class WebsiteFetch(Protocol):
    """Provider-agnostic website fetcher."""

    name: str

    async def fetch(self, website: str) -> list[Evidence]:
        """Fetch the standard pages of ``website`` and return evidence items."""
        ...


class WebsiteFetcher:
    """Fetch homepage, ``/about`` and ``/careers`` with explicit timeouts.

    Args:
        timeout_seconds: HTTP timeout per page.
        max_chars: Cap on the text stored per page.
        transport: Optional ``httpx`` transport (tests inject a mock).
        paths: Paths to fetch relative to the site root.
    """

    name = "httpx"

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        max_chars: int = 1500,
        transport: httpx.AsyncBaseTransport | None = None,
        paths: tuple[str, ...] = PAGE_PATHS,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars
        self._transport = transport
        self.paths = paths

    async def fetch(self, website: str) -> list[Evidence]:
        """Fetch the configured pages.

        Pages returning non-2xx or non-HTML responses are skipped; transport
        errors are retried once and then skipped with a warning, so a dead
        careers page never aborts the research run.

        Args:
            website: Domain or URL of the company.

        Returns:
            One evidence item per successfully fetched page.
        """
        base = normalise_website(website)
        evidence: list[Evidence] = []
        headers = {"User-Agent": "sales-intelligence-agent/1.0 (+research bot)"}
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers=headers,
            transport=self._transport,
        ) as client:
            for path in self.paths:
                url = base + path
                try:
                    html = await self._get(client, url)
                except httpx.TransportError as exc:
                    logger.warning(f"website fetch failed url={url} error={exc!r}")
                    continue
                if html is None:
                    continue
                page = parse_html(html, url, max_chars=self.max_chars)
                evidence.append(extract_to_evidence(page, path))
        logger.debug(f"website base={base} pages={len(evidence)}")
        return evidence

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def _get(self, client: httpx.AsyncClient, url: str) -> str | None:
        """GET one page and return its HTML, or ``None`` if not usable.

        Args:
            client: Shared HTTP client.
            url: Absolute URL.

        Returns:
            The HTML body, or ``None`` on non-2xx / non-HTML responses.
        """
        response = await client.get(url)
        if response.status_code >= 400:
            logger.debug(f"website skip url={url} status={response.status_code}")
            return None
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and not response.text.lstrip().startswith("<"):
            logger.debug(f"website skip url={url} content_type={content_type!r}")
            return None
        return response.text


class StubWebsiteFetcher:
    """Deterministic website evidence for offline runs."""

    name = "stub"

    async def fetch(self, website: str) -> list[Evidence]:
        """Return canned homepage/about/careers evidence for ``website``.

        Args:
            website: Domain or URL of the company.

        Returns:
            Three evidence items with ``source="website"``.
        """
        base = normalise_website(website)
        host = urlparse(base).netloc
        return [
            Evidence(
                title=f"{host} — homepage",
                url=base,
                content=(
                    "Modern B2B software platform. Headings: Trusted by teams worldwide | "
                    "Automate your workflow"
                ),
                score=0.75,
                source="website",
            ),
            Evidence(
                title=f"About {host}",
                url=base + "/about",
                content="Founded by a team of engineers. Headquartered in San Francisco, CA.",
                score=0.7,
                source="website",
            ),
            Evidence(
                title=f"Careers at {host}",
                url=base + "/careers",
                content="Open roles detected: 4 Senior Software Engineer, Account Executive",
                score=0.7,
                source="website",
            ),
        ]
