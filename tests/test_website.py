from typing import Any

import httpx
import pytest

from src.tools.website import (
    StubWebsiteFetcher,
    WebsiteFetcher,
    extract_to_evidence,
    normalise_website,
    parse_html,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("acme.com", "https://acme.com"),
        ("www.acme.com/about?x=1", "https://www.acme.com"),
        ("http://acme.io", "http://acme.io"),
        ("HTTPS://Acme.com/", "https://Acme.com"),
    ],
)
def test_normalise_website(raw: str, expected: str) -> None:
    assert normalise_website(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "https://"])
def test_normalise_website_rejects_garbage(raw: str) -> None:
    with pytest.raises(ValueError):
        normalise_website(raw)


def test_parse_homepage_extracts_title_description_headings_and_text(html_fixture: Any) -> None:
    page = parse_html(html_fixture("homepage.html"), "https://acme.com")
    assert page.title.startswith("Acme Robotics")
    assert page.description.startswith("Acme Robotics builds autonomous picking robots")
    assert page.headings == ["Robots that pick, pack and ship", "Trusted by 120 logistics teams"]
    assert "$25M Series A" in page.text
    assert "dataLayer" not in page.text and "color: red" not in page.text
    assert "About" not in page.text  # nav stripped


def test_parse_careers_counts_job_like_items(html_fixture: Any) -> None:
    page = parse_html(html_fixture("careers.html"), "https://acme.com/careers")
    assert page.job_count >= 3
    ev = extract_to_evidence(page, "/careers")
    assert ev["source"] == "website" and "Open roles detected:" in ev["content"]


def test_parse_html_truncates_text() -> None:
    page = parse_html("<html><body>" + "word " * 2000 + "</body></html>", "u", max_chars=100)
    assert len(page.text) == 100


async def test_fetcher_collects_pages_and_skips_failures(html_fixture: Any) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/":
            return httpx.Response(
                200, text=html_fixture("homepage.html"), headers={"content-type": "text/html"}
            )
        if request.url.path == "/about":
            raise httpx.ConnectError("refused")
        return httpx.Response(404, text="nope")

    fetcher = WebsiteFetcher(transport=httpx.MockTransport(handler), timeout_seconds=1)
    evidence = await fetcher.fetch("acme.com")
    assert len(evidence) == 1
    assert evidence[0]["url"] == "https://acme.com"
    assert evidence[0]["title"].startswith("Acme Robotics")
    assert calls.count("/about") == 2  # retried once, then skipped


async def test_fetcher_skips_non_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"a": 1})

    fetcher = WebsiteFetcher(transport=httpx.MockTransport(handler), paths=("",))
    assert await fetcher.fetch("acme.com") == []


async def test_stub_fetcher_returns_three_pages() -> None:
    evidence = await StubWebsiteFetcher().fetch("acme.com")
    assert [e["url"] for e in evidence] == [
        "https://acme.com",
        "https://acme.com/about",
        "https://acme.com/careers",
    ]
