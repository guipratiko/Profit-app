from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from html.parser import HTMLParser
from threading import Lock
from urllib.parse import urljoin

import httpx

from app.config import INITIAL_ASSETS, OFFICIAL_WEBSITES


LOGO_CACHE_TTL = timedelta(hours=12)
LOGO_HTTP_TIMEOUT = 20.0
LOGO_MAX_BYTES = 1_500_000
DISCOVERY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ProfitAppAlpha/0.9)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
}
IMAGE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ProfitAppAlpha/0.9)",
    "Accept": "image/avif,image/webp,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
}


@dataclass(slots=True)
class CachedLogo:
    content: bytes
    media_type: str
    source_url: str
    expires_at: datetime


_logo_cache: dict[str, CachedLogo] = {}
_logo_cache_lock = Lock()


class LogoCandidateParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self._scored_candidates: dict[str, int] = {}

    def _add_candidate(self, url: str | None, score: int) -> None:
        if not url:
            return
        candidate = url.strip()
        if not candidate or candidate.startswith("data:"):
            return
        absolute = urljoin(self.base_url, candidate)
        if not absolute.startswith(("http://", "https://")):
            return
        current_score = self._scored_candidates.get(absolute)
        if current_score is None or score > current_score:
            self._scored_candidates[absolute] = score

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {
            key.lower(): value
            for key, value in attrs
            if key and value
        }
        tag_name = tag.lower()
        if tag_name == "img":
            src = attributes.get("src") or attributes.get("data-src") or attributes.get("data-lazy-src")
            descriptor = " ".join(
                [
                    attributes.get("alt", ""),
                    attributes.get("class", ""),
                    attributes.get("id", ""),
                    src or "",
                ]
            ).lower()
            if src and "logo" in descriptor:
                score = 110
                if src.lower().endswith(".svg"):
                    score += 8
                self._add_candidate(src, score)
            return

        if tag_name == "link":
            href = attributes.get("href")
            rel = " ".join(attributes.get("rel", "").lower().split())
            if not href or not rel:
                return
            if "apple-touch-icon" in rel:
                score = 92
            elif "shortcut icon" in rel or rel == "icon" or " icon" in rel:
                score = 86
            elif "mask-icon" in rel:
                score = 82
            else:
                return
            if "logo" in href.lower():
                score += 8
            if href.lower().endswith(".svg"):
                score += 6
            self._add_candidate(href, score)
            return

        if tag_name != "meta":
            return

        content = attributes.get("content")
        if not content:
            return
        meta_name = (attributes.get("property") or attributes.get("name") or attributes.get("itemprop") or "").lower()
        if meta_name in {"og:logo", "og:site_logo", "logo"}:
            score = 78
        elif meta_name in {"og:image", "twitter:image", "twitter:image:src", "itemprop:image"}:
            score = 34
        else:
            return
        if "logo" in content.lower():
            score += 12
        if content.lower().endswith(".svg"):
            score += 6
        self._add_candidate(content, score)

    def ordered_candidates(self) -> list[str]:
        ordered = sorted(self._scored_candidates.items(), key=lambda item: (-item[1], item[0]))
        return [url for url, _score in ordered]


def _placeholder_logo(ticker: str) -> CachedLogo:
    short_ticker = escape(ticker.split(".")[0][:5])
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <defs>
        <linearGradient id="brand" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#082f49"/>
          <stop offset="100%" stop-color="#0f766e"/>
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="16" fill="url(#brand)"/>
      <rect x="6" y="6" width="52" height="52" rx="12" fill="rgba(255,255,255,0.08)" stroke="rgba(255,255,255,0.18)"/>
      <text x="32" y="36" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#e2e8f0">{short_ticker}</text>
    </svg>
    """.strip()
    return CachedLogo(
        content=svg.encode("utf-8"),
        media_type="image/svg+xml",
        source_url="placeholder",
        expires_at=datetime.now(timezone.utc) + LOGO_CACHE_TTL,
    )


def _fallback_candidates(website: str) -> list[str]:
    return [
        urljoin(website, "/favicon.svg"),
        urljoin(website, "/favicon.ico"),
        urljoin(website, "/favicon.png"),
        urljoin(website, "/apple-touch-icon.png"),
        urljoin(website, "/apple-touch-icon-precomposed.png"),
        urljoin(website, "/favicon-32x32.png"),
    ]


def _discover_logo_candidates(client: httpx.Client, website: str) -> list[str]:
    try:
        response = client.get(website, headers=DISCOVERY_HEADERS)
        response.raise_for_status()
    except Exception:
        return _fallback_candidates(website)

    parser = LogoCandidateParser(str(response.url))
    parser.feed(response.text)
    ordered = parser.ordered_candidates()
    for candidate in _fallback_candidates(str(response.url)):
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _fetch_candidate_image(client: httpx.Client, website: str, candidate_url: str) -> tuple[bytes, str] | None:
    try:
        with client.stream(
            "GET",
            candidate_url,
            headers={**IMAGE_HEADERS, "Referer": website},
        ) as response:
            if response.status_code >= 400:
                return None
            media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if not media_type.startswith("image/"):
                return None
            declared_length = int(response.headers.get("content-length") or 0)
            if declared_length and declared_length > LOGO_MAX_BYTES:
                return None
            chunks: list[bytes] = []
            total_size = 0
            for chunk in response.iter_bytes():
                total_size += len(chunk)
                if total_size > LOGO_MAX_BYTES:
                    return None
                chunks.append(chunk)
    except Exception:
        return None

    if not chunks:
        return None
    return b"".join(chunks), media_type


def get_company_logo(ticker: str) -> CachedLogo:
    now = datetime.now(timezone.utc)
    with _logo_cache_lock:
        cached = _logo_cache.get(ticker)
        if cached and cached.expires_at > now:
            return cached

    website = OFFICIAL_WEBSITES.get(ticker)
    if not website:
        return _placeholder_logo(ticker)

    logo = _placeholder_logo(ticker)
    with httpx.Client(follow_redirects=True, timeout=LOGO_HTTP_TIMEOUT) as client:
        for candidate in _discover_logo_candidates(client, website):
            image = _fetch_candidate_image(client, website, candidate)
            if not image:
                continue
            content, media_type = image
            logo = CachedLogo(
                content=content,
                media_type=media_type,
                source_url=candidate,
                expires_at=now + LOGO_CACHE_TTL,
            )
            break

    with _logo_cache_lock:
        _logo_cache[ticker] = logo
    return logo


def asset_branding_record(ticker: str, logo_url: str) -> dict[str, str | None]:
    return {
        "ticker": ticker,
        "name": INITIAL_ASSETS[ticker],
        "website": OFFICIAL_WEBSITES.get(ticker),
        "logo_url": logo_url,
    }