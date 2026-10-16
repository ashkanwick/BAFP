"""
Persian comment collection for the BAFP sentiment-analysis capstone.
====================================================================

THREE collection paths live in this module, and the distinction between them
matters more than the code itself:

1. :class:`Varzesh3NewsClient`  -- PERSIAN NEWS COMMENTS (what the brief asked
   for). Varzesh3 exposes the JSON API its own site calls, verified from this
   network (HTTP 200):

       GET https://web-api.varzesh3.com/v1.0/news/latest
       GET https://web-api.varzesh3.com/v1.0/news/<news_id>/comments?skip=<n>&sort=<s>

   ``latest`` returns news stubs (``id``, ``title``, ``link``); the comments
   endpoint returns::

       {"items": [ {...}, ... ], "hasMore": true,
        "_links": [{"href": "...?skip=20&sort=MostPopular&l=1790383868", "rel": "next"}]}

   and each item carries (keys verified against news 2431742)::

       id, message, createdOn, createdOnUtc, replyCount, writer{name,image},
       vote{likeCount,dislikeCount}, currentUser, replies, isPending, _links

   FIELD NOTES THAT AFFECT MODELLING
     * the comment text is ``message`` (NOT ``body`` -- Digikala uses ``body``)
     * there is NO star rating. News comments are UNLABELLED. ``vote.likeCount``
       / ``vote.dislikeCount`` are the only signal, and they measure agreement
       with other readers, not writer sentiment.
     * ``createdOnUtc`` is a real ISO-8601 UTC timestamp (unlike Digikala's
       Jalali display string).
     * pagination is ``skip`` plus an opaque ``l`` cursor echoed from
       ``_links[rel=next]``; ``hasMore`` is authoritative, page counts are not
       returned.
     * ``isPending`` is True for comments awaiting moderation -- filtered out.

   So: news comments are used for the corpus, EDA, word clouds and the
   dashboard's live-prediction inputs. They are NOT the training labels.

2. :class:`DigikalaAPIClient` -- the LABELED corpus, and therefore the training
   set. Digikala's public JSON API hands back star ratings, which we bin into
   three classes (see :mod:`src.preprocess`). Verified endpoints (HTTP 200)::

       GET https://api.digikala.com/v1/search/?q=<query>&page=<n>
       GET https://api.digikala.com/v1/product/<product_id>/comments/?page=<n>&sort=<s>

   Comments payload (verified against product 18575915, 1,984 rated comments)::

       data.pager    = {"current_page": 1, "total_pages": 100, "total_items": 1984}
       data.comments = [ {comment}, ... ]   # 20 per page

   each comment carrying ``id, title, body, created_at, rate, reactions, files,
   is_buyer, user_name, is_anonymous, purchased_item, advantages,
   disadvantages, user_reactions, relative_date, social_profile,
   contribution_badge``. Notes:
     * text is ``body`` (``title`` is usually null); rating is ``rate``, and it is
       NOT reliably an int -- a 4,004-comment scrape returned 0/1/2/3/4/5 plus
       fractional values (0.5, 1.2, 2.55, 3.65). ``0`` means "unrated"; rounding
       before binning is handled in src/preprocess.py
     * likes/dislikes are NESTED (``reactions.likes`` / ``reactions.dislikes``)
     * ``created_at`` is a JALALI display string such as "3 مهر 1405", parsed
       with jdatetime
     * buyer flag is the FLAT boolean ``is_buyer``; there is no
       ``recommendation_status`` object
     * ``data.product`` is ABSENT, so titles come from the search endpoint

3. :func:`scrape_with_selenium` -- FALLBACK for any site with no JSON API. It
   drives Microsoft Edge through Selenium and parses rendered HTML with
   BeautifulSoup. Every CSS selector it uses lives in :data:`SELECTORS` so that
   retargeting another site is a config edit, not a code edit.

   CAUTION -- status of this path: this machine has Edge 153 and NO Chrome, and
   msedgedriver must be fetched by Selenium Manager on first run. That fetch has
   NOT been verified here, so treat this path as untested. Both JSON APIs above
   work without it; prefer them.

Usage
-----
Smoke-test every path before committing to a long run::

    python -m src.scraper --smoke

Collect Persian news comments (unlabelled corpus)::

    python -m src.scraper --source news --target 4000 --out data/raw_news_comments.csv

Collect rated Digikala comments (labelled training data)::

    python -m src.scraper --source digikala --target 5000 --out data/raw_digikala_comments.csv
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# --------------------------------------------------------------------------
# HTTP configuration
# --------------------------------------------------------------------------
# A browser-like UA is required; both APIs 403 or throttle the default UA.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 25  # seconds
DEFAULT_DELAY = 0.4  # polite pause between requests (seconds)
MAX_RETRIES = 4

# --- Varzesh3 (news) -------------------------------------------------------
V3_API_BASE = "https://web-api.varzesh3.com/v1.0"
V3_LATEST_ENDPOINT = f"{V3_API_BASE}/news/latest"
V3_COMMENTS_ENDPOINT = f"{V3_API_BASE}/news/{{news_id}}/comments"
# ``sort`` values seen accepted by the API. MostPopular returns the most-liked
# comments first (better text quality, blander sentiment); Newest is the wider net.
V3_SORTS: tuple[str, ...] = ("MostPopular", "Newest")
V3_PAGE_SIZE = 20  # server-side page size; informational only

# --- Digikala (rated) ------------------------------------------------------
DIGIKALA_API_BASE = "https://api.digikala.com/v1"
SEARCH_ENDPOINT = f"{DIGIKALA_API_BASE}/search/"
COMMENTS_ENDPOINT = f"{DIGIKALA_API_BASE}/product/{{product_id}}/comments/"
COMMENTS_PER_PAGE = 20  # server-side page size; informational only

# Valid values for Digikala's ``sort`` query parameter, taken from the API's own
# ``data.sort_options``. These are STRING ids, not integers -- ``?sort=1``
# returns HTTP 400. They matter for label quality, because neutral (rate=3) is
# the rarest class. Sampling product 18575915 page 1 showed:
#   newest -> {0:1, 2:1, 3:3, 4:1, 5:14}   broadest mix, surfaces low ratings
#   buyers -> {3:1, 4:6, 5:13}             rated only, buyer-skewed
#   likes  -> {0:3, 3:1, 4:5, 5:11}        "most useful", older/popular
# Walking several sorts and de-duplicating widens the rating spread.
DIGIKALA_SORTS: tuple[str, ...] = ("newest", "buyers", "likes")

# Broad, popular categories: frequently-reviewed products give enough labelled
# comments per product to matter.
DEFAULT_QUERIES: tuple[str, ...] = (
    "موبایل",
    "هدفون",
    "لپ تاپ",
    "ساعت هوشمند",
    "کتاب",
    "کفش ورزشی",
    "یخچال",
    "ماشین لباسشویی",
)

# Search terms used to find news articles for the (unlabelled) news corpus. The
# API has no full-text search, so these are only used to filter titles from
# ``news/latest``; drop them and you get a chronologically ordered corpus.
DEFAULT_NEWS_TOPICS: tuple[str, ...] = ("ایران", "تیم ملی", "فوتبال", "اقتصاد")

# --------------------------------------------------------------------------
# Selectors for the Selenium/BeautifulSoup FALLBACK path only.
#
# Written against a generic Persian news/review page and meant to be edited:
# each key is the CSS selector for one logical field, so no selector literal
# appears in the parsing logic. To scrape a different site, change these values
# (and the URL you pass in) and nothing else.
# --------------------------------------------------------------------------
SELECTORS: dict[str, str] = {
    "comment_container": (
        ".comment-item, .review-item, article.comment, .box_news_comment, "
        ".comments-list > li, [id^='comment-']"
    ),
    "text": ".comment-body, .review-text, .comment-text, p.text, .x-comment-text",
    "author": ".comment-author, .reviewer-name, .user-name, .writer",
    "rating": ".comment-rating, .star-rating, [itemprop='ratingValue']",
    "date": ".comment-date, time, .date",
    "likes": ".comment-likes, .like-count, .fa-thumbs-up",
    "dislikes": ".comment-dislikes, .dislike-count, .fa-thumbs-down",
    "load_more": "button.load-more, a.show-more, button[data-action='load-more'], .comments-load-more",
    "next_page": "a.next, li.next > a, a[rel='next']",
}

LOG = logging.getLogger("bafp.scraper")


# ==========================================================================
# Shared HTTP plumbing
# ==========================================================================
class BaseJSONClient:
    """Thin, polite JSON client: bounded retries, backoff, honours 429.

    Concrete clients supply ``headers`` overrides and the endpoints.
    """

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        if headers:
            self.session.headers.update(headers)
        self.request_count = 0

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any | None:
        """GET a JSON endpoint. Returns ``None`` on permanent failure instead of
        raising, so one bad id cannot abort a long collection run."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                self.request_count += 1

                # Both endpoints may throttle; honour Retry-After when present.
                if response.status_code == 429:
                    wait = float(response.headers.get("Retry-After", 5))
                    LOG.warning("Rate limited (429); sleeping %.1fs", wait)
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                time.sleep(self.delay)
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.max_retries:
                    LOG.warning("Giving up on %s after %d tries: %s", url, attempt, exc)
                    return None
                backoff = self.delay + 2**attempt * 0.5 + random.uniform(0, 0.3)
                LOG.debug("Retry %d/%d for %s in %.1fs", attempt, self.max_retries, url, backoff)
                time.sleep(backoff)
        return None


# ==========================================================================
# Jalali date handling (Digikala only -- Varzesh3 returns ISO-8601 UTC)
# ==========================================================================
_PERSIAN_MONTHS: dict[str, int] = {
    "فروردین": 1, "فروردين": 1,
    "اردیبهشت": 2, "ارديبهشت": 2,
    "خرداد": 3,
    "تیر": 4, "تير": 4,
    "مرداد": 5, "امرداد": 5,
    "شهریور": 6, "شهريور": 6,
    "مهر": 7,
    "آبان": 8,
    "آذر": 9, "اذر": 9,
    "دی": 10, "دي": 10,
    "بهمن": 11,
    "اسفند": 12,
}

# Persian (۰-۹) and Arabic-Indic (٠-٩) digits -> ASCII digits.
_DIGIT_TRANSLATION = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _to_ascii_digits(text: str) -> str:
    return text.translate(_DIGIT_TRANSLATION)


def parse_jalali(value: Any) -> str | None:
    """Parse a Jalali display date such as ``"3 مهر 1405"`` into ISO ``YYYY-MM-DD``.

    Returns ``None`` when the string is missing or unparseable (for example the
    relative form ``"دیروز"`` = "yesterday"), so callers must tolerate ``None``
    rather than assume a date is always available.
    """
    if not value or not isinstance(value, str):
        return None
    try:  # lazy import: the Varzesh3 path works even without jdatetime
        import jdatetime
    except ImportError:  # pragma: no cover - dependency is declared
        LOG.warning("jdatetime is not installed; leaving dates unparsed")
        return None

    cleaned = _to_ascii_digits(value.strip())
    match = re.search(r"(\d{1,2})\s*([^\d\s]+)\s*(\d{2,4})", cleaned)
    if not match:
        return None
    day, month_name, year = match.groups()
    month = _PERSIAN_MONTHS.get(month_name)
    if month is None:
        return None

    year_i = int(year)
    if year_i < 100:  # two-digit years: 05 -> 1405
        year_i += 1400
    try:
        gregorian = jdatetime.date(year_i, month, int(day)).togregorian()
    except ValueError:
        return None
    return gregorian.isoformat()


# ==========================================================================
# Path 1 -- Varzesh3 news comments (unlabelled Persian news corpus)
# ==========================================================================
class Varzesh3NewsClient(BaseJSONClient):
    """Client for Varzesh3's public news-comment JSON API."""

    def latest_news(self, pages: int = 5, skip: int = 0, size: int = 20) -> list[dict[str, Any]]:
        """Return news stubs ``{news_id, news_title, news_link, published_on}``."""
        stubs: list[dict[str, Any]] = []
        for page in range(pages):
            payload = self.get_json(
                V3_LATEST_ENDPOINT, params={"skip": skip + page * size, "l": size}
            )
            items = payload if isinstance(payload, list) else (payload or {}).get("items") or []
            if not items:
                break
            for item in items:
                stubs.append(
                    {
                        "news_id": item.get("id"),
                        "news_title": item.get("title"),
                        "news_link": item.get("link"),
                        "published_on": item.get("publishedOn"),
                    }
                )
            LOG.info("news/latest skip=%d -> %d stubs", skip + page * size, len(items))
        return stubs

    def fetch_comments(
        self,
        news_id: int,
        news_title: str | None = None,
        news_link: str | None = None,
        max_pages: int = 25,
        sort: str = "MostPopular",
    ) -> list[dict[str, Any]]:
        """Walk one article's comment pages.

        Pagination uses ``skip``; the ``l`` cursor echoed by
        ``_links[rel=next]`` is carried forward when present because the API
        does not report a total page count -- ``hasMore`` is the only stop
        signal.
        """
        url = V3_COMMENTS_ENDPOINT.format(news_id=news_id)
        params: dict[str, Any] = {"skip": 0, "sort": sort}
        collected: list[dict[str, Any]] = []

        for _ in range(max_pages):
            payload = self.get_json(url, params=params)
            if not payload:
                break
            items = payload.get("items") or []
            if not items:
                break

            for item in items:
                if item.get("isPending"):  # awaiting moderation: not public text
                    continue
                vote = item.get("vote") or {}
                writer = item.get("writer") or {}
                collected.append(
                    {
                        "comment_id": item.get("id"),
                        "source": "varzesh3_api",
                        "news_id": news_id,
                        "news_title": news_title,
                        "news_link": news_link,
                        "body": item.get("message"),
                        "rate": None,  # news comments carry no star rating
                        "likes": vote.get("likeCount"),
                        "dislikes": vote.get("dislikeCount"),
                        "reply_count": item.get("replyCount"),
                        "created_at_raw": item.get("createdOn"),
                        "created_at": item.get("createdOnUtc"),
                        "user_name": writer.get("name"),
                        "is_buyer": None,
                        "sort": sort,
                    }
                )

            if not payload.get("hasMore"):
                break

            next_href = next(
                (l.get("href") for l in payload.get("_links") or [] if l.get("rel") == "next"),
                None,
            )
            if next_href:
                # Follow the server's own cursor rather than guessing parameters.
                parsed = _query_params(next_href)
                params = {
                    "skip": parsed.get("skip", params["skip"] + V3_PAGE_SIZE),
                    "sort": parsed.get("sort", sort),
                }
                if "l" in parsed:
                    params["l"] = parsed["l"]
            else:
                params["skip"] = params["skip"] + V3_PAGE_SIZE

        LOG.info("news %s -> %d comments (%s)", news_id, len(collected), sort)
        return collected


def _query_params(href: str) -> dict[str, str]:
    """Extract query parameters from an absolute href returned by the API."""
    from urllib.parse import parse_qs, urlparse

    query = urlparse(href).query
    return {k: v[0] for k, v in parse_qs(query).items() if v}


# ==========================================================================
# Path 2 -- Digikala rated comments (labelled training data)
# ==========================================================================
class DigikalaAPIClient(BaseJSONClient):
    """Client for Digikala's public product-comment JSON API."""

    def search_products(self, query: str, pages: int = 1) -> list[dict[str, Any]]:
        """Return product stubs (id, title, review count) for a search query."""
        found: list[dict[str, Any]] = []
        for page in range(1, pages + 1):
            payload = self.get_json(SEARCH_ENDPOINT, params={"q": query, "page": page})
            if not payload:
                break
            data = payload.get("data") or {}
            products = data.get("products") or data.get("items") or []
            if not products:
                break
            for product in products:
                rating = product.get("rating") or {}
                found.append(
                    {
                        "product_id": product.get("id"),
                        "product_title": product.get("title_fa"),
                        "product_rate": rating.get("rate"),
                        "product_review_count": rating.get("count") or 0,
                    }
                )
            LOG.info("search %-14s page %d -> %d products", query, page, len(products))
        return found

    def fetch_comments(
        self,
        product_id: int,
        product_title: str | None = None,
        max_pages: int = 30,
        sort: str = "newest",
    ) -> list[dict[str, Any]]:
        """Fetch up to ``max_pages`` pages of comments for one product.

        ``sort`` must be one of :data:`DIGIKALA_SORTS`; the API rejects integers
        with HTTP 400.
        """
        url = COMMENTS_ENDPOINT.format(product_id=product_id)
        collected: list[dict[str, Any]] = []

        for page in range(1, max_pages + 1):
            payload = self.get_json(url, params={"page": page, "sort": sort})
            if not payload:
                break
            data = payload.get("data") or {}
            comments = data.get("comments") or []
            if not comments:
                break

            for comment in comments:
                reactions = comment.get("reactions") or {}
                collected.append(
                    {
                        "comment_id": comment.get("id"),
                        "source": "digikala_api",
                        "product_id": product_id,
                        "product_title": product_title,
                        "body": comment.get("body"),
                        "rate": comment.get("rate"),
                        "likes": reactions.get("likes"),
                        "dislikes": reactions.get("dislikes"),
                        "created_at_raw": comment.get("created_at"),
                        "created_at": parse_jalali(comment.get("created_at")),
                        "user_name": comment.get("user_name"),
                        "is_buyer": comment.get("is_buyer"),
                        "sort": sort,
                    }
                )

            pager = data.get("pager") or {}
            if page >= (pager.get("total_pages") or 1):
                break

        LOG.info("product %s -> %d comments", product_id, len(collected))
        return collected


# ==========================================================================
# Text filters
# ==========================================================================
def _looks_like_persian(text: str) -> bool:
    persian = len(re.findall(r"[؀-ۿ]", text))
    letters = len(re.findall(r"\w", text))
    return persian >= 4 and (letters == 0 or persian / letters > 0.3)


def _is_usable(body: Any) -> bool:
    """Filter out empty, near-empty or non-Persian comment bodies."""
    if not isinstance(body, str):
        return False
    stripped = body.strip()
    if len(stripped) < 8:  # too short to carry sentiment ("خوب", "بد" etc.)
        return False
    return _looks_like_persian(stripped)


def _has_signal(row: dict[str, Any], min_words: int = 3) -> bool:
    """News comments need a minimum amount of text to be worth plotting."""
    return len(str(row.get("body", "")).split()) >= min_words


# ==========================================================================
# Collection drivers
# ==========================================================================
@dataclass
class CollectionStats:
    """Bookkeeping for one collection run, echoed to the user at the end."""

    products_seen: int = 0
    products_used: int = 0
    comments_raw: int = 0
    comments_kept: int = 0
    duplicates_removed: int = 0
    requests_made: int = 0
    per_product: dict[int, int] = field(default_factory=dict)


def collect_news(
    target: int = 4000,
    topics: Sequence[str] = DEFAULT_NEWS_TOPICS,
    max_articles: int = 60,
    max_pages_per_article: int = 25,
    sort: str = "MostPopular",
    delay: float = DEFAULT_DELAY,
) -> tuple[pd.DataFrame, CollectionStats]:
    """Collect up to ``target`` Persian news comments from Varzesh3.

    Discovers articles through ``news/latest`` (there is no search endpoint),
    optionally filtered by ``topics`` against the article title, then walks each
    article's comment pages. The result is UNLABELLED text: no sentiment label
    is derived here.
    """
    client = Varzesh3NewsClient(delay=delay)
    stats = CollectionStats()
    rows: list[dict[str, Any]] = []
    seen: set[Any] = set()

    stubs = client.latest_news(pages=max(1, max_articles // 20 + 1))
    if topics:
        filtered = [s for s in stubs if any(t in (s.get("news_title") or "") for t in topics)]
        # Keep the topic match but never starve the run: fall back to all stubs.
        stubs = filtered or stubs
    stubs = stubs[:max_articles]
    stats.products_seen = len(stubs)
    LOG.info("discovered %d candidate articles", len(stubs))

    for stub in stubs:
        if len(rows) >= target:
            break
        try:
            comments = client.fetch_comments(
                stub["news_id"],
                news_title=stub.get("news_title"),
                news_link=stub.get("news_link"),
                max_pages=max_pages_per_article,
                sort=sort,
            )
        except KeyboardInterrupt:
            LOG.warning("Interrupted; returning what has been collected so far")
            break

        stats.comments_raw += len(comments)
        kept_here = 0
        for row in comments:
            if not _is_usable(row.get("body")) or not _has_signal(row):
                continue
            key = row.get("comment_id")
            if key is not None and key in seen:
                stats.duplicates_removed += 1
                continue
            if key is not None:
                seen.add(key)
            rows.append(row)
            kept_here += 1
        if kept_here:
            stats.products_used += 1
        stats.per_product[stub["news_id"]] = kept_here
        LOG.info("progress: %d/%d comments from %d articles", len(rows), target, stats.products_used)

    stats.comments_kept = len(rows)
    stats.requests_made = client.request_count
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
    return frame, stats


def collect_digikala(
    target: int = 5000,
    queries: Sequence[str] = DEFAULT_QUERIES,
    max_pages_per_product: int = 5,
    min_review_count: int = 50,
    delay: float = DEFAULT_DELAY,
) -> tuple[pd.DataFrame, CollectionStats]:
    """Collect up to ``target`` usable, STAR-RATED Persian comments from Digikala.

    Strategy: search popular queries, keep products with at least
    ``min_review_count`` reviews, then walk their comment pages (once per sort
    order) until the target is reached. The per-product page cap spreads the
    corpus across many products instead of letting one mega-reviewed item
    dominate the label distribution.
    """
    client = DigikalaAPIClient(delay=delay)
    stats = CollectionStats()
    rows: list[dict[str, Any]] = []
    seen_comment_ids: set[Any] = set()

    candidates: dict[int, dict[str, Any]] = {}
    for query in queries:
        for stub in client.search_products(query, pages=2):
            pid = stub.get("product_id")
            if pid is None or (stub.get("product_review_count") or 0) < min_review_count:
                continue
            existing = candidates.get(pid)
            if existing is None or stub["product_review_count"] > existing["product_review_count"]:
                candidates[pid] = stub

    stats.products_seen = len(candidates)
    ordered = sorted(candidates.values(), key=lambda s: s["product_review_count"], reverse=True)
    LOG.info("discovered %d candidate products (>= %d reviews)", len(ordered), min_review_count)

    for stub in ordered:
        if len(rows) >= target:
            break
        pid = stub["product_id"]
        kept_here = 0

        for sort in DIGIKALA_SORTS:
            if len(rows) >= target:
                break
            try:
                comments = client.fetch_comments(
                    pid,
                    product_title=stub.get("product_title"),
                    max_pages=max_pages_per_product,
                    sort=sort,
                )
            except KeyboardInterrupt:
                LOG.warning("Interrupted; returning what has been collected so far")
                stats.comments_kept = len(rows)
                stats.requests_made = client.request_count
                frame = pd.DataFrame(rows)
                if not frame.empty:
                    frame = frame.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
                return frame, stats

            stats.comments_raw += len(comments)
            for row in comments:
                if not _is_usable(row.get("body")):
                    continue
                key = row.get("comment_id")
                if key is not None and key in seen_comment_ids:
                    stats.duplicates_removed += 1
                    continue
                if key is not None:
                    seen_comment_ids.add(key)
                rows.append(row)
                kept_here += 1

        stats.per_product[pid] = kept_here
        if kept_here:
            stats.products_used += 1
        LOG.info("progress: %d/%d comments from %d products", len(rows), target, stats.products_used)

    stats.comments_kept = len(rows)
    stats.requests_made = client.request_count

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
    return frame, stats


# ==========================================================================
# Path 3 -- Selenium fallback (for sites without a JSON API)
# ==========================================================================
def scrape_with_selenium(
    url: str,
    max_clicks: int = 5,
    headless: bool = True,
    pause: float = 1.5,
) -> list[dict[str, Any]]:
    """Scrape a JS-rendered Persian page using Edge + BeautifulSoup.

    FALLBACK path, used only for sites with no JSON API. Self-contained: all CSS
    selectors come from :data:`SELECTORS`.

    Machine notes: Microsoft Edge 153 is installed and Google Chrome is NOT;
    Selenium Manager must download ``msedgedriver`` on first run, which needs
    access to ``msedgedriver.microsoft.com``. That download is unverified here,
    so failures from this function are expected to be driver/network failures,
    not parser bugs.

    Args:
        url: page to scrape.
        max_clicks: how many "load more" clicks to attempt.
        headless: run the browser without a visible window.
        pause: seconds to wait after each interaction for content to render.

    Returns:
        List of dicts with the same text/rate/vote keys as the API paths.
        Empty list if dependencies or the driver are missing.
    """
    try:
        from bs4 import BeautifulSoup
        from selenium import webdriver
        from selenium.common.exceptions import NoSuchElementException, WebDriverException
        from selenium.webdriver.edge.options import Options as EdgeOptions
    except ImportError as exc:  # pragma: no cover
        LOG.error("Selenium fallback unavailable: %s", exc)
        return []

    options = EdgeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,1000")
    options.add_argument(f"--user-agent={USER_AGENT}")

    try:
        driver = webdriver.Edge(options=options)
    except WebDriverException as exc:
        LOG.error("Could not start Edge (is msedgedriver downloadable?): %s", exc)
        return []

    records: list[dict[str, Any]] = []
    try:
        driver.get(url)
        time.sleep(pause)

        # Click through "load more" so later comments are in the DOM before parse.
        for _ in range(max_clicks):
            try:
                button = driver.find_element("css selector", SELECTORS["load_more"])
            except NoSuchElementException:
                break
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                button.click()
                time.sleep(pause)
            except WebDriverException:
                break

        soup = BeautifulSoup(driver.page_source, "lxml")
        for node in soup.select(SELECTORS["comment_container"]):
            def _text(key: str) -> str | None:
                found = node.select_one(SELECTORS[key])
                return found.get_text(strip=True) if found else None

            def _int(key: str) -> int | None:
                raw = _text(key)
                if not raw:
                    return None
                digits = re.sub(r"\D", "", _to_ascii_digits(raw))
                return int(digits) if digits else None

            records.append(
                {
                    "comment_id": None,
                    "source": "selenium",
                    "body": _text("text"),
                    "rate": _int("rating"),
                    "likes": _int("likes"),
                    "dislikes": _int("dislikes"),
                    "created_at_raw": _text("date"),
                    "created_at": parse_jalali(_text("date")),
                    "user_name": _text("author"),
                }
            )
    finally:
        driver.quit()

    LOG.info("selenium fallback -> %d records", len(records))
    return records


# ==========================================================================
# CLI
# ==========================================================================
def _smoke_test() -> int:
    """Verify both JSON contracts before committing to a long collection run."""
    ok = True

    print("== Varzesh3 news: discovery ==")
    client = Varzesh3NewsClient(delay=0.0)
    stubs = client.latest_news(pages=1)
    if not stubs:
        print("FAIL: news/latest returned nothing (network blocked?)")
        ok = False
    else:
        for stub in stubs[:3]:
            print(f"  id={stub['news_id']}  title={str(stub['news_title'])[:48]}")

    if ok:
        print("\n== Varzesh3 news: comments ==")
        comments = client.fetch_comments(stubs[0]["news_id"], max_pages=1)
        if not comments:
            print(f"FAIL: no comments for article {stubs[0]['news_id']}")
            ok = False
        else:
            first = comments[0]
            print(f"  article {stubs[0]['news_id']}: {len(comments)} comments on page 1")
            for key in ("comment_id", "body", "rate", "likes", "dislikes",
                        "reply_count", "created_at", "user_name"):
                print(f"  {key:<13}= {str(first.get(key))[:60]}")

    print("\n== Digikala: search ==")
    dk = DigikalaAPIClient(delay=0.0)
    products = dk.search_products("موبایل", pages=1)
    if not products:
        print("FAIL: search returned nothing")
        ok = False
    else:
        top = sorted(products, key=lambda s: s["product_review_count"], reverse=True)[:3]
        for stub in top:
            print(f"  id={stub['product_id']:<10} reviews={stub['product_review_count']:<6} "
                  f"title={str(stub['product_title'])[:44]}")
        print("\n== Digikala: rated comments ==")
        comments = dk.fetch_comments(top[0]["product_id"], max_pages=1)
        if not comments:
            print(f"FAIL: no comments for product {top[0]['product_id']}")
            ok = False
        else:
            rates = pd.Series([c["rate"] for c in comments]).value_counts().sort_index().to_dict()
            print(f"  product {top[0]['product_id']}: {len(comments)} comments, rates={rates}")

    print("\n== Selenium fallback: SKIPPED (driver download unverified) ==")
    print("\n" + ("OK: both JSON APIs reachable and schemas match the scraper."
                  if ok else "FAILED: see the first error above."))
    return 0 if ok else 1


def _save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  wrote {len(frame):>6} rows -> {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=("news", "digikala", "both"), default="both",
                        help="news = unlabelled corpus, digikala = labelled training data")
    parser.add_argument("--target", type=int, default=4000, help="target comments per source")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path for the selected source; overrides the defaults "
                             "(ignored when --source both, which writes --news-out/--digikala-out)")
    parser.add_argument("--news-out", type=Path, default=DATA_DIR / "raw_news_comments.csv")
    parser.add_argument("--digikala-out", type=Path, default=DATA_DIR / "raw_digikala_comments.csv")
    parser.add_argument("--queries", type=str, default=",".join(DEFAULT_QUERIES),
                        help="comma-separated Digikala search queries")
    parser.add_argument("--topics", type=str, default=",".join(DEFAULT_NEWS_TOPICS),
                        help="comma-separated Varzesh3 title filters ('' to disable)")
    parser.add_argument("--max-articles", type=int, default=60)
    parser.add_argument("--max-pages-per-product", type=int, default=5,
                        help="comment pages per product per sort; keep this low -- at 30 a "
                             "single mega-reviewed product can supply the whole target and "
                             "dominate the label mix (verified: 600 comments from product "
                             "11801878 on one run)")
    parser.add_argument("--min-review-count", type=int, default=50)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--smoke", action="store_true", help="test the API contracts and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.smoke:
        return _smoke_test()

    # --out addresses the single selected source; --source both keeps the two
    # explicit paths so nothing is silently overwritten.
    news_out = args.out if (args.out is not None and args.source == "news") else args.news_out
    if args.out is not None and args.source == "digikala":
        args.digikala_out = args.out

    if args.source in ("news", "both"):
        topics = tuple(t.strip() for t in args.topics.split(",") if t.strip())
        frame, stats = collect_news(
            target=args.target,
            topics=topics,
            max_articles=args.max_articles,
            delay=args.delay,
        )
        if frame.empty:
            print("No news comments collected. Run with --smoke to diagnose.")
            return 1
        _save(frame, news_out)
        print(f"\nnews: {stats.comments_raw} fetched, {stats.duplicates_removed} duplicates, "
              f"{stats.comments_kept} kept from {stats.products_used} articles, "
              f"{stats.requests_made} requests")

    if args.source in ("digikala", "both"):
        queries = tuple(q.strip() for q in args.queries.split(",") if q.strip())
        frame, stats = collect_digikala(
            target=args.target,
            queries=queries,
            max_pages_per_product=args.max_pages_per_product,
            min_review_count=args.min_review_count,
            delay=args.delay,
        )
        if frame.empty:
            print("No Digikala comments collected. Run with --smoke to diagnose.")
            return 1
        _save(frame, args.digikala_out)
        rates = frame["rate"].value_counts().sort_index().to_dict()
        print(f"\ndigikala: {stats.comments_kept} kept from {stats.products_used} products, "
              f"{stats.requests_made} requests")
        print(f"  rating distribution (0 = unrated, dropped during labelling): {rates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
