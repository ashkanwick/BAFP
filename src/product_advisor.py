"""
Digikala Product Sentiment Advisor for BAFP.
============================================

Takes a Digikala product URL or DKP code, scrapes recent customer comments,
analyzes their sentiment using fine-tuned ParsBERT, and outputs a clear,
data-driven purchase recommendation (خرید / عدم خرید / احتیاط).

Usage:
------
    python -m src.product_advisor --url "https://www.digikala.com/product/dkp-18575915/"
    python -m src.product_advisor --url "dkp-18575915" --max-comments 60
    python analyze_product.py "https://www.digikala.com/product/dkp-18575915/"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "parsbert-sentiment-3class"

LOG = logging.getLogger("bafp.advisor")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
}

__all__ = [
    "extract_product_id",
    "fetch_product_info",
    "fetch_product_comments",
    "analyze_comments_sentiment",
    "generate_purchase_recommendation",
    "analyze_digikala_product",
    "RecommendationVerdict",
    "ProductAnalysisReport",
]


# ==========================================================================
# 1. URL & ID Parsing
# ==========================================================================
def extract_product_id(url_or_id: str | int) -> int | None:
    """Extract integer product ID from any Digikala URL, slug, or DKP string.

    Examples:
        - 'https://www.digikala.com/product/dkp-18575915/slug/' -> 18575915
        - 'https://digikala.com/product/18575915'               -> 18575915
        - 'dkp-18575915'                                       -> 18575915
        - '18575915'                                           -> 18575915
    """
    if isinstance(url_or_id, int):
        return url_or_id
    raw = str(url_or_id).strip()
    if not raw:
        return None

    patterns = (
        r"dkp-(\d+)",
        r"/product/(\d+)",
        r"\b(\d{5,10})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


# ==========================================================================
# 2. Digikala API Fetchers
# ==========================================================================
def fetch_product_info(product_id: int, timeout: int = 15) -> dict[str, Any]:
    """Retrieve metadata for a product via Digikala v2 API."""
    url = f"https://api.digikala.com/v2/product/{product_id}/"
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        if response.status_code == 404:
            return {"error": f"Product with ID {product_id} not found on Digikala."}
        response.raise_for_status()
        data = response.json().get("data", {})
        product = data.get("product", {})

        rating = product.get("rating", {})
        default_variant = product.get("default_variant", {})
        price = default_variant.get("price", {})
        selling_price_rial = price.get("selling_price") or 0

        images = product.get("images", {})
        main_img = (images.get("main", {}) or {}).get("url", [])
        img_url = main_img[0] if isinstance(main_img, list) and main_img else None

        return {
            "product_id": product_id,
            "title_fa": product.get("title_fa") or "بدون عنوان",
            "title_en": product.get("title_en") or "",
            "brand": product.get("brand", {}).get("title_fa") or "نامشخص",
            "category": (product.get("category", {}) or {}).get("title_fa") or "",
            "url": f"https://www.digikala.com/product/dkp-{product_id}/",
            "image_url": img_url,
            "status": product.get("status") or "unknown",
            "rating_score": rating.get("rate"),  # e.g. 91.1 (percentage/points)
            "rating_count": rating.get("count") or 0,
            "price_toman": selling_price_rial // 10 if selling_price_rial else None,
            "has_discount": bool(price.get("discount_percent", 0) > 0),
            "discount_percent": price.get("discount_percent", 0),
        }
    except Exception as exc:
        LOG.warning("Could not fetch product details for %s: %s", product_id, exc)
        return {
            "product_id": product_id,
            "title_fa": f"کالای دیجی‌کالا (کد {product_id})",
            "url": f"https://www.digikala.com/product/dkp-{product_id}/",
            "error": str(exc),
        }


def fetch_product_comments(
    product_id: int,
    max_comments: int = 50,
    sort: str = "newest",
    delay: float = 0.25,
    timeout: int = 15,
) -> list[dict[str, Any]]:
    """Scrape customer comments for a product across pages."""
    url = f"https://api.digikala.com/v1/product/{product_id}/comments/"
    collected: list[dict[str, Any]] = []
    page = 1
    max_pages = (max_comments + 19) // 20

    while page <= max_pages and len(collected) < max_comments:
        try:
            resp = requests.get(
                url,
                params={"page": page, "sort": sort},
                headers=DEFAULT_HEADERS,
                timeout=timeout,
            )
            if resp.status_code != 200:
                break
            payload = resp.json()
            data = payload.get("data", {})
            comments = data.get("comments", [])
            if not comments:
                break

            for c in comments:
                body = (c.get("body") or "").strip()
                if len(body) < 5:  # skip empty or single-word noises
                    continue
                reactions = c.get("reactions") or {}
                collected.append(
                    {
                        "comment_id": c.get("id"),
                        "body": body,
                        "rate": c.get("rate"),
                        "is_buyer": bool(c.get("is_buyer")),
                        "likes": reactions.get("likes", 0),
                        "dislikes": reactions.get("dislikes", 0),
                        "created_at": c.get("created_at"),
                        "advantages": c.get("advantages") or [],
                        "disadvantages": c.get("disadvantages") or [],
                    }
                )
                if len(collected) >= max_comments:
                    break

            pager = data.get("pager") or {}
            total_pages = pager.get("total_pages", 1)
            if page >= total_pages:
                break
            page += 1
            time.sleep(delay)
        except Exception as exc:
            LOG.warning("Comments fetch error on page %d: %s", page, exc)
            break

    return collected


# ==========================================================================
# 3. Sentiment Analysis Engine
# ==========================================================================
def _get_predictor(model_dir: Path | None = None) -> Any:
    """Load or initialize SentimentPredictor."""
    target_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    if not (target_dir / "config.json").exists():
        LOG.warning("Checkpoint %s not found. Using fallback sentiment heuristics.", target_dir)
        return None
    try:
        from src.train import SentimentPredictor

        return SentimentPredictor(model_dir=target_dir)
    except Exception as exc:
        LOG.error("Failed to load SentimentPredictor: %s", exc)
        return None


def analyze_comments_sentiment(
    comments: list[dict[str, Any]],
    predictor: Any | None = None,
    model_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run sentiment analysis on every scraped comment."""
    if predictor is None:
        predictor = _get_predictor(model_dir)

    analyzed = []
    for c in comments:
        text = c["body"]
        if predictor is not None:
            pred = predictor.predict(text)
            label = pred["label"] or "neutral"
            label_fa = pred["label_fa"] or "خنثی"
            conf = float(pred["confidence"])
            probs = pred.get("probabilities", {})
        else:
            # Fallback heuristic using user rate stars if model not present
            star = c.get("rate") or 0
            if star >= 4:
                label, label_fa = "positive", "مثبت"
            elif star in (1, 2):
                label, label_fa = "negative", "منفی"
            else:
                label, label_fa = "neutral", "خنثی"
            conf = 0.70
            probs = {label: 0.70}

        analyzed.append({
            **c,
            "sentiment": label,
            "sentiment_fa": label_fa,
            "confidence": conf,
            "probabilities": probs,
        })

    return analyzed


# ==========================================================================
# 4. Recommendation Decision Engine
# ==========================================================================
class RecommendationVerdict:
    RECOMMENDED = "🟢 پیشنهاد به خرید"
    CAUTION = "🟡 خرید با احتیاط / بررسی بیشتر"
    NOT_RECOMMENDED = "🔴 پیشنهاد به عدم خرید"
    INSUFFICIENT_DATA = "⚪ داده ناکافی برای تصمیم‌گیری"


@dataclass
class RecommendationResult:
    verdict: str
    verdict_en: str
    score_out_of_100: float
    summary_text_fa: str
    reasons_fa: list[str]
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    buyer_positive_ratio: float
    total_comments: int
    buyer_comments_count: int
    top_advantages: list[str] = field(default_factory=list)
    top_disadvantages: list[str] = field(default_factory=list)


def generate_purchase_recommendation(
    analyzed_comments: list[dict[str, Any]],
    product_info: dict[str, Any] | None = None,
) -> RecommendationResult:
    """Analyze the sentiment distribution and produce an actionable recommendation."""
    total = len(analyzed_comments)
    if total < 3:
        return RecommendationResult(
            verdict=RecommendationVerdict.INSUFFICIENT_DATA,
            verdict_en="INSUFFICIENT_DATA",
            score_out_of_100=50.0,
            summary_text_fa="تعداد نظرات ثبت‌شده برای این کالا کافی نیست تا بتوان تحلیل آماری مطمئنی ارائه داد.",
            reasons_fa=["حداقل ۳ نظر فعال برای نتیجه‌گیری لازم است."],
            positive_ratio=0.0,
            negative_ratio=0.0,
            neutral_ratio=0.0,
            buyer_positive_ratio=0.0,
            total_comments=total,
            buyer_comments_count=0,
        )

    pos_count = sum(1 for c in analyzed_comments if c["sentiment"] == "positive")
    neg_count = sum(1 for c in analyzed_comments if c["sentiment"] == "negative")
    neu_count = sum(1 for c in analyzed_comments if c["sentiment"] == "neutral")

    pos_ratio = pos_count / total
    neg_ratio = neg_count / total
    neu_ratio = neu_count / total

    # Buyer specific stats
    buyers = [c for c in analyzed_comments if c.get("is_buyer")]
    if buyers:
        buyer_pos = sum(1 for c in buyers if c["sentiment"] == "positive")
        buyer_pos_ratio = buyer_pos / len(buyers)
    else:
        buyer_pos_ratio = pos_ratio

    # Aggregate advantages / disadvantages
    all_advs: list[str] = []
    all_diss: list[str] = []
    for c in analyzed_comments:
        for adv in c.get("advantages") or []:
            if isinstance(adv, str) and adv.strip():
                all_advs.append(adv.strip())
        for dis in c.get("disadvantages") or []:
            if isinstance(dis, str) and dis.strip():
                all_diss.append(dis.strip())

    top_advs = [w for w, _ in Counter(all_advs).most_common(5)]
    top_diss = [w for w, _ in Counter(all_diss).most_common(5)]

    # Scoring Formula (0 to 100):
    # Base = 50 + (positive_ratio - negative_ratio) * 45 + (buyer_positive_ratio - 0.5) * 10
    base_score = 50.0 + (pos_ratio - neg_ratio) * 45.0
    if buyers:
        base_score += (buyer_pos_ratio - 0.5) * 10.0
    score = max(0.0, min(100.0, round(base_score, 1)))

    reasons: list[str] = []

    # Decision Matrix:
    if pos_ratio >= 0.70 and neg_ratio <= 0.18:
        verdict = RecommendationVerdict.RECOMMENDED
        verdict_en = "RECOMMENDED"
        summary = (
            f"اکثریت قاطع خریداران ({pos_ratio * 100:.0f}٪) حس مثبت داشته و از خرید خود رضایت دارند. "
            "کیفیت و کارایی کالا از دیدگاه مصرف‌کنندگان مطلوب ارزیابی شده است."
        )
        reasons.append(f"بیش از ۷۰٪ نظرات ({pos_ratio * 100:.1f}٪) بار معنایی مثبت دارند.")
        if buyers and buyer_pos_ratio >= 0.75:
            reasons.append(f"رضایت خریداران تاییدشده {buyer_pos_ratio * 100:.1f}٪ است.")
        if neg_ratio < 0.10:
            reasons.append(f"شکایت و نارضایتی شدید بسیار نادر است (کمتر از {neg_ratio * 100:.1f}٪).")
    elif neg_ratio >= 0.30 or (pos_ratio < 0.45 and neg_ratio > 0.25):
        verdict = RecommendationVerdict.NOT_RECOMMENDED
        verdict_en = "NOT_RECOMMENDED"
        summary = (
            f"سهم بالای نظرات منفی ({neg_ratio * 100:.0f}٪) نشان‌دهنده مشکلات فنی، کیفیت نامناسب یا عدم تطابق "
            "با مشخصات اعلام‌شده است. خرید این کالا توصیه نمی‌شود."
        )
        reasons.append(f"نرخ نارضایتی بالاست: {neg_ratio * 100:.1f}٪ از کاربران نظر منفی ثبت کرده‌اند.")
        if top_diss:
            reasons.append(f"نقاط ضعف پربسامد گزارش‌شده: {', '.join(top_diss[:3])}")
    else:
        verdict = RecommendationVerdict.CAUTION
        verdict_en = "CAUTION"
        summary = (
            "نظرات کاربران دوقطبی یا میانه است. بخشی از خریداران راضی و بخشی دیگر ناراضی بوده‌اند. "
            "پیشنهاد می‌شود نقاط ضعف گزارش‌شده را قبل از ثبت سفارش دقیقاً بررسی کنید."
        )
        reasons.append(f"توزیع نظرات مختلط است: {pos_ratio * 100:.1f}٪ مثبت در برابر {neg_ratio * 100:.1f}٪ منفی.")
        reasons.append(f"{neu_ratio * 100:.1f}٪ از کاربران تجربه‌ای معمولی یا خنثی داشته‌اند.")
        if top_diss:
            reasons.append(f"ایرادات مکرر: {', '.join(top_diss[:3])}")

    return RecommendationResult(
        verdict=verdict,
        verdict_en=verdict_en,
        score_out_of_100=score,
        summary_text_fa=summary,
        reasons_fa=reasons,
        positive_ratio=round(pos_ratio, 3),
        negative_ratio=round(neg_ratio, 3),
        neutral_ratio=round(neu_ratio, 3),
        buyer_positive_ratio=round(buyer_pos_ratio, 3),
        total_comments=total,
        buyer_comments_count=len(buyers),
        top_advantages=top_advs,
        top_disadvantages=top_diss,
    )


# ==========================================================================
# 5. Full Pipeline Runner
# ==========================================================================
@dataclass
class ProductAnalysisReport:
    product: dict[str, Any]
    recommendation: RecommendationResult
    comments_sample: list[dict[str, Any]]
    analyzed_count: int
    elapsed_seconds: float


def analyze_digikala_product(
    url_or_id: str | int,
    max_comments: int = 50,
    model_dir: Path | None = None,
) -> ProductAnalysisReport:
    """End-to-end pipeline: parse URL -> fetch info -> scrape comments -> analyze sentiment -> buy recommendation."""
    started = time.perf_counter()
    pid = extract_product_id(url_or_id)
    if pid is None:
        raise ValueError(f"Could not parse valid Digikala product ID from '{url_or_id}'.")

    LOG.info("Scraping product metadata for ID %d...", pid)
    product_info = fetch_product_info(pid)

    LOG.info("Scraping up to %d customer comments for product %d...", max_comments, pid)
    raw_comments = fetch_product_comments(pid, max_comments=max_comments)

    LOG.info("Analyzing sentiments with ParsBERT for %d comments...", len(raw_comments))
    analyzed = analyze_comments_sentiment(raw_comments, model_dir=model_dir)

    LOG.info("Generating purchase recommendation...")
    rec = generate_purchase_recommendation(analyzed, product_info)
    elapsed = round(time.perf_counter() - started, 2)

    return ProductAnalysisReport(
        product=product_info,
        recommendation=rec,
        comments_sample=analyzed,
        analyzed_count=len(analyzed),
        elapsed_seconds=elapsed,
    )


# ==========================================================================
# 6. Terminal Formatter
# ==========================================================================
def print_terminal_report(report: ProductAnalysisReport) -> None:
    """Print an aesthetic, readable Persian report in terminal."""
    prod = report.product
    rec = report.recommendation

    width = 72
    print("\n" + "=" * width)
    print(" 🛍️  سامانه هوشمند ارزیابی احساسات و مشاوره خرید دیجی‌کالا (BAFP) ")
    print("=" * width)
    print(f"📦 کالا: {prod.get('title_fa')}")
    print(f"🏷️  برند: {prod.get('brand')}   |   شناسه: dkp-{prod.get('product_id')}")
    if prod.get("price_toman"):
        print(f"💰 قیمت فعلی: {prod.get('price_toman'):,} تومان")
    if prod.get("rating_score"):
        print(f"⭐ رضایت سایت دیجی‌کالا: {prod.get('rating_score')}٪ (از {prod.get('rating_count'):,} رأی)")
    print(f"🔗 پیوند: {prod.get('url')}")
    print("-" * width)

    print(f"\n📊 آمار تحلیل نظرات ({report.analyzed_count} نظر مشتریان):")
    print(f"   🟢 احساسات مثبت : {rec.positive_ratio * 100:.1f}٪")
    print(f"   🟡 احساسات خنثی : {rec.neutral_ratio * 100:.1f}٪")
    print(f"   🔴 احساسات منفی : {rec.negative_ratio * 100:.1f}٪")
    if rec.buyer_comments_count:
        print(f"   🛒 رضایت خریداران قطعی: {rec.buyer_positive_ratio * 100:.1f}٪ (از {rec.buyer_comments_count} خریدار)")

    print("\n" + "-" * width)
    print(f"🎯 نتیجه و جمع‌بندی: {rec.verdict}")
    print(f"📈 امتیاز ارزش خرید مدل: {rec.score_out_of_100:.1f} از ۱۰۰")
    print("-" * width)
    print(f"\n💡 توضیح تحلیل‌گر:\n   {rec.summary_text_fa}\n")

    if rec.reasons_fa:
        print("📌 دلایل شاخص:")
        for r in rec.reasons_fa:
            print(f"   • {r}")

    if rec.top_advantages:
        print(f"\n👍 نقاط قوت پررنگ: {', '.join(rec.top_advantages)}")
    if rec.top_disadvantages:
        print(f"👎 نقاط ضعف پررنگ: {', '.join(rec.top_disadvantages)}")

    print(f"\n⏱️  زمان استنتاج: {report.elapsed_seconds} ثانیه روی CPU")
    print("=" * width + "\n")


# ==========================================================================
# 7. CLI Entry Point
# ==========================================================================
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "target_url",
        nargs="?",
        default=None,
        help="Digikala product URL or DKP identifier (e.g. https://www.digikala.com/product/dkp-18575915/)",
    )
    parser.add_argument("--url", dest="flag_url", default=None, help="Explicit URL flag")
    parser.add_argument(
        "--max-comments",
        type=int,
        default=40,
        help="Maximum comments to scrape and evaluate (default: 40)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to fine-tuned ParsBERT checkpoint",
    )
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Save analyzed comments to a CSV file",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_input = args.target_url or args.flag_url
    if not raw_input:
        try:
            raw_input = input("لطفاً لینک یا کد کالای دیجی‌کالا را وارد کنید: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nعملیات لغو شد.")
            return 1

    if not raw_input:
        print("خطا: آدرس کالا وارد نشده است.")
        return 1

    try:
        report = analyze_digikala_product(
            raw_input,
            max_comments=args.max_comments,
            model_dir=args.model_dir,
        )
    except Exception as exc:
        print(f"\nخطا در ارزیابی کالا: {exc}")
        return 1

    if args.json:
        payload = {
            "product": report.product,
            "recommendation": asdict(report.recommendation),
            "analyzed_count": report.analyzed_count,
            "elapsed_seconds": report.elapsed_seconds,
            "comments_sample": report.comments_sample,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_terminal_report(report)

    if args.csv_out:
        import pandas as pd

        df = pd.DataFrame(report.comments_sample)
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv_out, index=False, encoding="utf-8-sig")
        print(f"خروجی نظرات در {args.csv_out} ذخیره شد.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
