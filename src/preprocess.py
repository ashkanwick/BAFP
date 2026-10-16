"""
Persian text cleaning + 3-class labelling for the BAFP capstone.
===============================================================

Two responsibilities live here, in the order the pipeline runs them:

PHASE A -- cleaning (:func:`clean_text`, :func:`clean_dataframe`)
    A ``hazm`` pipeline: normalisation, URL/emoji/Latin removal, tokenisation,
    whitespace collapse.

    COMPOUND-WORD RULE (the part that is a deliberate deviation)
    ------------------------------------------------------------------
    Persian writes compound forms with a zero-width non-joiner (ZWNJ, U+200C):
    "به‌شدت" (so intensely), "ارسال‌آن". ``hazm.Normalizer`` PRESERVES ZWNJ --
    verified on this machine, ``'\\u200c' in normalizer.normalize(text)`` is True
    -- and ``hazm.WordTokenizer(join_verb_parts=True)`` happily returns
    ``'به\\u200cشدت'`` as ONE token.

    This project requires full spaces instead, so :data:`ZWNJ_TO_SPACE` is True
    and the substitution is applied AFTER normalisation but BEFORE tokenisation.
    The order is the whole point: swap it and the tokenizer re-bonds the
    compound and the ZWNJ survives inside a single token.

        "به‌شدت"      -> "به شدت"
        "ارسال‌آن"     -> "ارسال آن"
        "کتاب‌هایم"    -> "کتاب هایم"

    Set :data:`ZWNJ_TO_SPACE` to False only if you want stock ``hazm`` behaviour
    (e.g. to reproduce a published benchmark).

PHASE B -- labelling (:func:`label_from_rate`)
    3-class sentiment from Digikala star ratings:

        rate 1-2 -> 0 negative | rate 3 -> 1 neutral | rate 4-5 -> 2 positive
        rate 0 / null -> DROPPED ("unrated" is not a sentiment)

    Ratings are rounded to the nearest star first: a 4,004-comment scrape
    contained fractional values (2.55, 3.65, ...) that must not be truncated.

    The distribution is skewed about 14:1 between five-star and two-star
    comments, so ``neutral`` is the scarce class; the split keeps a stratified
    slice of it and :mod:`src.train` weights the loss accordingly.

    News comments (Varzesh3) have NO rating. :func:`label_from_votes` can derive
    a weak label from like/dislike counts, but that measures reader agreement,
    not the writer's sentiment, so it is OFF by default and its output is marked
    ``label_source='votes'`` for auditability.

Outputs
-------
    data/cleaned_comments.csv   columns: raw_text, clean_text, eda_text, label,
                                label_name, label_source, rate, likes,
                                dislikes, split, n_tokens, char_len, ...

``clean_text`` is what the model reads (stop words intact -- BERT needs them for
syntax). ``eda_text`` is the stop-word-free, stemmed form used by word clouds and
frequency charts. ``raw_text`` is kept for the dashboard's before/after demo.

Usage
-----
    python -m src.preprocess --in data/raw_digikala_comments.csv \
                             --out data/cleaned_comments.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

LOG = logging.getLogger("bafp.preprocess")

__all__ = [
    "clean_text", "clean_dataframe", "expand_zwnj",
    "label_from_rate", "label_from_votes", "make_splits",
    "load_raw", "summarise",
    "LABEL_NAMES", "LABEL_NAMES_FA", "RATE_TO_LABEL",
    "ZWNJ_TO_SPACE", "ZWNJ",
]

# --------------------------------------------------------------------------
# The compound-word policy. True == full space replaces ZWNJ (project rule).
# --------------------------------------------------------------------------
ZWNJ_TO_SPACE = True
ZWNJ = "‌"  # U+200C zero-width non-joiner

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES_FA = {0: "منفی", 1: "خنثی", 2: "مثبت"}
RATE_TO_LABEL: dict[int, int] = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

# Weak-label thresholds for vote-based news comments (unused unless enabled).
VOTES_NET_POSITIVE = 3  # likes - dislikes >= this  -> positive
VOTES_NET_NEGATIVE = -3  # likes - dislikes <= this -> negative
# Anything between is "neutral", which for votes mostly means "low engagement",
# so this is a poor proxy for sentiment -- hence default off.

# --------------------------------------------------------------------------
# Regex patterns
# --------------------------------------------------------------------------
PERSIAN_CHARS = r"؀-ۿ"
LATIN_RUN = re.compile(r"[A-Za-z]{2,}")
URL = re.compile(
    r"(?:https?://|www\.)[^\s]+|[\w\-]+\.(?:ir|com|net|org|info)(?:/[^\s]*)?", re.I
)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
MENTION = re.compile(r"@[\w_]+")
HASHTAG = re.compile(r"#([\w\-۰-۹]+)")
EMOJI = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U00002600-\U000027BF"
    "\U0001FA70-\U0001FAFF️‍"
    "]"
)
REPEATED_PUNCT = re.compile(r"([!?.،،\-–—])\1{2,}")
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ASCII_DIGITS = "0123456789"
_DIGIT_TRANSLATION = str.maketrans(PERSIAN_DIGITS + "٠١٢٣٤٥٦٧٨٩", ASCII_DIGITS * 2)
NON_PERSIAN_TOKEN = re.compile(rf"^[^{PERSIAN_CHARS}\d]+$")
KEEP_AT_LEAST = 2  # minimum characters for a token to survive filtering


class _HazmTools:
    """Lazily built hazm objects.

    ``hazm.Lemmatizer`` is NOT used: on hazm 0.12.1 its ``lemmatize()`` returns
    the input unchanged unless external embedding/word-list files are supplied,
    which would silently make it a no-op. ``hazm.Stemmer`` is rule-based and
    verified working (کتاب‌ها -> کتاب), so it does the morphological job here.
    """

    def __init__(self) -> None:
        self._normalizer = None
        self._tokenizer = None
        self._stemmer = None
        self._stopwords = None

    @property
    def normalizer(self):
        if self._normalizer is None:
            from hazm import Normalizer

            # persian_numbers=False keeps ASCII digits, so downstream length and
            # number features behave the same regardless of how the site rendered them.
            self._normalizer = Normalizer(persian_numbers=False)
        return self._normalizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from hazm import WordTokenizer

            self._tokenizer = WordTokenizer(join_verb_parts=False, separate_emoji=False)
        return self._tokenizer

    @property
    def stemmer(self):
        if self._stemmer is None:
            from hazm import Stemmer

            self._stemmer = Stemmer()
        return self._stemmer

    @property
    def stopwords(self) -> frozenset[str]:
        if self._stopwords is None:
            from hazm.utils import stopwords_list

            self._stopwords = frozenset(stopwords_list())
        return self._stopwords


_TOOLS = _HazmTools()


def expand_zwnj(text: str) -> str:
    """Replace every ZWNJ with a full space when :data:`ZWNJ_TO_SPACE` is set."""
    if ZWNJ_TO_SPACE:
        return text.replace(ZWNJ, " ")
    return text


def clean_text(
    text: Any,
    *,
    remove_latin: bool = True,
    keep_hashtag_words: bool = True,
    for_eda: bool = False,
    remove_stopwords: bool = False,
    stem: bool = False,
) -> str:
    """Normalise, strip noise and re-space a Persian comment.

    Args:
        text: raw comment; non-strings become ``""`` so a dirty CSV cannot crash
            the pipeline.
        remove_latin: drop Latin runs of 2+ letters (brand names, URLs' residue).
        keep_hashtag_words: strip the ``#`` but keep the word ("#ایران" -> "ایران").
        for_eda: also drop stop words and stem -- pass True only for the
            ``eda_text`` column, never for model input.
        remove_stopwords: explicit override of ``for_eda``'s stop-word step.
        stem: explicit override of ``for_eda``'s stemming step.

    Returns:
        Cleaned, single-spaced text. Empty string when nothing Persian survives.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    tools = _TOOLS
    cleaned = tools.normalizer.normalize(text)
    cleaned = cleaned.translate(_DIGIT_TRANSLATION)

    cleaned = EMAIL.sub(" ", cleaned)
    cleaned = URL.sub(" ", cleaned)
    cleaned = MENTION.sub(" ", cleaned)
    cleaned = HASHTAG.sub(r"\1" if keep_hashtag_words else " ", cleaned)
    cleaned = EMOJI.sub(" ", cleaned)
    cleaned = REPEATED_PUNCT.sub(r"\1", cleaned)
    if remove_latin:
        cleaned = LATIN_RUN.sub(" ", cleaned)

    # >>> the compound-word rule: ZWNJ -> space BEFORE tokenising <<<
    cleaned = expand_zwnj(cleaned)

    tokens = tools.tokenizer.tokenize(cleaned)
    drop_stop = remove_stopwords or for_eda
    do_stem = stem or for_eda

    kept: list[str] = []
    for token in tokens:
        if not token or token.isspace():
            continue
        if NON_PERSIAN_TOKEN.match(token) and not token.isdigit():
            continue  # punctuation and stray Latin
        if drop_stop and token in tools.stopwords:
            continue
        if do_stem:
            token = tools.stemmer.stem(token)
        if len(token) < KEEP_AT_LEAST and not token.isdigit():
            continue
        kept.append(token)

    return " ".join(kept).strip()


def label_from_rate(rate: Any) -> int | None:
    """Map a Digikala star rating to a 3-class label id, or ``None`` to drop.

    ``rate`` is NOT guaranteed integral. A 4,004-comment scrape contained
    fractional values (0.5, 1.2, 2.55, 3.65, ...) alongside integers, so the
    value is rounded to the nearest star and clamped to 1..5 BEFORE binning.
    Truncation would be a silent labelling bug: int(3.65) == 3 puts a clearly
    positive comment in ``neutral``.

    ``rate`` 0 / null / < 0.5 means "unrated", which is not a sentiment class.
    """
    if rate is None or (isinstance(rate, float) and pd.isna(rate)):
        return None
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return None
    stars = int(np.floor(value + 0.5))  # round half up, like the star widget
    if stars < 1:  # 0 or sub-0.5 -> unrated
        return None
    return RATE_TO_LABEL.get(min(stars, 5))


def label_from_votes(likes: Any, dislikes: Any) -> int | None:
    """WEAK label from vote counts (news comments). No rating involved.

    Measures how much other readers agreed with the comment, not whether the
    comment is positive -- genuinely noisy, off by default.
    """
    try:
        net = int(likes or 0) - int(dislikes or 0)
    except (TypeError, ValueError):
        return None
    if net >= VOTES_NET_POSITIVE:
        return 2
    if net <= VOTES_NET_NEGATIVE:
        return 0
    return 1


# ==========================================================================
# DataFrame pipeline
# ==========================================================================
def make_splits(frame: pd.DataFrame, label_col: str, seed: int = 42,
                train_size: float = 0.8, val_size: float = 0.1) -> pd.Series:
    """Stratified train/val/test assignment as a ``split`` column.

    Stratification is what keeps ``neutral`` present in val/test at all: with a
    ~14:1 skew, a random split can leave the rarest class almost absent.
    """
    from sklearn.model_selection import train_test_split

    # Positional labels, so this works whether or not the caller's index is
    # contiguous (rows are dropped and de-duplicated upstream).
    labels = frame[label_col].reset_index(drop=True)
    positions = pd.Index(range(len(labels)))
    test_size = 1.0 - train_size - val_size
    train_pos, rest_pos = train_test_split(
        positions, test_size=test_size + val_size, random_state=seed, stratify=labels
    )
    val_pos, test_pos = train_test_split(
        rest_pos,
        test_size=test_size / (test_size + val_size),
        random_state=seed,
        stratify=labels.iloc[rest_pos],
    )
    split = pd.Series("none", index=frame.index, dtype="object")
    split.iloc[train_pos] = "train"
    split.iloc[val_pos] = "val"
    split.iloc[test_pos] = "test"
    return split


def clean_dataframe(
    frame: pd.DataFrame,
    *,
    text_col: str = "body",
    label_source: str = "rate",
    min_tokens: int = 3,
    min_chars: int = 10,
    seed: int = 42,
    keep_unlabelled: bool = False,
) -> pd.DataFrame:
    """Clean, label and split one raw-comment DataFrame.

    Args:
        frame: output of :func:`src.scraper.collect_digikala` / ``collect_news``.
        text_col: column holding the comment text (``body`` for both APIs).
        label_source: ``rate`` (Digikala stars), ``votes`` (weak, news) or
            ``none`` (unlabelled corpus -- no split, no label).
        min_tokens: comments shorter than this after cleaning are dropped.
        keep_unlabelled: with ``label_source='none'``, keep rows for EDA even
            though they cannot train a model.
        seed: split determinism.

    Returns:
        DataFrame with ``raw_text``/``clean_text``/``eda_text`` plus label and
        split columns, de-duplicated on cleaned text.
    """
    if text_col not in frame.columns:
        raise KeyError(f"{text_col!r} not in columns {list(frame.columns)}")

    frame = frame.copy()
    # Scrub whitespace (the sites emit newlines and non-breaking spaces) before
    # anything else, so the ZWNJ rule sees plain text.
    frame["raw_text"] = (
        frame[text_col].astype("string").fillna("").str.replace(r"[ \t\r\n]+", " ", regex=True).str.strip()
    )

    if "source" not in frame:
        frame["source"] = "unknown"

    LOG.info("cleaning %d rows", len(frame))
    try:
        from tqdm.auto import tqdm
        frame["clean_text"] = [clean_text(t) for t in tqdm(frame["raw_text"], desc="Cleaning", unit="row")]
        frame["eda_text"] = [clean_text(t, for_eda=True) for t in tqdm(frame["raw_text"], desc="EDA text", unit="row")]
    except ImportError:
        frame["clean_text"] = frame["raw_text"].map(clean_text)
        frame["eda_text"] = frame["raw_text"].map(lambda t: clean_text(t, for_eda=True))
    frame["n_tokens"] = frame["clean_text"].str.split().map(lambda w: len(w) if w != [""] else 0)
    frame["char_len"] = frame["clean_text"].str.len()
    frame["word_len"] = frame["raw_text"].str.split().map(lambda w: len(w) if w else 0)

    before = len(frame)
    frame = frame[frame["n_tokens"] >= min_tokens]
    frame = frame[frame["char_len"] >= min_chars]
    LOG.info("dropped %d rows for being too short/empty after cleaning", before - len(frame))

    # Exact duplicates of the same cleaned text (same comment on several products
    # or re-fetched across sort orders) would leak across the split.
    before = len(frame)
    frame = frame.drop_duplicates(subset=["clean_text"], keep="first").reset_index(drop=True)
    LOG.info("removed %d exact duplicate comments", before - len(frame))

    if label_source == "rate":
        frame["label"] = frame.get("rate", pd.Series([None] * len(frame))).map(label_from_rate)
        frame["label_source"] = "digikala_rate"
    elif label_source == "votes":
        frame["label"] = [label_from_votes(l, d) for l, d in zip(frame.get("likes", []), frame.get("dislikes", []))]
        frame["label_source"] = "varzesh3_votes_weak"
    elif label_source == "none":
        frame["label"] = None
        frame["label_source"] = "none"
    else:
        raise ValueError(f"unknown label_source {label_source!r}")

    if label_source == "none":
        frame["label_name"] = None
        frame["split"] = "none"
        if not keep_unlabelled:
            raise ValueError(
                "label_source='none' produces no training data; pass keep_unlabelled=True "
                "to build an EDA/live-testing corpus"
            )
    else:
        dropped = int(frame["label"].isna().sum())
        LOG.info("dropping %d rows with no derivable label (e.g. rate=0 'unrated')", dropped)
        frame = frame[frame["label"].notna()].copy()
        frame["label"] = frame["label"].astype(int)
        frame["label_name"] = frame["label"].map(LABEL_NAMES)
        LOG.info("label distribution: %s", frame["label_name"].value_counts().to_dict())
        frame["split"] = make_splits(frame, "label", seed=seed)
        LOG.info("split sizes: %s", frame["split"].value_counts().to_dict())

    return frame.reset_index(drop=True)


def load_raw(path: Path) -> pd.DataFrame:
    """Read a raw collection CSV the way the scraper wrote it (utf-8-sig)."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    # A CSV round-trip turns absent rates into NaN; label_from_rate wants None-ish.
    if "rate" in frame:
        frame["rate"] = frame["rate"].where(pd.notna(frame["rate"]), None)
    return frame


def summarise(frame: pd.DataFrame) -> dict[str, Any]:
    """Small human-readable report for the CLI and the dashboard."""
    stats: dict[str, Any] = {
        "rows": int(len(frame)),
        "sources": frame["source"].value_counts().to_dict(),
        "mean_clean_tokens": float(frame["n_tokens"].mean()) if len(frame) else 0.0,
        "zwnj_left_in_clean_text": int(frame["clean_text"].str.contains(ZWNJ, regex=False).sum()),
    }
    if frame["label"].notna().any():
        stats["labels"] = (
            frame.loc[frame["label"].notna(), "label_name"].value_counts().to_dict()
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in", dest="inp", type=Path, default=DATA_DIR / "raw_digikala_comments.csv")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "cleaned_comments.csv")
    parser.add_argument("--text-col", default="body")
    parser.add_argument("--label-source", choices=("rate", "votes", "none"), default="rate")
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="print before/after cleaning examples")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.demo:
        samples = ["این محصول به‌شدت بد بود و ارسال‌آن دیر رسید.",
                   "کیفیت عالی‌ست و پیشنهاد می‌کنم 🙂 https://x.com/a",
                   "۵ ستاره، کتاب‌های خوبی هستند"]
        for sample in samples:
            print(f"RAW : {sample}")
            print(f"MOD : {clean_text(sample)}")
            print(f"EDA : {clean_text(sample, for_eda=True)}\n")
        print(f"ZWNJ_TO_SPACE = {ZWNJ_TO_SPACE}")
        return 0

    if not args.inp.exists():
        print(f"Input not found: {args.inp}\nCollect data first:\n"
              f"  python -m src.scraper --source digikala --out {args.inp}")
        return 1

    frame = clean_dataframe(
        load_raw(args.inp),
        text_col=args.text_col,
        label_source=args.label_source,
        min_tokens=args.min_tokens,
        min_chars=args.min_chars,
        seed=args.seed,
        keep_unlabelled=(args.label_source == "none"),
    )
    if frame.empty:
        print("Nothing survived cleaning. Check --text-col and the raw file.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, encoding="utf-8-sig")

    stats = summarise(frame)
    print("\n" + "=" * 62)
    print("Preprocessing complete")
    print("=" * 62)
    print(f"  output                 : {args.out}")
    print(f"  rows                   : {stats['rows']}")
    print(f"  sources                : {stats['sources']}")
    print(f"  mean tokens (clean)    : {stats['mean_clean_tokens']:.1f}")
    print(f"  ZWNJ chars left        : {stats['zwnj_left_in_clean_text']} "
          f"(expected 0 with ZWNJ_TO_SPACE={ZWNJ_TO_SPACE})")
    if "labels" in stats:
        print(f"  labels                 : {stats['labels']}")
    print(f"  splits                 : {frame['split'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
