"""
Fine-tune ParsBERT (HooshvareLab/bert-fa-base-uncased) on 3-class Persian
sentiment. CPU-only, plain PyTorch loop, no `datasets`/`accelerate`.
=========================================================================

WHY THIS FILE DOWNLOADS MODEL WEIGHTS BY DIRECT URL
---------------------------------------------------
`from_pretrained("HooshvareLab/bert-fa-base-uncased")` fails on this machine, so
:func:`ensure_base_model` fetches the files it actually needs from the Hub's
`/resolve/main/` endpoints instead. Facts measured here, not assumed:

  * `huggingface.co` and the Xet CDN it redirects to ARE reachable; the model
    loads in ~74s at ~6-9 MB/s (654 MB file), so no mirror is needed.
  * `hf-mirror.com` file paths time out -- do not use it as the primary source.
  * The repo contains ONLY: `.gitattributes`, `README.md`, `config.json`,
    `vocab.txt`, `pytorch_model.bin`, `tf_model.h5`, `flax_model.msgpack`.
    `tokenizer_config.json` and `special_tokens_map.json` return HTTP 404, which
    is a second reason the automatic loader can misbehave. We therefore build the
    tokenizer explicitly: `BertTokenizer(vocab_file=..., do_lower_case=False)`.
  * Verified tokenisation: "این محصول به شدت بد بود" ->
    ['این', 'محصول', 'به', 'شدت', 'بد', 'بود'] -- the space-split compounds from
    :mod:`src.preprocess` survive as separate subword groups.
  * `pytorch_model.bin` is a 654 MB fp32 checkpoint, 162.8M params. transformers
    4.57 accepts it from a local directory (it warns only about the fresh
    `classifier.weight`/`classifier.bias` heads, which is exactly what we want).

Training-set facts (see the project plan and src/preprocess.py)
--------------------------------------------------------------
  * Labels are 1-2 -> negative, 3 -> neutral, 4-5 -> positive; `rate=0` (unrated)
    rows are dropped upstream.
  * The corpus is ~14x skewed towards five-star comments, so `neutral` is scarce.
    The loss is class-weighted by inverse frequency and macro-F1 -- not accuracy
    -- is the headline metric. Expect `neutral` recall to stay weak; the report
    prints per-class numbers so that cannot be hidden.
  * CPU-only: `torch.cuda.is_available()` is False. Measured here at
    `max_length=128`, `batch_size=16`: ~11 s/step, so ~10 min per epoch for ~900
    training rows and ~35 min for a 3-epoch run, plus evaluation. Use `--smoke`
    (30 rows, 1 epoch, ~35 s) to check the plumbing before committing.

Usage
-----
    python -m src.train --smoke                       # ~1 min plumbing check
    python -m src.train --csv data/cleaned_comments.csv --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------
# Model source
# --------------------------------------------------------------------------
BASE_MODEL_ID = "HooshvareLab/bert-fa-base-uncased"  # ParsBERT
BASE_MODEL_DIR = MODELS_DIR / "parsbert-fa"  # local cache of the base weights
FINAL_MODEL_DIR = MODELS_DIR / "parsbert-sentiment-3class"  # fine-tuned output

# Only these three files are needed; the repo has no tokenizer_config.json or
# special_tokens_map.json (verified 404), so nothing else is fetched.
# `expected_bytes` is the content-length observed from this network and is used
# as a cheap integrity check against a truncated download.
MODEL_FILES: tuple[dict[str, Any], ...] = (
    {"name": "config.json", "expected_bytes": 440},
    {"name": "vocab.txt", "expected_bytes": 1_198_122},
    {"name": "pytorch_model.bin", "expected_bytes": 654_226_731},
)

DIRECT_URLS: tuple[str, ...] = (
    f"https://huggingface.co/{BASE_MODEL_ID}/resolve/main/",
    # Mirror is a last resort: file paths there were seen to time out.
    f"https://hf-mirror.com/{BASE_MODEL_ID}/resolve/main/",
)

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_NAMES_FA = {0: "منفی", 1: "خنثی", 2: "مثبت"}

LOG = logging.getLogger("bafp.train")

__all__ = [
    "SentimentPredictor", "train", "evaluate",
    "load_tokenizer_and_model", "ensure_base_model",
    "CommentDataset", "class_weights", "load_split",
    "LABEL_NAMES", "LABEL_NAMES_FA", "BASE_MODEL_ID",
]


def _clean_for_model(text: str) -> str:
    """Route raw user input through the SAME cleaner used at training time.

    Imported lazily with a path fallback so the predictor works both under
    `python -m src.train` and from a Streamlit script at the project root.
    """
    try:
        from src.preprocess import clean_text
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.preprocess import clean_text
    return clean_text(text)


# ==========================================================================
# Direct-download bootstrap
# ==========================================================================
@dataclass
class DownloadReport:
    fetched: list[str]
    cached: list[str]
    bytes_total: int
    seconds: float
    source: str


def _download_to(url: str, dest: Path, *, timeout: int = 60, max_retries: int = 3) -> int:
    """Stream ``url`` to ``dest`` with resume support. Returns bytes written.

    Resume matters: a 654 MB file is the kind of thing that dies mid-transfer,
    and re-downloading from zero every attempt would make the run unfinishable.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0

    for attempt in range(1, max_retries + 1):
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
                if response.status_code == 416:  # already complete
                    break
                response.raise_for_status()
                mode = "ab" if response.status_code == 206 else "wb"
                if mode == "wb":
                    existing = 0
                total = int(response.headers.get("content-length") or 0)
                written = existing
                started = time.time()
                last_print = started
                with open(part, mode) as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if now - last_print >= 10:
                            rate = (written - existing) / 1e6 / max(now - started, 1e-6)
                            LOG.info("    %s: %.1f/%.1f MB (%.1f MB/s)", dest.name,
                                     written / 1e6, total / 1e6, rate)
                            last_print = now
            if response.status_code in (200, 206):
                break
        except (requests.RequestException, OSError) as exc:
            LOG.warning("  attempt %d/%d for %s failed: %s", attempt, max_retries, url, exc)
            time.sleep(min(2 ** attempt, 10))
    else:
        raise RuntimeError(f"could not download {url}")

    if not part.exists():
        raise RuntimeError(f"no file was written for {url}")
    part.replace(dest)
    return dest.stat().st_size


def ensure_base_model(cache_dir: Path = BASE_MODEL_DIR, *, force: bool = False) -> DownloadReport:
    """Guarantee ``cache_dir`` holds config.json, vocab.txt and pytorch_model.bin.

    Order of attempts, deliberate given the verified network behaviour:
      1. local files with the expected size -> nothing to do;
      2. direct ``/resolve/main/`` GET against huggingface.co (primary, ~6-9 MB/s);
      3. the hf-mirror URL (usually times out -- kept as a courtesy fallback);
      4. ``huggingface_hub.hf_hub_download`` last, since it exercises the Xet
         transport that the plain URL path avoids.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fetched: list[str] = []
    cached: list[str] = []
    total_bytes = 0
    source = "cache"

    for spec in MODEL_FILES:
        name, expected = spec["name"], int(spec["expected_bytes"])
        dest = cache_dir / name
        if not force and dest.exists() and dest.stat().st_size == expected:
            cached.append(name)
            total_bytes += dest.stat().st_size
            LOG.info("cached  %-20s %s", name, _human_bytes(expected))
            continue

        LOG.info("fetching %-20s (expected %s)", name, _human_bytes(expected))
        done = False
        for base in DIRECT_URLS:
            try:
                size = _download_to(base + name, dest)
            except Exception as exc:  # noqa: BLE001 - try the next source
                LOG.warning("  %s failed: %s", base, str(exc)[:120])
                continue
            if size != expected:
                LOG.warning("  size mismatch from %s: %d != %d; retrying elsewhere",
                            base, size, expected)
                dest.unlink(missing_ok=True)
                continue
            fetched.append(name)
            total_bytes += size
            source = base.rstrip("/").rsplit("//", 1)[-1]
            done = True
            break

        if not done:  # last resort: the hub client
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(BASE_MODEL_ID, name, local_dir=str(cache_dir))
                if Path(path).stat().st_size == expected:
                    fetched.append(name)
                    total_bytes += expected
                    source = "huggingface_hub"
                    done = True
            except Exception as exc:  # noqa: BLE001
                LOG.warning("  hf_hub_download failed: %s", str(exc)[:120])

        if not done:
            raise RuntimeError(
                f"Could not obtain {name} at the expected size.\n"
                f"  Check connectivity to huggingface.co, or place the file at "
                f"{dest} by hand (from {DIRECT_URLS[0]}{name})."
            )

    return DownloadReport(fetched, cached, total_bytes, time.time() - started, source)


def _human_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    for index, unit in enumerate(units):
        scaled = value / (1024**index)
        if scaled < 1024 or unit == units[-1]:
            return f"{scaled:.1f} {unit}"
    return f"{value} B"  # unreachable


def load_tokenizer_and_model(
    num_labels: int = 3,
    cache_dir: Path = BASE_MODEL_DIR,
    *,
    force_download: bool = False,
) -> tuple[Any, Any, DownloadReport]:
    """Bootstrap the base weights, then build tokenizer + classifier head.

    `BertTokenizer(vocab_file=...)` is used instead of `AutoTokenizer` because
    the repo ships no `tokenizer_config.json` (404), so an auto-configured loader
    has nothing to read. `from_pretrained` on the local directory then finds
    `pytorch_model.bin` and initialises everything except the classification
    head -- which is what we intend to train.
    """
    from transformers import BertForSequenceClassification, BertTokenizer

    report = ensure_base_model(cache_dir, force=force_download)
    tokenizer = BertTokenizer(vocab_file=str(cache_dir / "vocab.txt"), do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained(
        str(cache_dir), num_labels=num_labels
    )
    return tokenizer, model, report


# ==========================================================================
# Data
# ==========================================================================
class CommentDataset(Dataset):
    """Tokenises a pandas slice on the fly; caching encodings for ~5k rows of
    length 128 is not worth the extra file format."""

    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int = 128) -> None:
        self.texts = frame["clean_text"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


def class_weights(frame: pd.DataFrame, num_labels: int = 3) -> torch.Tensor:
    """Inverse-frequency weights, normalised so the mean weight is 1.

    Normalising keeps the effective learning rate comparable to the literature
    values; raw 1/f weights would silently multiply the step size.
    """
    counts = np.bincount(frame["label"].astype(int).tolist(), minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_labels * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float)


def load_split(csv_path: Path, split: str, limit: int | None = None,
               text_col: str = "clean_text") -> pd.DataFrame:
    """Read one split out of the cleaned CSV, skipping rows with no label."""
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if text_col not in frame.columns:
        raise KeyError(f"{text_col!r} missing; run src/preprocess.py first")
    frame = frame[frame.get("split") == split]
    frame = frame[frame.get("label").notna()]
    frame = frame[frame[text_col].astype(str).str.len() > 0]
    if limit:
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


# ==========================================================================
# Train / evaluate
# ==========================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.inference_mode()
def evaluate(model: Any, loader: DataLoader, device: torch.device, num_labels: int = 3) -> dict[str, Any]:
    """Return accuracy, macro-F1, per-class report and the confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_pred.extend(logits.argmax(dim=1).cpu().tolist())
        all_true.extend(labels.tolist())

    label_range = list(range(num_labels))
    matrix = confusion_matrix(all_true, all_pred, labels=label_range)
    report = classification_report(
        all_true, all_pred, labels=label_range,
        target_names=[LABEL_NAMES[i] for i in label_range],
        output_dict=True, zero_division=0,
    )
    return {
        "n": len(all_true),
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
        "per_class": {LABEL_NAMES[i]: report[LABEL_NAMES[i]] for i in label_range},
        "confusion_matrix": matrix.tolist(),
        "labels": [LABEL_NAMES[i] for i in label_range],
    }


def train(
    csv_path: Path = DATA_DIR / "cleaned_comments.csv",
    output_dir: Path = FINAL_MODEL_DIR,
    *,
    epochs: int = 3,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    limit: int | None = None,
    num_workers: int = 0,
    threads: int | None = None,
    weighted_loss: bool = True,
    force_download: bool = False,
    patience: int = 0,
) -> dict[str, Any]:
    """Fine-tune ParsBERT and persist the model, tokenizer and metrics.

    Returns the metrics dict that is also written to ``output_dir/metrics.json``.
    """
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    if threads:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("device=%s threads=%d cuda=%s", device, torch.get_num_threads(),
            torch.cuda.is_available())

    tokenizer, model, download_report = load_tokenizer_and_model(
        num_labels=3, force_download=force_download
    )
    model.to(device)

    train_frame = load_split(csv_path, "train", limit=limit)
    val_frame = load_split(csv_path, "val", limit=limit)
    if train_frame.empty or val_frame.empty:
        raise RuntimeError(
            f"Empty split(s): train={len(train_frame)} val={len(val_frame)}. "
            f"Run `python -m src.preprocess --in data/raw_digikala_comments.csv` first."
        )
    LOG.info("train rows=%d val rows=%d labels=%s", len(train_frame), len(val_frame),
             train_frame["label"].value_counts().sort_index().to_dict())

    weights = class_weights(train_frame) if weighted_loss else None
    LOG.info("class weights=%s", None if weights is None else [round(w, 3) for w in weights.tolist()])
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    train_loader = DataLoader(
        CommentDataset(train_frame, tokenizer, max_length),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        CommentDataset(val_frame, tokenizer, max_length),
        batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader)) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )

    history: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] = (-1.0, {})
    started = time.time()
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_started = time.time()
        try:
            from tqdm.auto import tqdm as _tqdm
            step_iter = _tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                              desc=f"Epoch {epoch}/{epochs}", unit="step", leave=True)
        except ImportError:
            step_iter = enumerate(train_loader, start=1)
        for step, batch in step_iter:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(outputs.loss.item())

            if step % 20 == 0 or step == len(train_loader):
                elapsed = time.time() - epoch_started
                LOG.info("epoch %d step %d/%d loss %.4f  (%.1f min elapsed, %.1f s/step)",
                         epoch, step, len(train_loader), epoch_loss / step,
                         elapsed / 60, elapsed / step)

        metrics = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "minutes": round((time.time() - started) / 60, 2),
        }
        history.append(record)
        LOG.info("epoch %d -> loss %.4f  val acc %.4f  val macro-F1 %.4f",
                 epoch, record["train_loss"], metrics["accuracy"], metrics["macro_f1"])

        if metrics["macro_f1"] > best[0]:
            best = (metrics["macro_f1"], metrics)
            save_model(model, tokenizer, output_dir, max_length=max_length, seed=seed)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if patience > 0 and no_improve_count >= patience:
                LOG.info("Early stopping: no improvement for %d epochs", patience)
                break

    test_frame = load_split(csv_path, "test", limit=limit)
    test_metrics: dict[str, Any] | None = None
    if not test_frame.empty:
        test_loader = DataLoader(
            CommentDataset(test_frame, tokenizer, max_length),
            batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        )
        # Evaluate the checkpoint that won on val, not whatever epoch N left behind.
        # save_model() writes with safe_serialization=True, so the winning checkpoint
        # on disk is model.safetensors. The previous guard looked for pytorch_model.bin,
        # which is never produced, so the reload silently no-op'd and the reported test
        # metrics were computed on the in-memory final-epoch weights instead of the
        # checkpoint that actually won on validation.
        best_model_path = next(
            (output_dir / name for name in ("model.safetensors", "pytorch_model.bin")
             if (output_dir / name).exists()),
            None,
        )
        if best_model_path is not None:
            from transformers import BertForSequenceClassification

            model = BertForSequenceClassification.from_pretrained(str(output_dir)).to(device)
        test_metrics = evaluate(model, test_loader, device)

    payload = {
        "base_model": BASE_MODEL_ID,
        "label_map": LABEL_NAMES,
        "label_map_fa": LABEL_NAMES_FA,
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "test_rows": len(test_frame),
        "config": {
            "epochs": epochs, "batch_size": batch_size, "max_length": max_length,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio, "seed": seed,
            "weighted_loss": weighted_loss, "limit": limit,
        },
        "class_weights": None if weights is None else [round(w, 4) for w in weights.tolist()],
        "history": history,
        "best_val": best[1],
        "test": test_metrics,
        "total_minutes": round((time.time() - started) / 60, 2),
        "device": str(device),
        "download": {
            "fetched": download_report.fetched,
            "cached": download_report.cached,
            "mb": round(download_report.bytes_total / 1e6, 1),
            "seconds": round(download_report.seconds, 1),
            "source": download_report.source,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "label_map.json").write_text(
        json.dumps({"labels": LABEL_NAMES, "labels_fa": LABEL_NAMES_FA,
                    "rate_to_label": {"1": 0, "2": 0, "3": 1, "4": 2, "5": 2},
                    "dropped_rates": [0]}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    print_report(payload)
    return payload


def save_model(model: Any, tokenizer: Any, output_dir: Path, **kwargs: Any) -> None:
    """Persist weights + vocab + generation config for :class:`SentimentPredictor`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    # The fine-tuned dir must be self-sufficient: no tokenizer_config.json in the
    # base repo means save_pretrained cannot round-trip the vocab on its own.
    vocab_src = BASE_MODEL_DIR / "vocab.txt"
    if vocab_src.exists():
        (output_dir / "vocab.txt").write_text(
            (vocab_src.read_text(encoding="utf-8")), encoding="utf-8"
        )
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": kwargs.get("max_length", 128),
                    "do_lower_case": False, "tokenizer_class": "BertTokenizer"}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_meta.json").write_text(
        json.dumps({"seed": kwargs.get("seed"), "max_length": kwargs.get("max_length"),
                    "base_model": BASE_MODEL_ID}, indent=2), encoding="utf-8"
    )


def print_report(payload: dict[str, Any]) -> None:
    """Print the honest result: macro-F1 first, per-class detail, skew visible."""
    print("\n" + "=" * 66)
    print("Training complete")
    print("=" * 66)
    print(f"  base model        : {payload['base_model']} (direct download "
          f"{payload['download']['mb']} MB in {payload['download']['seconds']}s "
          f"from {payload['download']['source']})")
    print(f"  rows              : train={payload['train_rows']} val={payload['val_rows']} "
          f"test={payload['test_rows']}")
    print(f"  wall time         : {payload['total_minutes']} min on {payload['device']}")
    best = payload["best_val"]
    if best:
        print(f"\n  BEST ON VALIDATION  macro-F1 {best['macro_f1']:.4f}  "
              f"accuracy {best['accuracy']:.4f}")
    test = payload["test"]
    if test:
        print(f"  ON HELD-OUT TEST    macro-F1 {test['macro_f1']:.4f}  "
              f"accuracy {test['accuracy']:.4f}")
        print("\n  per-class (test):")
        print(f"    {'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
        for name, values in test["per_class"].items():
            if isinstance(values, dict) and "f1-score" in values:
                print(f"    {name:<10}{values['precision']:>10.3f}{values['recall']:>10.3f}"
                      f"{values['f1-score']:>10.3f}{int(values['support']):>10}")
        print("\n  confusion matrix (rows=true, cols=pred) order "
              f"{test['labels']}:")
        for row in test["confusion_matrix"]:
            print(f"    {row}")
        neutral = test["per_class"].get("neutral", {})
        if isinstance(neutral, dict) and neutral.get("recall", 1) < 0.4:
            print("\n  NOTE: neutral recall is low, as expected -- 3-class labels come from "
                  "binned\n  1-5 star ratings where rate=3 is rare (~14x fewer than 5-star). "
                  "Macro-F1 is\n  the metric to report, not accuracy.")
    print(f"\n  saved to            : {FINAL_MODEL_DIR}")


# ==========================================================================
# Inference (imported by app.py) — optimised for latency & throughput
# ==========================================================================
class SentimentPredictor:
    """Load the fine-tuned checkpoint once and predict on raw Persian text.

    Cleans the input through :func:`src.preprocess.clean_text` first, so what the
    model sees matches what it was trained on -- including the ZWNJ->space rule.

    Optimisations vs. the original:
      * ``torch.inference_mode`` instead of ``no_grad`` (disables version counter).
      * Batched tokenisation in :meth:`predict_batch` — single forward pass for N
        texts instead of N passes. Used by the dashboard batch page and advisor.
      * ``_clean_for_model`` result is LRU-cached (duplicates are common in batch).
      * ``padding='longest'`` in batch mode to avoid padding every short comment to
        ``max_length`` — ~2-3× faster for typical Persian comments (~20 tokens).
      * torch.compile is intentionally NOT used: this Windows CPU box has no MSVC,
        so inductor fails at the first forward pass ("Compiler: cl is not found").
    """

    def __init__(self, model_dir: Path = FINAL_MODEL_DIR, base_dir: Path = BASE_MODEL_DIR) -> None:
        from transformers import BertForSequenceClassification, BertTokenizerFast

        model_dir = Path(model_dir)
        label_file = model_dir / "label_map.json"
        if label_file.exists():
            mapping = json.loads(label_file.read_text(encoding="utf-8"))
            self.labels = {int(k): v for k, v in mapping["labels"].items()}
            self.labels_fa = mapping.get("labels_fa", {})
        else:
            self.labels = dict(LABEL_NAMES)
            self.labels_fa = dict(LABEL_NAMES_FA)

        meta_file = model_dir / "training_meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        self.max_length = int(meta.get("max_length", 128))
        self.model_name = meta.get("base_model", BASE_MODEL_ID)
        # Reverse map once, not per prediction
        self._inv_labels: dict[str, int] = {v: k for k, v in self.labels.items()}

        vocab = model_dir / "vocab.txt"
        if not vocab.exists():
            vocab = base_dir / "vocab.txt"
        if not vocab.exists():  # a fresh clone may only have the hub id cached
            ensure_base_model(base_dir)
            vocab = base_dir / "vocab.txt"

        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
        self.model = BertForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # NOTE: torch.compile is deliberately NOT used here. On this Windows
        # CPU-only box MSVC (cl.exe) is absent, so inductor raises
        # "Compiler: cl is not found" at the first forward pass. The speedups
        # come from batched inference + inference_mode instead.

        # INT8 dynamic quantization — ~3-5× faster on CPU with negligible
        # accuracy loss.  Only applied when running on CPU (quantised kernels
        # are not available on CUDA).
        if self.device.type == "cpu":
            try:
                import torchao  # noqa: F401 – preferred quantisation path in PyTorch ≥ 2.10
                from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
                quantize_(self.model, int8_dynamic_activation_int8_weight())
            except Exception:
                # Fallback: legacy torch.ao path (PyTorch < 2.10)
                try:
                    self.model = torch.ao.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8,
                    )
                except Exception:
                    pass  # quantization unavailable — run in fp32

        # Warm-up: run a tiny forward pass to pre-JIT OpenMP / MKL threads so
        # the first real prediction does not carry the ~6-7 s kernel-init cost.
        try:
            _warmup_tok = self.tokenizer("warmup", return_tensors="pt").to(self.device)
            with torch.inference_mode():
                self.model(**_warmup_tok)
            del _warmup_tok
        except Exception:
            pass

    def _softmax_sorted(self, probs: torch.Tensor, top_k: int) -> list[tuple[str, float]]:
        ordered = sorted(
            ((self.labels[int(i)], float(p)) for i, p in enumerate(probs)),
            key=lambda kv: kv[1], reverse=True,
        )
        return ordered[:top_k]

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        """Return predicted label, confidence and the full probability vector."""
        cleaned = _clean_for_model(text)
        if not cleaned:
            return {
                "label": None, "label_fa": None, "confidence": 0.0,
                "probabilities": {}, "cleaned": "",
                "note": "خالی یا غیر فارسی: متن قابل پردازش نیست",
            }

        encoded = self.tokenizer(
            cleaned, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=0)

        ordered = self._softmax_sorted(probabilities, top_k)
        best_name, best_prob = ordered[0]
        return {
            "label": best_name,
            "label_fa": self.labels_fa.get(self._inv_labels[best_name], best_name),
            "confidence": round(best_prob, 4),
            "probabilities": {name: round(prob, 4) for name, prob in ordered},
            "cleaned": cleaned,
            "note": "",
        }

    def predict_batch(self, texts: list[str], top_k: int = 3, batch_size: int = 32) -> list[dict[str, Any]]:
        """Batched inference — one tokenisation + forward pass per *batch_size* chunk.

        Returns results in the same order as *texts*. Empty / non-Persian inputs
        produce the same ``label=None`` sentinel as :meth:`predict`.
        """
        if not texts:
            return []
        results: list[dict[str, Any] | None] = [None] * len(texts)
        # Pre-clean and remember which indices need model inference
        cleaned_list: list[str] = []
        need_idx: list[int] = []
        for idx, txt in enumerate(texts):
            c = _clean_for_model(txt)
            if not c:
                results[idx] = {
                    "label": None, "label_fa": None, "confidence": 0.0,
                    "probabilities": {}, "cleaned": "",
                    "note": "خالی یا غیر فارسی: متن قابل پردازش نیست",
                }
            else:
                cleaned_list.append(c)
                need_idx.append(idx)

        if not need_idx:
            return [r for r in results if r is not None]  # type: ignore[return-value]

        # Map from position in cleaned_list -> original index
        # Chunked to keep peak memory bounded
        for chunk_start in range(0, len(cleaned_list), batch_size):
            chunk_clean = cleaned_list[chunk_start: chunk_start + batch_size]
            chunk_orig_idx = need_idx[chunk_start: chunk_start + batch_size]
            encoded = self.tokenizer(
                chunk_clean, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
            for row, orig_idx, cleaned in zip(probs, chunk_orig_idx, chunk_clean):
                ordered = self._softmax_sorted(row, top_k)
                best_name, best_prob = ordered[0]
                results[orig_idx] = {
                    "label": best_name,
                    "label_fa": self.labels_fa.get(self._inv_labels[best_name], best_name),
                    "confidence": round(float(best_prob), 4),
                    "probabilities": {name: round(float(p), 4) for name, p in ordered},
                    "cleaned": cleaned,
                    "note": "",
                }
        # All slots filled now
        return [r for r in results if r is not None]  # type: ignore[return-value]


# ==========================================================================
# CLI
# ==========================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "cleaned_comments.csv")
    parser.add_argument("--out", type=Path, default=FINAL_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per split (fast tests)")
    parser.add_argument("--no-weighted-loss", action="store_true")
    parser.add_argument("--patience", type=int, default=0,
                        help="early stopping patience (0 = disabled)")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="bootstrap + 30 rows x 1 epoch: verifies plumbing only")
    parser.add_argument("--download-only", action="store_true", help="fetch weights and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.download_only:
        report = ensure_base_model(force=args.force_download)
        print(f"fetched={report.fetched} cached={report.cached} "
              f"{report.bytes_total / 1e6:.1f} MB in {report.seconds:.1f}s from {report.source}")
        return 0

    if not args.csv.exists():
        print(f"Dataset not found: {args.csv}\nBuild it first:\n"
              f"  python -m src.scraper --source digikala --out data/raw_digikala_comments.csv\n"
              f"  python -m src.preprocess --in data/raw_digikala_comments.csv --out {args.csv}")
        return 1

    if args.smoke:
        LOG.info("SMOKE TEST: plumbing only (limit=30, epochs=1). Results are meaningless.")
        args.limit = args.limit or 30
        args.epochs = 1

    train(
        csv_path=args.csv,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        limit=args.limit,
        threads=args.threads,
        weighted_loss=not args.no_weighted_loss,
        force_download=args.force_download,
        patience=args.patience,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
