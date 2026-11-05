"""
Re-evaluate the shipped fine-tuned checkpoints against their real held-out
splits and rewrite metrics.json.
==========================================================================

WHY THIS EXISTS
---------------
The base checkpoint's metrics.json was overwritten by a `--smoke` run
(30 rows, 1 epoch, "results are meaningless" per its own docstring) on
2026-09-28, and the training runs that produced the CURRENT weights were
interrupted before their end-of-run metrics write. So the weights on disk are
real fine-tuned weights while the metrics beside them describe a toy run —
and app.py displays metrics.json on the Live Prediction chips and the model
evaluation page. The wide checkpoint never got a metrics.json at all.

This script measures the DEPLOYED weights on the same test/val splits the
training pipeline uses (evaluate() from src.train, fp32, batch 32, the
checkpoint's own vocab) and writes an honest metrics.json with provenance
notes. It is evaluation-only: no training, no weight changes.

The old base metrics.json is preserved as metrics.json.smoke-backup.

Usage (from the project root, venv python):
    python -X utf8 rebuild_metrics.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import (  # noqa: E402
    LABEL_NAMES,
    LABEL_NAMES_FA,
    CommentDataset,
    class_weights,
    evaluate,
    load_split,
)

# (checkpoint directory, corpus CSV its training split lives in)
JOBS: tuple[tuple[str, str], ...] = (
    ("parsbert-sentiment-3class", "data/cleaned_comments.csv"),
    ("parsbert-sentiment-3class-wide", "data/cleaned_comments_wide.csv"),
)


def rebuild(model_name: str, csv_path: str) -> None:
    from transformers import BertForSequenceClassification, BertTokenizerFast

    model_dir = PROJECT_ROOT / "models" / model_name
    csv = PROJECT_ROOT / csv_path
    started = time.time()
    meta = json.loads((model_dir / "training_meta.json").read_text(encoding="utf-8"))
    max_length = int(meta.get("max_length", 128))

    tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=False)
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  model loaded on {device} in {time.time() - started:.0f}s", flush=True)

    train_frame = load_split(csv, "train")
    val_frame = load_split(csv, "val")
    test_frame = load_split(csv, "test")
    weights = class_weights(train_frame)

    scored: dict[str, dict] = {}
    for split_name, frame in (("val", val_frame), ("test", test_frame)):
        loader = DataLoader(
            CommentDataset(frame, tokenizer, max_length), batch_size=32, shuffle=False
        )
        t0 = time.time()
        scored[split_name] = evaluate(model, loader, device)
        print(
            f"  {split_name}: {len(frame)} rows in {time.time() - t0:.0f}s -> "
            f"acc {scored[split_name]['accuracy']:.4f}  macro-F1 {scored[split_name]['macro_f1']:.4f}",
            flush=True,
        )

    payload = {
        "base_model": meta.get("base_model"),
        "label_map": LABEL_NAMES,
        "label_map_fa": LABEL_NAMES_FA,
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "test_rows": len(test_frame),
        "config": {
            "reconstructed": True,
            "max_length": max_length,
            "seed": meta.get("seed"),
            "weighted_loss": True,
            "note": (
                "Metrics re-measured from the shipped model.safetensors by "
                "rebuild_metrics.py on 2026-11-05. The original per-epoch history "
                "was lost with the interrupted training run, so history is empty."
            ),
        },
        "class_weights": [round(w, 4) for w in weights.tolist()],
        "history": [],
        "best_val": scored["val"],
        "test": scored["test"],
        "total_minutes": round((time.time() - started) / 60, 2),
        "device": str(device),
        "download": {
            "fetched": [],
            "cached": [],
            "mb": 0.0,
            "seconds": 0.0,
            "source": "n/a (weights already on disk)",
        },
        "provenance": (
            "rebuild_metrics.py evaluation-only reconstruction. The previous "
            "metrics.json was written by a --smoke run (30 rows, 1 epoch) that "
            "overwrote the original full-run metrics on 2026-09-28; the later "
            "full run saved its best checkpoint but was interrupted before the "
            "end-of-run metrics write."
        ),
    }

    backup = model_dir / "metrics.json.smoke-backup"
    if (model_dir / "metrics.json").exists() and not backup.exists():
        shutil.copy2(model_dir / "metrics.json", backup)
        print(f"  old metrics preserved -> {backup.name}")

    (model_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  wrote {model_dir / 'metrics.json'}")

    test = scored["test"]
    print("  per-class (test):")
    for name, vals in test["per_class"].items():
        print(
            f"    {name:<9} P {vals['precision']:.3f}  R {vals['recall']:.3f}  "
            f"F1 {vals['f1-score']:.3f}  (n={int(vals['support'])})"
        )
    print(f"  confusion matrix {test['labels']}: {test['confusion_matrix']}")
    print(f"  wall time: {(time.time() - started) / 60:.1f} min", flush=True)


def main() -> int:
    for model_name, csv_path in JOBS:
        print(f"=== {model_name}  (scored against {csv_path}) ===", flush=True)
        rebuild(model_name, csv_path)
        print(flush=True)
    print("Done. Restart the Streamlit server (or hard-reload) so load_metrics picks up the new files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
