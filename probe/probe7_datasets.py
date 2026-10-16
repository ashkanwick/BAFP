"""Probe SnappFood dataset schemas via hf-mirror. Determines final training data."""
import os, io, json, traceback
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

CANDIDATES = [
    "ParsiAI/snappfood-sentiment-analysis",
    "mohsen2/snappfood3",
    "intai2070/snappfood-refined-sentiment-dataset",
    "mteb/persian_food_sentiment",
    "asparius/Persian-Food-Sentiment",
    "sepidmnorozy/Persian_sentiment",
    "aictsharif/persian-sentiment-comments",
]

def list_files(ds):
    for rev in ("main", "refs/convert/parquet"):
        try:
            return rev, [s.rfilename for s in api.dataset_info(ds, revision=rev).siblings]
        except Exception:
            continue
    try:
        return "main", [s.rfilename for s in api.dataset_info(ds).siblings]
    except Exception as e:
        return None, f"ERR {type(e).__name__}: {str(e)[:100]}"

def peek(ds, files):
    """Download the smallest data file and print schema + label distribution."""
    import pandas as pd
    data_files = [f for f in files if f.lower().endswith((".csv", ".parquet", ".json", ".jsonl", ".tsv"))]
    if not data_files:
        return "  no data files"
    # prefer csv then parquet, and pinned-known filenames
    order = sorted(data_files, key=lambda f: (0 if f.endswith(".csv") else 1 if f.endswith(".parquet") else 2, len(f)))
    out = []
    for fname in order[:2]:
        try:
            path = hf_hub_download(ds, fname, repo_type="dataset")
            size = os.path.getsize(path) / 1e6
            out.append(f"  FILE {fname} ({size:.1f} MB)")
            if fname.endswith(".csv"):
                df = pd.read_csv(path)
            elif fname.endswith(".parquet"):
                df = pd.read_parquet(path)
            elif fname.endswith((".jsonl", ".json")):
                df = pd.read_json(path, lines=fname.endswith(".jsonl"))
            else:
                df = pd.read_csv(path, sep="\t")
            out.append(f"    shape={df.shape}")
            out.append(f"    columns={list(df.columns)}")
            for col in df.columns:
                if df[col].dtype == object or df[col].nunique() < 12:
                    vc = df[col].value_counts().head(6).to_dict()
                    if df[col].nunique() < 12:
                        out.append(f"    LABELS {col}: {vc}")
            for col in df.columns:
                if df[col].dtype == object:
                    out.append(f"    sample {col}[0]: {repr(str(df[col].iloc[0])[:110])}")
            break
        except Exception as e:
            out.append(f"  FILE {fname} FAILED {type(e).__name__}: {str(e)[:140]}")
    return "\n".join(out)

for ds in CANDIDATES:
    print("=" * 70)
    print(ds)
    rev, files = list_files(ds)
    if isinstance(files, str):
        print(" ", files)
        continue
    print(f"  revision={rev}  n_files={len(files)}")
    print(f"  files={files[:14]}")
    try:
        print(peek(ds, files))
    except Exception:
        traceback.print_exc()
