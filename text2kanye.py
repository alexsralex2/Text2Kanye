from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import onnxruntime as ort


# ---------------- Tuned defaults for your use-case ----------------

SENTENCE_FILE = "sentencelist.txt"

CACHE_DIR = Path(".cache_fast_match_win11")
CACHE_DIR.mkdir(exist_ok=True)

EMB_NPY = CACHE_DIR / "sentencelist.embeddings.float32.npy"
META_JSON = CACHE_DIR / "sentencelist.meta.json"
FAISS_INDEX = CACHE_DIR / "sentencelist.faiss.index"
ONNX_DIR = CACHE_DIR / "onnx_model"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Latency-focused: shorter sequence length helps a lot.
DEFAULT_MAX_LENGTH = 64

# Building embeddings: big batch (RAM/space is fine; build once).
BUILD_BATCH_SIZE = 384

# Retrieval defaults
DEFAULT_TOP_K = 1
DEFAULT_RETRIEVE_K = 1

# Similarity thresholds
DEFAULT_MIN_SCORE = 0.45
DEFAULT_AMBIGUITY_MARGIN = 0.05

# Provider benchmark settings (only used for --provider auto)
PROVIDER_BENCH_WARMUP = 2
PROVIDER_BENCH_RUNS = 6

# -----------------------------------------------------------------


def read_sentences(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        sents = [line.strip() for line in f if line.strip()]
    if not sents:
        raise ValueError(f"No sentences found in {path}")
    return sents


def fingerprint_sentences(sentences: List[str]) -> str:
    joined = "\n".join(sentences).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def make_session_options_for_latency() -> ort.SessionOptions:
    """
    Speed-first ONNX Runtime settings, tuned for interactive latency.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    cpu_count = os.cpu_count() or 6
    so.intra_op_num_threads = max(1, cpu_count)
    so.inter_op_num_threads = 1

    return so


def ensure_onnx_export(model_id: str, onnx_dir: Path) -> None:
    onnx_dir.mkdir(exist_ok=True)
    if any(onnx_dir.iterdir()):
        return

    print(f"[ONNX] Exporting model to {onnx_dir} (one-time)...")
    so = make_session_options_for_latency()
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider",
        session_options=so,
    )
    model.save_pretrained(onnx_dir)
    print("[ONNX] Export complete.")


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)
    mask_exp = mask[:, :, None]
    summed = (last_hidden_state * mask_exp).sum(axis=1)
    counts = np.clip(mask.sum(axis=1, keepdims=True), 1e-9, None)
    return summed / counts


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


def embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    ort_model: ORTModelForFeatureExtraction,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """
    ONNX -> last_hidden_state -> mean pool -> L2 normalize.
    Returns float32 contiguous (N, D).
    """
    out_chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        out = ort_model(**tok)
        last_hidden = out.last_hidden_state  # (B, T, H)
        pooled = mean_pool(last_hidden, tok["attention_mask"])
        pooled = l2_normalize(pooled)
        out_chunks.append(pooled.astype(np.float32, copy=False))

    embs = np.vstack(out_chunks)
    return np.asarray(embs, dtype=np.float32, order="C")


def load_or_build_embeddings(
    sentences: List[str],
    tokenizer: AutoTokenizer,
    ort_model: ORTModelForFeatureExtraction,
    max_length: int,
) -> np.ndarray:
    fp = fingerprint_sentences(sentences)
    if EMB_NPY.exists() and META_JSON.exists():
        try:
            meta = json.loads(META_JSON.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fp and meta.get("count") == len(sentences) and meta.get("max_length") == max_length:
                embs = np.load(EMB_NPY, mmap_mode=None)
                return np.asarray(embs, dtype=np.float32, order="C")
        except Exception:
            pass

    print("[EMB] Building embeddings (cache miss/invalid)...")
    t0 = time.perf_counter()
    embs = embed_texts(
        sentences,
        tokenizer=tokenizer,
        ort_model=ort_model,
        batch_size=BUILD_BATCH_SIZE,
        max_length=max_length,
    )
    np.save(EMB_NPY, embs)
    META_JSON.write_text(
        json.dumps(
            {
                "fingerprint": fp,
                "count": len(sentences),
                "dim": int(embs.shape[1]),
                "max_length": max_length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    t1 = time.perf_counter()
    print(f"[EMB] Built {len(sentences)} embeddings in {t1 - t0:.3f}s")
    return embs


def try_load_or_build_faiss(embs: np.ndarray):
    """
    Optional fastest retrieval. If faiss isn't available, fall back to NumPy.
    """
    try:
        import faiss  # type: ignore

        d = embs.shape[1]
        if FAISS_INDEX.exists():
            index = faiss.read_index(str(FAISS_INDEX))
            return index, True

        index = faiss.IndexFlatIP(d)
        index.add(embs)
        faiss.write_index(index, str(FAISS_INDEX))
        return index, True
    except Exception:
        return None, False


def numpy_topk_ip(embs: np.ndarray, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = embs @ q
    k = min(k, sims.shape[0])
    idx_part = np.argpartition(sims, -k)[-k:]
    idx_sorted = idx_part[np.argsort(sims[idx_part])[::-1]]
    return sims[idx_sorted], idx_sorted


def normalize_provider_arg(arg: str) -> str:
    a = arg.strip().lower()
    if a in ("cpu", "cpuexecutionprovider"):
        return "CPUExecutionProvider"
    if a in ("dml", "directml", "dmlexecutionprovider"):
        return "DmlExecutionProvider"
    if a in ("auto",):
        return "auto"
    # Allow passing raw provider name (case-sensitive typically, but we'll accept)
    return arg


def benchmark_provider(
    provider: str,
    onnx_dir: Path,
    tokenizer: AutoTokenizer,
    max_length: int,
    debug: bool = False,
) -> float:
    """
    Measure average embed time for a tiny batch (single sentence) on a given provider.
    Lower is better.
    """
    so = make_session_options_for_latency()
    model = ORTModelForFeatureExtraction.from_pretrained(
        onnx_dir,
        provider=provider,
        session_options=so,
    )

    text = "quick benchmark sentence for provider selection"
    for _ in range(PROVIDER_BENCH_WARMUP):
        _ = embed_texts([text], tokenizer, model, batch_size=1, max_length=max_length)

    times = []
    for _ in range(PROVIDER_BENCH_RUNS):
        t0 = time.perf_counter()
        _ = embed_texts([text], tokenizer, model, batch_size=1, max_length=max_length)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = float(sum(times) / len(times))
    if debug:
        print(f"[BENCH] provider={provider} avg_embed_time={avg*1000:.2f}ms")
    return avg


def pick_provider_auto(
    tokenizer: AutoTokenizer,
    max_length: int,
    debug: bool,
) -> str:
    """
    Auto-select between DML and CPU (whichever is available and faster).
    """
    avail = ort.get_available_providers()
    candidates = []
    if "DmlExecutionProvider" in avail:
        candidates.append("DmlExecutionProvider")
    if "CPUExecutionProvider" in avail:
        candidates.append("CPUExecutionProvider")
    if not candidates:
        return avail[0] if avail else "CPUExecutionProvider"

    # Benchmark each candidate
    results = []
    for p in candidates:
        try:
            avg = benchmark_provider(p, ONNX_DIR, tokenizer, max_length, debug=debug)
            results.append((avg, p))
        except Exception as e:
            if debug:
                print(f"[BENCH] provider={p} failed: {e}")

    if not results:
        return candidates[0]
    results.sort()
    return results[0][1]


def load_ort_model(onnx_dir: Path, provider: str) -> ORTModelForFeatureExtraction:
    so = make_session_options_for_latency()
    return ORTModelForFeatureExtraction.from_pretrained(
        onnx_dir,
        provider=provider,
        session_options=so,
    )


def match_once(
    query: str,
    sentences: List[str],
    embs: np.ndarray,
    tokenizer: AutoTokenizer,
    ort_model: ORTModelForFeatureExtraction,
    max_length: int,
    retrieve_k: int,
    top_k: int,
    min_score: float,
    ambiguity_margin: float,
    faiss_index=None,
    using_faiss: bool = False,
) -> None:
    q = embed_texts([query], tokenizer, ort_model, batch_size=1, max_length=max_length)[0]
    k = max(retrieve_k, top_k)

    if using_faiss:
        scores, idx = faiss_index.search(q[None, :].astype(np.float32, copy=False), k)
        scores = scores[0]
        idx = idx[0]
    else:
        scores, idx = numpy_topk_ip(embs, q.astype(np.float32, copy=False), k)

    best_score = float(scores[0]) if len(scores) else -1.0
    second_score = float(scores[1]) if len(scores) > 1 else -1.0

    if best_score < min_score:
        status = "no_match"
    else:
        status = "ambiguous" if (best_score - second_score) < ambiguity_margin else "ok"

    print("\n=== STATUS ===")
    print(status)

    if status == "no_match":
        print("No sufficiently close match found.")
        return

    best_idx = int(idx[0])
    print("\n=== BEST MATCH ===")
    print(f"{sentences[best_idx]}\n(score: {best_score:.4f})")

    print(f"\n=== TOP {top_k} ===")
    for r in range(min(top_k, len(idx))):
        i = int(idx[r])
        s = float(scores[r])
        print(f"{r+1:>2}. {sentences[i]}  (score: {s:.4f})")

    if status == "ambiguous":
        print("\n⚠ Ambiguous: top results are close. Consider adding context or using top-k.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FAST semantic matcher (Windows 11 tuned; provider selectable; default CPU)."
    )
    ap.add_argument("query", nargs="*", help="Input sentence (omit for interactive loop).")
    ap.add_argument("--provider", default="cpu",
                    help="ONNX Runtime provider: cpu (default), dml, auto, or a raw provider name.")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--retrieve-k", type=int, default=DEFAULT_RETRIEVE_K)
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--ambiguity-margin", type=float, default=DEFAULT_AMBIGUITY_MARGIN)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--no-loop", action="store_true", help="Prompt once and exit (instead of interactive loop).")
    ap.add_argument("--debug", action="store_true", help="Print debug info (including provider benchmarks).")
    args = ap.parse_args()

    provider_arg = normalize_provider_arg(args.provider)

    sentences = read_sentences(SENTENCE_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # Ensure ONNX export exists (one-time)
    ensure_onnx_export(MODEL_ID, ONNX_DIR)

    if provider_arg == "auto":
        chosen = pick_provider_auto(tokenizer, args.max_length, debug=args.debug)
    else:
        chosen = provider_arg

    avail = ort.get_available_providers()
    if chosen not in avail:
        print(f"[WARN] Provider '{chosen}' not available. Available providers: {avail}", file=sys.stderr)
        print("[WARN] Falling back to CPUExecutionProvider.", file=sys.stderr)
        chosen = "CPUExecutionProvider"

    ort_model = load_ort_model(ONNX_DIR, chosen)
    print(f"[ONNX] Using provider: {chosen} (available: {avail})")

    embs = load_or_build_embeddings(sentences, tokenizer, ort_model, max_length=args.max_length)

    faiss_index, using_faiss = try_load_or_build_faiss(embs)
    print("[FAISS] Enabled (exact IP)." if using_faiss else "[FAISS] Not available; using NumPy (still fast at your scale).")

    # One-shot
    if args.query:
        query = " ".join(args.query).strip()
        if not query:
            print("No query provided.", file=sys.stderr)
            return 1
        print("\n=== INPUT ===")
        print(query)
        match_once(
            query=query,
            sentences=sentences,
            embs=embs,
            tokenizer=tokenizer,
            ort_model=ort_model,
            max_length=args.max_length,
            retrieve_k=args.retrieve_k,
            top_k=args.top_k,
            min_score=args.min_score,
            ambiguity_margin=args.ambiguity_margin,
            faiss_index=faiss_index,
            using_faiss=using_faiss,
        )
        return 0

    # Interactive loop (best for your usage pattern)
    if args.no_loop:
        query = input("Enter sentence: ").strip()
        if not query:
            return 0
        print("\n=== INPUT ===")
        print(query)
        match_once(
            query=query,
            sentences=sentences,
            embs=embs,
            tokenizer=tokenizer,
            ort_model=ort_model,
            max_length=args.max_length,
            retrieve_k=args.retrieve_k,
            top_k=args.top_k,
            min_score=args.min_score,
            ambiguity_margin=args.ambiguity_margin,
            faiss_index=faiss_index,
            using_faiss=using_faiss,
        )
        return 0

    print("\nInteractive mode. Type your query and press Enter.")
    print("Type /q to quit. Type /help for commands.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query in ("/q", "/quit", "/exit"):
            break
        if query == "/help":
            print("Commands:")
            print("  /q            quit")
            print("  /help         show commands")
            print("  /maxlen N     set max token length (e.g., /maxlen 96)")
            print("  /topk N       set printed top-k")
            print("  /retr N       set retrieval-k")
            continue
        if query.startswith("/maxlen "):
            try:
                args.max_length = int(query.split(None, 1)[1])
                print(f"max_length set to {args.max_length} (note: embeddings cache uses max_length; rebuild may be needed if you change this permanently).")
            except Exception:
                print("Usage: /maxlen 96")
            continue
        if query.startswith("/topk "):
            try:
                args.top_k = int(query.split(None, 1)[1])
                print(f"top_k set to {args.top_k}")
            except Exception:
                print("Usage: /topk 5")
            continue
        if query.startswith("/retr "):
            try:
                args.retrieve_k = int(query.split(None, 1)[1])
                print(f"retrieve_k set to {args.retrieve_k}")
            except Exception:
                print("Usage: /retr 20")
            continue

        print("\n=== INPUT ===")
        print(query)
        match_once(
            query=query,
            sentences=sentences,
            embs=embs,
            tokenizer=tokenizer,
            ort_model=ort_model,
            max_length=args.max_length,
            retrieve_k=args.retrieve_k,
            top_k=args.top_k,
            min_score=args.min_score,
            ambiguity_margin=args.ambiguity_margin,
            faiss_index=faiss_index,
            using_faiss=using_faiss,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())