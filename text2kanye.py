from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------

SENTENCE_FILE = "sentencelist.txt"
EMB_FILE = "sentencelist.embeddings.npy"
META_FILE = "sentencelist.meta.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ----------------------------------------


def read_sentences(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def fingerprint(sentences: List[str]) -> str:
    joined = "\n".join(sentences).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def load_or_build_embeddings(
    sentences: List[str],
    model_name: str,
    emb_path: Path,
    meta_path: Path,
) -> np.ndarray:
    """
    Loads cached embeddings if valid, otherwise rebuilds them.
    """
    fp = fingerprint(sentences)

    if emb_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("model") == model_name
                and meta.get("fingerprint") == fp
                and meta.get("count") == len(sentences)
            ):
                return np.load(emb_path)
        except Exception:
            pass  # fall through to rebuild

    print("Building embeddings (cache miss or invalid)...")
    model = SentenceTransformer(model_name)
    embs = model.encode(sentences, normalize_embeddings=True)

    np.save(emb_path, embs)
    meta = {
        "model": model_name,
        "fingerprint": fp,
        "count": len(sentences),
        "dim": int(embs.shape[1]),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return embs


class Retriever:
    def __init__(self, sentences: List[str], embeddings: np.ndarray):
        self.sentences = sentences
        self.embs = embeddings

    def top_k(self, query_emb: np.ndarray, k: int) -> List[Tuple[int, float]]:
        sims = self.embs @ query_emb
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[int(i)])) for i in idx]


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        pairs = [(query, c) for c in candidates]
        scores = self.model.predict(pairs)
        order = np.argsort(-scores)
        return [(int(i), float(scores[int(i)])) for i in order]


def match(
    query: str,
    model: SentenceTransformer,
    retriever: Retriever,
    reranker: Optional[Reranker],
    retrieve_k: int,
    return_k: int,
    min_score: float,
    ambiguity_margin: float,
) -> Dict[str, Any]:
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    retrieved = retriever.top_k(query_emb, retrieve_k)

    if retrieved[0][1] < min_score:
        return {"status": "no_match", "best": None, "candidates": []}

    if reranker:
        idxs = [i for i, _ in retrieved]
        texts = [retriever.sentences[i] for i in idxs]
        reranked = reranker.rerank(query, texts)
        final = [(idxs[i], score) for i, score in reranked]
    else:
        final = retrieved

    best_idx, best_score = final[0]
    second_score = final[1][1] if len(final) > 1 else -1.0
    ambiguous = (best_score - second_score) < ambiguity_margin

    return {
        "status": "ambiguous" if ambiguous else "ok",
        "best": {
            "sentence": retriever.sentences[best_idx],
            "score": best_score,
        },
        "candidates": [
            {"sentence": retriever.sentences[i], "score": s}
            for i, s in final[:return_k]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="Input sentence")
    ap.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--retrieve-k", type=int, default=1)
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--ambiguity-margin", type=float, default=0.05)
    args = ap.parse_args()

    query = " ".join(args.query).strip() if args.query else input("Enter sentence: ").strip()
    if not query:
        print("No input sentence provided.", file=sys.stderr)
        return 1

    sentences = read_sentences(SENTENCE_FILE)
    emb_path = Path(EMB_FILE)
    meta_path = Path(META_FILE)

    embeddings = load_or_build_embeddings(
        sentences, EMBED_MODEL, emb_path, meta_path
    )

    model = SentenceTransformer(EMBED_MODEL)
    retriever = Retriever(sentences, embeddings)
    reranker = Reranker(RERANK_MODEL) if args.rerank else None

    result = match(
        query=query,
        model=model,
        retriever=retriever,
        reranker=reranker,
        retrieve_k=args.retrieve_k,
        return_k=args.top_k,
        min_score=args.min_score,
        ambiguity_margin=args.ambiguity_margin,
    )

    print("\n=== INPUT ===")
    print(query)

    print("\n=== STATUS ===")
    print(result["status"])

    if result["status"] == "no_match":
        print("\nNo sufficiently close match found.")
        return 0

    print("\n=== BEST MATCH ===")
    print(f"{result['best']['sentence']}  (score: {result['best']['score']:.4f})")

    print(f"\n=== TOP {len(result['candidates'])} ===")
    for i, c in enumerate(result["candidates"], 1):
        print(f"{i:>2}. {c['sentence']}  (score: {c['score']:.4f})")

    if result["status"] == "ambiguous":
        print("\n⚠ Ambiguous: multiple interpretations detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())