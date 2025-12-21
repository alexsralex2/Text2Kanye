"""
Semantic sentence matcher.

- Reads preset sentences from: sentencelist.txt (one sentence per line)
- Reads an input sentence from stdin (either as CLI args or interactively)
- Finds the closest-meaning sentence(s) using sentence embeddings
- Optionally reranks top candidates with a cross-encoder for better nuance
- Prints the best match + top-k list + an ambiguity/no-match status

Install:
  pip install sentence-transformers numpy

Usage:
  python match_sentences.py "your input sentence here"
  # or
  python match_sentences.py
  (then type/paste your sentence and press Enter)

Notes:
- For ~1000 sentences, brute-force cosine similarity is fast.
- If your sentences are often ambiguous (double meanings), top-k output is useful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# CrossEncoder is optional; we'll import lazily if enabled.


DEFAULT_SENTENCE_FILE = "sentencelist.txt"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def read_sentences(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sentence list file not found: {p.resolve()}")

    sentences: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:  # skip blank lines
                sentences.append(s)

    if not sentences:
        raise ValueError(f"No sentences found in file: {p.resolve()}")

    return sentences


class EmbedRetriever:
    def __init__(self, presets: List[str], model_name: str = DEFAULT_EMBED_MODEL):
        self.presets = presets
        self.model = SentenceTransformer(model_name)
        # Normalize so cosine similarity == dot product
        self.embs = self.model.encode(presets, normalize_embeddings=True)

    def top_k(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.embs @ q  # vector of similarities
        k = min(k, len(self.presets))
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[int(i)])) for i in idx]


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        """
        Returns list of (candidate_index_relative_to_input_list, score),
        sorted best-to-worst.
        """
        pairs = [(query, c) for c in candidates]
        scores = self.model.predict(pairs)
        order = np.argsort(-scores)
        return [(int(i), float(scores[int(i)])) for i in order]


def match_sentence(
    query: str,
    presets: List[str],
    retriever: EmbedRetriever,
    reranker: Optional[Reranker] = None,
    k_retrieve: int = 20,
    return_k: int = 5,
    min_score: float = 0.45,
    ambiguity_margin: float = 0.05,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "status": "ok" | "ambiguous" | "no_match",
        "best": {"sentence": str, "score": float} | None,
        "candidates": [{"sentence": str, "score": float}, ...],
      }
    """
    retrieved = retriever.top_k(query, k=k_retrieve)
    top_idxs = [i for i, _ in retrieved]
    top_scores = [s for _, s in retrieved]

    # "No match" check from retriever similarity
    if top_scores[0] < min_score:
        return {"status": "no_match", "best": None, "candidates": []}

    # Optionally rerank the top candidates for better nuance
    if reranker is not None:
        candidate_texts = [presets[i] for i in top_idxs]
        reranked_rel = reranker.rerank(query, candidate_texts)
        # Map relative indices back to original preset indices
        final = [(top_idxs[i_rel], score) for i_rel, score in reranked_rel]
    else:
        final = retrieved

    best_idx, best_score = final[0]
    s1 = best_score
    s2 = final[1][1] if len(final) > 1 else -1.0
    ambiguous = (s1 - s2) < ambiguity_margin

    top = [
        {"sentence": presets[i], "score": float(s)}
        for i, s in final[: min(return_k, len(final))]
    ]

    return {
        "status": "ambiguous" if ambiguous else "ok",
        "best": {"sentence": presets[best_idx], "score": float(best_score)},
        "candidates": top,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Find closest-meaning sentence from a preset list.")
    ap.add_argument(
        "query",
        nargs="*",
        help="Input sentence to match. If omitted, you'll be prompted to type one.",
    )
    ap.add_argument(
        "--file",
        default=DEFAULT_SENTENCE_FILE,
        help=f"Path to sentence list file (default: {DEFAULT_SENTENCE_FILE})",
    )
    ap.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Sentence embedding model (default: {DEFAULT_EMBED_MODEL})",
    )
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking for better precision (slower).",
    )
    ap.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help=f"Cross-encoder model to rerank (default: {DEFAULT_RERANK_MODEL})",
    )
    ap.add_argument(
        "--k-retrieve",
        type=int,
        default=20,
        help="How many candidates to retrieve before reranking (default: 20).",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many matches to print (default: 5).",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.45,
        help="If best similarity is below this, report no match (default: 0.45).",
    )
    ap.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.05,
        help="If (best - 2nd best) < margin, report ambiguous (default: 0.05).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    presets = read_sentences(args.file)

    if args.query:
        query = " ".join(args.query).strip()
    else:
        # Interactive input
        try:
            query = input("Enter a sentence to match: ").strip()
        except EOFError:
            return 1

    if not query:
        print("No input sentence provided.", file=sys.stderr)
        return 1

    retriever = EmbedRetriever(presets, model_name=args.embed_model)

    reranker = None
    if args.rerank:
        try:
            reranker = Reranker(model_name=args.rerank_model)
        except Exception as e:
            print(f"Failed to load reranker model: {e}", file=sys.stderr)
            print("Continuing without reranking.", file=sys.stderr)
            reranker = None

    result = match_sentence(
        query=query,
        presets=presets,
        retriever=retriever,
        reranker=reranker,
        k_retrieve=args.k_retrieve,
        return_k=args.top_k,
        min_score=args.min_score,
        ambiguity_margin=args.ambiguity_margin,
    )

    print("\n=== Input ===")
    print(query)

    print("\n=== Status ===")
    print(result["status"])

    if result["status"] == "no_match":
        print("\nNo sufficiently close match found (try lowering --min-score).")
        return 0

    print("\n=== Best match ===")
    best = result["best"]
    print(f"{best['sentence']}\n(score: {best['score']:.4f})")

    print(f"\n=== Top {len(result['candidates'])} candidates ===")
    for rank, cand in enumerate(result["candidates"], start=1):
        print(f"{rank:>2}. {cand['sentence']}  (score: {cand['score']:.4f})")

    if result["status"] == "ambiguous":
        print(
            "\nNote: This looks ambiguous (top scores are close). "
            "Consider using the top-k list or adding more context."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())