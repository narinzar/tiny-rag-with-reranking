"""Evaluate retrieval precision/recall BEFORE vs AFTER cross-encoder reranking.

Loads the chunks + FAISS index from 01_build_index.py, runs each eval query,
retrieves the top-N with the bi-encoder, then reranks with the cross-encoder,
and reports precision@k / recall@k for both.

Run:
    python scripts/02_eval_rerank.py --retrieve 20 --k 5

Expected behavior: the cross-encoder pass should raise precision@k (and often
recall@k at a fixed small k) over bi-encoder-only retrieval, because it reads
each (query, chunk) pair jointly. Absolute numbers depend on your run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import Chunk  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.eval import aggregate, relevant_chunk_ids  # noqa: E402
from src.index import FaissIndex  # noqa: E402
from src.rerank import CrossEncoderReranker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
QA_PATH = ROOT / "data" / "qa.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieve", type=int, default=20, help="top-N from bi-encoder")
    ap.add_argument("--k", type=int, default=5, help="cutoff for precision/recall@k")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")

    chunks_path = OUTPUTS / "chunks.json"
    index_path = OUTPUTS / "index.faiss"
    if not chunks_path.exists() or not index_path.exists():
        print("Missing index. Run scripts/01_build_index.py first.")
        return 1

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    records = payload["chunks"]
    model_name = payload["model"]
    chunk_objs = [
        Chunk(text=r["text"], start=r["start"], end=r["end"], meta={"doc": r["doc"]})
        for r in records
    ]

    index = FaissIndex.load(str(index_path))
    embedder = Embedder(model_name=model_name)
    reranker = CrossEncoderReranker()

    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    queries = qa["queries"]

    base_retrieved, rerank_retrieved, per_relevant = [], [], []

    for q in tqdm(queries, desc="eval"):
        rel = relevant_chunk_ids(chunk_objs, q["answers"])
        per_relevant.append(rel)

        qvec = embedder.encode_one(q["question"])
        hits = index.search(qvec, args.retrieve)[0]
        retrieved_ids = [h.chunk_id for h in hits]
        base_retrieved.append(retrieved_ids)

        texts = [chunk_objs[cid].text for cid in retrieved_ids]
        reranked = reranker.rerank(q["question"], retrieved_ids, texts)
        rerank_retrieved.append([r.chunk_id for r in reranked])

    base = aggregate(base_retrieved, per_relevant, args.k)
    rr = aggregate(rerank_retrieved, per_relevant, args.k)

    print()
    print(f"queries: {base.n_queries}   retrieve top-{args.retrieve}   metric @k={args.k}")
    print("-" * 56)
    print(f"{'stage':<24}{'precision@k':>15}{'recall@k':>15}")
    print(f"{'bi-encoder only':<24}{base.precision:>15.4f}{base.recall:>15.4f}")
    print(f"{'+ cross-encoder rerank':<24}{rr.precision:>15.4f}{rr.recall:>15.4f}")
    print("-" * 56)
    dp = rr.precision - base.precision
    print(f"precision delta from reranking: {dp:+.4f}")

    (OUTPUTS / "eval_rerank.json").write_text(
        json.dumps(
            {
                "retrieve": args.retrieve,
                "k": args.k,
                "bi_encoder": {"precision": base.precision, "recall": base.recall},
                "reranked": {"precision": rr.precision, "recall": rr.recall},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
