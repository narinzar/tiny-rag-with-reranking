"""Sweep chunk sizes and report retrieval quality to find the sweet spot.

For each candidate chunk size, this rechunks the corpus, embeds the chunks into a
fresh FAISS index, runs every eval query, and reports precision@k / recall@k. It
prints a table and writes outputs/sweep.json plus outputs/sweep.png.

Run:
    python scripts/03_sweep_chunk_size.py --strategy adaptive --sizes 32 64 128 256 512 --k 5

Expected behavior: there is a chunk-size sweet spot. Too-small chunks lose the
surrounding context a passage needs, and too-large chunks dilute the relevant
span among unrelated text, both of which hurt retrieval. The best size sits in
the middle. Absolute numbers depend on your run.
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
from src.eval import sweep_chunk_sizes  # noqa: E402
from src.index import FaissIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
OUTPUTS = ROOT / "outputs"
QA_PATH = ROOT / "data" / "qa.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="adaptive")
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512]
    )
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    if not any(CORPUS.glob("*.txt")):
        print("No corpus found. Run scripts/00_prepare_corpus.py first.")
        return 1

    documents = {
        p.name: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.txt"))
    }
    queries = json.loads(QA_PATH.read_text(encoding="utf-8"))["queries"]

    # Build one embedder and reuse it across all sizes.
    embedder = Embedder()

    def retrieve_fn(chunks, question, k):
        vecs = embedder.encode([c.text for c in chunks], show_progress=False)
        idx = FaissIndex(dim=embedder.dim)
        idx.add(vecs)
        qvec = embedder.encode_one(question)
        hits = idx.search(qvec, k)[0]
        return [h.chunk_id for h in hits]

    rows = []
    for size in tqdm(args.sizes, desc="sweep"):
        row = sweep_chunk_sizes(
            documents=documents,
            queries=queries,
            sizes=[size],
            k=args.k,
            retrieve_fn=retrieve_fn,
            strategy=args.strategy,
        )[0]
        rows.append(row)

    print()
    print(f"strategy={args.strategy}   metric @k={args.k}")
    print("-" * 64)
    print(
        f"{'size':>6}{'n_chunks':>10}{'q_with_rel':>12}"
        f"{'precision':>13}{'recall':>13}"
    )
    for r in rows:
        print(
            f"{r.chunk_tokens:>6}{r.n_chunks:>10}{r.n_queries_with_relevant:>12}"
            f"{r.precision:>13.4f}{r.recall:>13.4f}"
        )
    best = max(rows, key=lambda r: (r.precision, r.recall))
    print("-" * 64)
    print(f"best size by precision@k: {best.chunk_tokens}")

    payload = [
        {
            "size": r.chunk_tokens,
            "n_chunks": r.n_chunks,
            "precision": r.precision,
            "recall": r.recall,
        }
        for r in rows
    ]
    (OUTPUTS / "sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r.chunk_tokens for r in rows]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, [r.precision for r in rows], "o-", label=f"precision@{args.k}")
        plt.plot(xs, [r.recall for r in rows], "s-", label=f"recall@{args.k}")
        plt.xlabel("chunk size (target tokens)")
        plt.ylabel("score")
        plt.title(f"Chunk-size sweep ({args.strategy})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUTS / "sweep.png", dpi=120)
        print(f"wrote {OUTPUTS / 'sweep.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"(skipped plot: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
