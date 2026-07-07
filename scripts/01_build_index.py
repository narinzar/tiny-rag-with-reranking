"""Chunk the corpus, embed the chunks, and build a FAISS index.

Writes to outputs/:
    outputs/chunks.json   chunk text + source offsets + doc id
    outputs/index.faiss   the FAISS inner-product index

Run (after 00_prepare_corpus.py):
    python scripts/01_build_index.py --strategy adaptive --size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_text  # noqa: E402
from src.embedder import Embedder, DEFAULT_MODEL  # noqa: E402
from src.index import FaissIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
OUTPUTS = ROOT / "outputs"


def build_chunks(strategy: str, size: int, overlap: int):
    records = []
    for path in sorted(CORPUS.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if strategy == "sliding_window":
            chunks = chunk_text(
                text, strategy=strategy, chunk_tokens=size, overlap_tokens=overlap
            )
        elif strategy == "adaptive":
            chunks = chunk_text(text, strategy=strategy, target_tokens=size)
        else:
            chunks = chunk_text(text, strategy=strategy, chunk_tokens=size)
        for ch in chunks:
            records.append(
                {
                    "doc": path.name,
                    "start": ch.start,
                    "end": ch.end,
                    "text": ch.text,
                }
            )
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="adaptive")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    if not any(CORPUS.glob("*.txt")):
        print("No corpus found. Run scripts/00_prepare_corpus.py first.")
        return 1

    records = build_chunks(args.strategy, args.size, args.overlap)
    print(f"chunked into {len(records)} chunks using {args.strategy} (size={args.size})")

    embedder = Embedder(model_name=args.model)
    vecs = embedder.encode([r["text"] for r in records], show_progress=True)

    index = FaissIndex(dim=embedder.dim)
    index.add(vecs)

    (OUTPUTS / "chunks.json").write_text(
        json.dumps({"model": args.model, "chunks": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    index.save(str(OUTPUTS / "index.faiss"))
    print(f"wrote {OUTPUTS / 'chunks.json'} and {OUTPUTS / 'index.faiss'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
