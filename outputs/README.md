# outputs/

Generated artifacts. Everything here except this README is gitignored.

- `chunks.json` and `index.faiss` written by `scripts/01_build_index.py`.
- `eval_rerank.json` written by `scripts/02_eval_rerank.py` (precision/recall
  before vs after reranking).
- `sweep.json` and `sweep.png` written by `scripts/03_sweep_chunk_size.py`
  (retrieval quality across chunk sizes).

Run the scripts to populate this directory.
