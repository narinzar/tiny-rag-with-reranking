# tiny-rag-with-reranking

A small, readable RAG pipeline over a real public-domain document set:
chunk -> embed (bi-encoder) -> FAISS retrieval -> cross-encoder reranking, with
retrieval precision/recall measured before and after reranking, and a chunk-size
sweep that picks a chunking strategy by measured retrieval quality.

## Problem

Retrieval-augmented generation lives or dies on retrieval. A bi-encoder embeds
the query and each passage independently and compares vectors, which is fast but
blurs fine-grained relevance, so the truly-relevant passage often sits a few
ranks below where it should. Two levers fix this without touching the generator:
rerank the shortlist with a cross-encoder that reads each (query, passage) pair
jointly, and chunk the source text at the right granularity so a passage carries
enough context without diluting the relevant span. This repo builds both and
measures the effect on precision@k and recall@k instead of asserting it.

## Approach

- Four chunking strategies in `src/chunking.py`: fixed-token, sentence-aware,
  sliding-window with overlap, and an original `adaptive` strategy that merges
  short sentences up to a target size and splits any single over-long sentence.
  Every chunk carries `[start, end)` character offsets back into its source.
- Bi-encoder embedding (`all-MiniLM-L6-v2`) with L2-normalized vectors so a FAISS
  inner-product index computes cosine similarity directly.
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) re-scores only the retrieved
  top-N, which is where the accuracy gain is cheap.
- Relevance is labeled by answer substring, not by chunk id, so the eval set
  stays valid no matter how the corpus is chunked. This is what makes the
  chunk-size sweep an apples-to-apples comparison.
- The chunk-size sweep chooses the chunking configuration empirically by
  precision@k / recall@k rather than by a guessed constant.

## Setup

```
# 1. create a virtual environment (either tool)
uv venv --python 3.12 .venv
# or: python -m venv .venv
# then activate it (Windows: .venv\Scripts\activate ; Unix: source .venv/bin/activate)

# 2. install torch from the CUDA 12.8 index first (RTX 5090 / sm_120), then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 3. copy the env template (no secrets required)
cp .env.example .env
```

`.env` needs nothing for public models. `HF_TOKEN` is optional and only speeds up
or authorizes Hugging Face model downloads.

## How to run

Run the pipeline in order from the repo root:

```
# 0. fetch the corpus (a few Project Gutenberg books) into data/corpus/
python scripts/00_prepare_corpus.py

# 1. chunk + embed + build the FAISS index
python scripts/01_build_index.py --strategy adaptive --size 128

# 2. precision/recall before vs after cross-encoder reranking
python scripts/02_eval_rerank.py --retrieve 20 --k 5

# 3. sweep chunk sizes to find the sweet spot (writes outputs/sweep.png)
python scripts/03_sweep_chunk_size.py --strategy adaptive --sizes 32 64 128 256 512 --k 5
```

Run the tests (pure-python, no model downloads needed):

```
pytest -q
```

## Results

Numbers below are produced by running the commands above; this repo ships the
code, run it to populate them.

Reranking effect (`scripts/02_eval_rerank.py`):

| stage                    | precision@k | recall@k |
|--------------------------|-------------|----------|
| bi-encoder only          | TBD (run)   | TBD (run)|
| + cross-encoder rerank   | TBD (run)   | TBD (run)|

Expected behavior: the cross-encoder pass should raise precision@k over
bi-encoder-only retrieval, because it reads each (query, passage) pair jointly
and pushes the genuinely-relevant chunk up the ranking. The delta is printed by
the script.

Chunk-size sweep (`scripts/03_sweep_chunk_size.py`, plotted in
`outputs/sweep.png`):

| chunk size | precision@k | recall@k |
|------------|-------------|----------|
| 32         | TBD (run)   | TBD (run)|
| 64         | TBD (run)   | TBD (run)|
| 128        | TBD (run)   | TBD (run)|
| 256        | TBD (run)   | TBD (run)|
| 512        | TBD (run)   | TBD (run)|

Expected behavior: there is a chunk-size sweet spot. Too-small chunks lose the
context a passage needs to match its query, and too-large chunks bury the
relevant span among unrelated text, so both extremes score worse than a middle
size. The script prints the best size by precision@k.

## What I'd do next at larger scale

Swap the flat FAISS index for an IVF/HNSW index and shard the corpus so
retrieval stays sub-linear as the document count grows into the millions. Replace
the substring relevance labels with graded human judgments and report nDCG/MRR
alongside precision/recall, and batch the cross-encoder across queries on GPU so
reranking a large shortlist stays cheap.
