# data/

Holds the corpus and the eval set.

- `qa.json` ships with the repo. It is the hand-written retrieval eval set:
  8-12 queries, each with one or more answer substrings that must appear
  verbatim in a relevant passage. This file is tracked in git.
- `corpus/` is created by `scripts/00_prepare_corpus.py`, which downloads a few
  small public-domain Project Gutenberg plaintext books and strips their license
  headers/footers. The corpus itself is gitignored (only this README and
  `qa.json` are tracked).

Everything else under `data/` is ignored by git. Run
`python scripts/00_prepare_corpus.py` to populate `corpus/`.
