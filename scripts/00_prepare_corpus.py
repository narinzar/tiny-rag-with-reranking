"""Fetch a small public-domain corpus and write it under data/.

Downloads a handful of Project Gutenberg plaintext books, strips the Gutenberg
license header/footer, and writes cleaned .txt files into data/corpus/. A small
hand-written eval set (data/qa.json) ships with this repo already; this script
does NOT overwrite it, it only checks that the answer substrings actually appear
in the fetched corpus and warns if any do not.

Run:
    python scripts/00_prepare_corpus.py

No API keys required. HF_TOKEN in .env is unrelated (only used later for model
downloads) and optional.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
QA_PATH = DATA / "qa.json"

# Small, well-known public-domain texts. Gutenberg plaintext mirrors.
BOOKS = {
    "alice_in_wonderland.txt": "https://www.gutenberg.org/files/11/11-0.txt",
    "the_time_machine.txt": "https://www.gutenberg.org/files/35/35-0.txt",
    "a_study_in_scarlet.txt": "https://www.gutenberg.org/files/244/244-0.txt",
}

_START_RE = re.compile(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
_END_RE = re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)


def strip_gutenberg(raw: str) -> str:
    """Remove the Gutenberg license header and footer, keeping the body."""
    start = _START_RE.search(raw)
    end = _END_RE.search(raw)
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    # Collapse Windows newlines and excessive blank lines.
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "tiny-rag/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main() -> int:
    load_dotenv(ROOT / ".env")
    CORPUS.mkdir(parents=True, exist_ok=True)

    for fname, url in tqdm(BOOKS.items(), desc="download"):
        dest = CORPUS / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            raw = fetch(url)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: failed to fetch {url}: {exc}", file=sys.stderr)
            continue
        dest.write_text(strip_gutenberg(raw), encoding="utf-8")

    if not QA_PATH.exists():
        print(f"ERROR: expected bundled eval set at {QA_PATH}", file=sys.stderr)
        return 1

    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    corpus_text = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.txt"))
    )

    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    hay = norm(corpus_text)
    missing = 0
    for item in qa["queries"]:
        for ans in item["answers"]:
            if norm(ans) not in hay:
                missing += 1
                print(
                    f"WARN: answer substring not found in corpus: "
                    f"{item['id']}: {ans!r}",
                    file=sys.stderr,
                )
    n_docs = len(list(CORPUS.glob("*.txt")))
    print(f"corpus ready: {n_docs} docs in {CORPUS}")
    print(f"eval set: {len(qa['queries'])} queries, {missing} unmatched substrings")
    if missing:
        print(
            "Some answer substrings were not found. If a download was skipped or "
            "a source edition differs, adjust data/qa.json accordingly.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
