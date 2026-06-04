from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream a FineWeb-Edu slice into a local text cache")
    p.add_argument("--name", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="./data/fineweb_edu_sample_10bt.txt")
    p.add_argument("--max_docs", type=int, default=25000)
    p.add_argument("--min_chars", type=int, default=1)
    return p.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", args.name, split=args.split, streaming=True)
    count = 0
    chars = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text") or row.get("content") or ""
            text = text.strip()
            if len(text) < args.min_chars:
                continue
            f.write(text.replace("\x00", " "))
            f.write("\n<|endoftext|>\n")
            count += 1
            chars += len(text)
            if count % 1000 == 0:
                print(f"cached docs={count} chars={chars} out={out}", flush=True)
            if args.max_docs and count >= args.max_docs:
                break
    print(f"done docs={count} chars={chars} out={out}", flush=True)


if __name__ == "__main__":
    main()
