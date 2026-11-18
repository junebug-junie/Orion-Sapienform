#!/usr/bin/env python3
"""Download one pinned checkpoint after checking disk headroom; never start a worker."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--draft", action="store_true", help="prepare the pinned optional DFlash2 draft instead")
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name("deep-thinker.lock.json").read_text())["draft" if args.draft else "model"]
    # Include metadata, partial download space and 30 GiB operator safety reserve.
    required = lock["tensor_bytes"] * 2 + 30 * 1024**3
    free = shutil.disk_usage(args.root).free
    print(json.dumps({"model": lock, "available_bytes": free, "required_free_bytes": required}))
    if free < required:
        raise SystemExit("insufficient storage headroom; no download started")
    if not args.download:
        return
    from huggingface_hub import snapshot_download
    dest = args.root / lock["directory"]
    snapshot_download(repo_id=lock["repo_id"], revision=lock["revision"], local_dir=dest,
                      allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"], max_workers=4)
    if not args.draft:
        index = json.loads((dest / "model.safetensors.index.json").read_text())
        for name in set(index["weight_map"].values()):
            if not (dest / name).is_file():
                raise SystemExit(f"missing weight shard: {name}")
    for shard in lock['shards']:
        path = dest / shard['file']
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if path.stat().st_size != shard['bytes'] or digest != shard['sha256']:
            raise SystemExit(f"checkpoint checksum mismatch: {shard['file']}")
    (dest / "orion-revision.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(f"download complete: {dest}; runtime quality remains UNVERIFIED")


if __name__ == "__main__":
    main()
