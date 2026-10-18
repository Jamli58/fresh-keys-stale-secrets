"""Verify downloaded model weights against ModelScope's published sha256 and record provenance.

  python scripts/model_provenance.py Qwen/Qwen2.5-32B-Instruct-AWQ /root/autodl-tmp/models/qwen32b \
      results/model_provenance_qwen32b.json

Checks every weight, config and tokenizer file the repo lists (README/LICENSE are
skipped; paths matching --exclude were deliberately not downloaded). Exits non-zero
on any mismatch or missing file, so a corrupt download never serves paper runs.
"""

import argparse
import fnmatch
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://modelscope.cn/api/v1/models/{model_id}/repo/files?Revision={revision}&Recursive=true"
CHECKED_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".tiktoken")
DEFAULT_EXCLUDE = ("original/*",)
CHUNK = 16 * 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def compare(remote_files: list[dict], local_dir: Path, exclude: tuple[str, ...] = DEFAULT_EXCLUDE) -> dict:
    checked, mismatched, missing = {}, [], []
    for f in remote_files:
        path = f["Path"]
        if not path.endswith(CHECKED_SUFFIXES) or any(fnmatch.fnmatch(path, p) for p in exclude):
            continue
        local = Path(local_dir) / path
        if not local.exists():
            missing.append(path)
            continue
        got = sha256_of(local)
        checked[path] = {"sha256": got, "size": local.stat().st_size, "revision": f.get("Revision")}
        if got != f.get("Sha256"):
            mismatched.append(path)
    return {"ok": not mismatched and not missing, "checked": checked, "mismatched": mismatched, "missing": missing}


def fetch_file_list(model_id: str, revision: str) -> list[dict]:
    with urllib.request.urlopen(API.format(model_id=model_id, revision=revision), timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    files = payload.get("Data", {}).get("Files")
    if not isinstance(files, list):
        raise SystemExit(f"unexpected ModelScope response for {model_id}: {str(payload)[:300]}")
    return [f for f in files if f.get("Type") == "blob"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("local_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--revision", default="master")
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))
    args = ap.parse_args()
    report = compare(fetch_file_list(args.model_id, args.revision), args.local_dir, tuple(args.exclude))
    record = {"model_id": args.model_id, "revision": args.revision, "local_dir": str(args.local_dir),
              "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    status = "OK" if report["ok"] else f"FAILED mismatched={report['mismatched']} missing={report['missing']}"
    print(f"{args.model_id}: {len(report['checked'])} files checked, {status} -> {args.out}")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
