#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def rewrite_paths(json_path: Path, old_prefix: str, new_prefix: str, field: str, backup: bool):
    if not json_path.exists():
        raise FileNotFoundError(f"json not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("sample_data.json must be a list")

    changed = 0
    total = 0
    for row in data:
        if not isinstance(row, dict):
            continue
        total += 1
        val = row.get(field)
        if isinstance(val, str) and val.startswith(old_prefix):
            row[field] = new_prefix + val[len(old_prefix):]
            changed += 1

    if backup:
        bak = json_path.with_suffix(json_path.suffix + ".bak")
        if not bak.exists():
            bak.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: total={total}, changed={changed}, file={json_path}")


def main():
    parser = argparse.ArgumentParser(description="Rewrite absolute path prefix in sample_data.json")
    parser.add_argument("--json", required=True, help="Path to sample_data.json")
    parser.add_argument("--old-prefix", required=True, help="Old absolute prefix")
    parser.add_argument("--new-prefix", required=True, help="New absolute prefix")
    parser.add_argument("--field", default="filename", help="Field name to rewrite, default=filename")
    parser.add_argument("--backup", action="store_true", help="Create .bak file")
    args = parser.parse_args()

    rewrite_paths(Path(args.json), args.old_prefix, args.new_prefix, args.field, args.backup)


if __name__ == "__main__":
    main()
