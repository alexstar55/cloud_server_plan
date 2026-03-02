#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def count_json_items(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data)
        return 0
    except Exception:
        return -2


def find_version_root(root: Path) -> Path:
    direct = root / "v1.0-mini"
    if direct.exists():
        return direct
    nested = root / "nuscenes" / "v1.0-mini"
    if nested.exists():
        return nested
    return direct


def main() -> int:
    parser = argparse.ArgumentParser(description="nuScenes-mini smoke helper for AlpaSim pipeline")
    parser.add_argument("--nuscenes-root", required=True, help="Path to mounted nuScenes mini root, e.g. /data/nu_mini")
    parser.add_argument("--output-dir", required=True, help="Output directory for smoke artifacts")
    parser.add_argument("--tag", default="smoke", help="Tag recorded in output manifest")
    args = parser.parse_args()

    root = Path(args.nuscenes_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    version_root = find_version_root(root)
    scene_json = version_root / "scene.json"
    sample_json = version_root / "sample.json"
    sample_data_json = version_root / "sample_data.json"

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "nuscenes_root": str(root),
        "resolved_version_root": str(version_root),
        "files": {
            "scene.json": str(scene_json),
            "sample.json": str(sample_json),
            "sample_data.json": str(sample_data_json),
        },
        "counts": {
            "scene": count_json_items(scene_json),
            "sample": count_json_items(sample_json),
            "sample_data": count_json_items(sample_data_json),
        },
        "status": "ok",
    }

    if summary["counts"]["scene"] < 0 or summary["counts"]["sample"] < 0:
        summary["status"] = "dataset_missing_or_invalid"

    manifest = out_dir / "smoke_manifest.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    marker = out_dir / "_smoke_complete"
    marker.write_text(datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")

    report = out_dir / "smoke_report.txt"
    report.write_text(
        "\n".join(
            [
                f"status={summary['status']}",
                f"nuscenes_root={summary['nuscenes_root']}",
                f"resolved_version_root={summary['resolved_version_root']}",
                f"scene_count={summary['counts']['scene']}",
                f"sample_count={summary['counts']['sample']}",
                f"sample_data_count={summary['counts']['sample_data']}",
                f"manifest={manifest}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[SMOKE] status={summary['status']}")
    print(f"[SMOKE] manifest={manifest}")
    print(f"[SMOKE] report={report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
