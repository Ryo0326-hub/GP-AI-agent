#!/usr/bin/env python3
"""Merge resumable label JSONL shards in canonical task order."""
import argparse
import json
from pathlib import Path


def load_records(paths):
    records = {}
    provenance = None
    for raw_path in paths:
        path = Path(raw_path)
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = str(record.get("task_id", ""))
            if not task_id:
                raise ValueError(f"{path}:{line_number}: missing task_id")
            if task_id in records and records[task_id] != record:
                raise ValueError(f"conflicting duplicate label: {task_id}")
            current = record.get("provenance")
            if not isinstance(current, dict):
                raise ValueError(
                    f"{path}:{line_number}: missing label provenance")
            if provenance is None:
                provenance = current
            elif provenance != current:
                raise ValueError(
                    f"{path}:{line_number}: mixed label provenance")
            records[task_id] = record
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text())
    records = load_records(args.inputs)
    ordered_ids = [str(task["task_id"]) for task in tasks]
    missing = [task_id for task_id in ordered_ids if task_id not in records]
    extras = sorted(set(records) - set(ordered_ids))
    if missing or extras:
        raise ValueError(
            f"label coverage mismatch: missing={missing[:10]} extras={extras[:10]}")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = "".join(
        json.dumps(records[task_id], ensure_ascii=False) + "\n"
        for task_id in ordered_ids
    )
    temporary.write_text(payload)
    temporary.replace(destination)
    print(f"merged {len(ordered_ids)} labels into {destination}")


if __name__ == "__main__":
    main()
