"""
Builds a blind rating sheet for the human-rated rubric (Part B of eval).

Pools baseline outputs (and fine-tuned outputs, once they exist) into one
shuffled list with no visible model label, so ratings aren't biased by
knowing which model produced which line. Writes two separate files:

  - data/rating_sheet.csv  — what you actually rate. No model identity.
  - data/rating_key.csv    — id -> which model generated it. Don't open
                              this until after you've filled in the sheet.

Run:
    python src/make_blind_rating_sheet.py
"""

import csv
import json
import random
from pathlib import Path

BASELINE_PATH = Path("data/baseline_outputs.jsonl")
FINETUNED_PATH = Path("data/finetuned_outputs.jsonl")  # may not exist yet

SHEET_PATH = Path("data/rating_sheet.csv")
KEY_PATH = Path("data/rating_key.csv")

RANDOM_SEED = 7  # just for reproducibility of the shuffle; doesn't affect blinding
RUBRIC_FIELDS = ["archetype_fit", "mood_fit", "naturalness"]


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pool(baseline_rows: list[dict], finetuned_rows: list[dict]) -> list[dict]:
    """Combine baseline + fine-tuned outputs into one flat list of records,
    each tagged with its source model and the actual dialogue text."""
    pool = []
    for row in baseline_rows:
        pool.append({
            "source": "baseline",
            "archetype": row["archetype"],
            "mood": row["mood"],
            "topic": row["topic"],
            "dialogue": row["baseline_generated"],
        })
    for row in finetuned_rows:
        pool.append({
            "source": "finetuned",
            "archetype": row["archetype"],
            "mood": row["mood"],
            "topic": row["topic"],
            "dialogue": row["finetuned_generated"],
        })
    return pool


def write_rating_files(pool: list[dict], sheet_path: Path, key_path: Path):
    """Write `sheet_path` (archetype/mood/topic/dialogue + blank rubric
    columns, NO `source` field) and `key_path` (id -> source), using the
    same `id` value to link rows across the two files.

    `pool` is a list of dicts like:
        {"source": "baseline", "archetype": ..., "mood": ..., "topic": ...,
         "dialogue": ...}
    Each needs an "id" field added (e.g. "r0", "r1", ...) before writing.
    """
    with open(sheet_path, "w", newline="") as sheet_file, open(key_path, "w", newline="") as key_file:
        sheet_writer = csv.DictWriter(sheet_file, fieldnames=["id", "archetype", "mood", "topic", "dialogue"] + RUBRIC_FIELDS)
        key_writer = csv.DictWriter(key_file, fieldnames=["id", "source"])
        sheet_writer.writeheader()
        key_writer.writeheader()

        for i, row in enumerate(pool):
            row_id = f"r{i}"
            sheet_row = {
                "id": row_id,
                "archetype": row["archetype"],
                "mood": row["mood"],
                "topic": row["topic"],
                "dialogue": row["dialogue"],
            }
            for field in RUBRIC_FIELDS:
                sheet_row[field] = ""  # blank rubric columns
            sheet_writer.writerow(sheet_row)

            key_row = {"id": row_id, "source": row["source"]}
            key_writer.writerow(key_row)


def main():
    baseline_rows = load_jsonl(BASELINE_PATH)
    finetuned_rows = load_jsonl(FINETUNED_PATH)
    if not finetuned_rows:
        print(f"Note: {FINETUNED_PATH} not found yet — pooling baseline only for now.")

    pool = build_pool(baseline_rows, finetuned_rows)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pool)

    write_rating_files(pool, SHEET_PATH, KEY_PATH)
    print(f"Wrote {len(pool)} rows to {SHEET_PATH} (rate this) and {KEY_PATH} (don't open yet).")


if __name__ == "__main__":
    main()
