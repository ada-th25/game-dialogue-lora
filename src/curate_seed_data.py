"""
Automatic cleanup pass over data/seed_raw.jsonl, followed by an
interactive manual review of anything flagged. Writes accepted
examples to data/seed_curated.jsonl.

Run:
    python src/curate_seed_data.py
"""

import json
import re
from pathlib import Path

RAW_PATH = Path("data/seed_raw.jsonl")
CURATED_PATH = Path("data/seed_curated.jsonl")

WORD_LIMIT = 25  # flag (not auto-reject) anything over this for manual review


def strip_quotes(text: str) -> str:
    """Remove leading/trailing quote marks the model didn't consistently drop."""
    text = text.strip()
    text = re.sub(r'^[\'"]+|[\'"]+$', "", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def load_raw(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_curated_keys(path: Path) -> set:
    """Resume support: skip rows already reviewed if the script reruns."""
    keys = set()
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    keys.add((row["archetype"], row["mood"], row["topic"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def main():
    raw_rows = load_raw(RAW_PATH)
    already_done = load_curated_keys(CURATED_PATH)
    print(f"{len(raw_rows)} raw examples, {len(already_done)} already reviewed\n")

    with open(CURATED_PATH, "a") as out_f:
        for i, row in enumerate(raw_rows):
            key = (row["archetype"], row["mood"], row["topic"])
            if key in already_done:
                continue

            dialogue = strip_quotes(row["dialogue"])
            wc = word_count(dialogue)
            flag = wc > WORD_LIMIT

            print(f"[{i}] {row['archetype']} / {row['mood']} / {row['topic']}")
            print(f"    {dialogue}")
            if flag:
                print(f"    ⚠ {wc} words (over {WORD_LIMIT})")

            # a = accept, e = edit, s = skip/reject, q = quit and save progress
            choice = input("    [a]ccept / [e]dit / [s]kip / [q]uit: ").strip().lower()

            if choice == "q":
                print("Stopping. Progress saved — rerun to resume.")
                break
            elif choice == "s":
                continue
            elif choice == "e":
                new_text = input("    New dialogue: ").strip()
                dialogue = new_text if new_text else dialogue

            row_out = {
                "archetype": row["archetype"],
                "mood": row["mood"],
                "topic": row["topic"],
                "dialogue": dialogue,
            }
            out_f.write(json.dumps(row_out) + "\n")
            out_f.flush()
            print("    ✓ saved\n")

    print("Done. Review data/seed_curated.jsonl")


if __name__ == "__main__":
    main()