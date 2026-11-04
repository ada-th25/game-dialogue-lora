"""
Rule-based (objective, deterministic) checks for generated dialogue lines.
Run this against baseline_outputs.jsonl now, and against the fine-tuned
model's outputs later, to compare compliance rates.

Run:
    python src/eval_rules.py
"""

import json
import re
from pathlib import Path

CURATED_PATH = Path("data/seed_curated.jsonl")
BASELINE_PATH = Path("data/baseline_outputs.jsonl")

MAX_WORDS = 20
NEAR_DUPLICATE_THRESHOLD = 0.6

# Phrases/patterns that indicate narration leaked into what should be a
# pure spoken line (e.g. "*sighs* Another day..." or 'He mutters, "...".')
NARRATION_PATTERNS = [
    r"\*[^*]+\*",           # *stage direction*
    r"\b(he|she|they) (says|say|mutters|mutter|whispers|whisper|shouts|shout)\b",
]


def load_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def word_count_ok(text: str, max_words: int = MAX_WORDS) -> bool:
    return len(text.split()) <= max_words


def no_narration_leakage(text: str) -> bool:
    for pattern in NARRATION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return False
    return True


def jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings, case-insensitive."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def is_near_duplicate(generated: str, training_dialogues: list[str], threshold: float = NEAR_DUPLICATE_THRESHOLD) -> bool:
    """Return True if `generated` is suspiciously similar to ANY dialogue in
    `training_dialogues` (word-level Jaccard similarity above `threshold`).

    This flags likely memorization rather than genuine generation.
    """
    for dialogue in training_dialogues:
        similarity = jaccard_similarity(generated, dialogue)
        if similarity > threshold:
            return True
    return False


def evaluate(outputs: list[dict], generated_key: str, training_dialogues: list[str]) -> dict:
    """Run all three checks over `outputs` and return pass-rate summary."""
    n = len(outputs)
    word_ok = sum(word_count_ok(o[generated_key]) for o in outputs)
    narration_ok = sum(no_narration_leakage(o[generated_key]) for o in outputs)
    not_duplicate = sum(
        not is_near_duplicate(o[generated_key], training_dialogues) for o in outputs
    )
    return {
        "n": n,
        "word_count_ok_pct": round(100 * word_ok / n, 1),
        "no_narration_leakage_pct": round(100 * narration_ok / n, 1),
        "not_near_duplicate_pct": round(100 * not_duplicate / n, 1),
    }


def main():
    curated = load_jsonl(CURATED_PATH)
    training_dialogues = [row["dialogue"] for row in curated]

    baseline_outputs = load_jsonl(BASELINE_PATH)
    summary = evaluate(baseline_outputs, "baseline_generated", training_dialogues)

    print("Baseline rule-based eval:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
