"""
One-off sanity check: are any of our curated training examples long enough
to get truncated by train_lora.py's max_length=128?

Run:
    python src/check_token_lengths.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B"
CURATED_PATH = Path("data/seed_curated.jsonl")

PROMPT_TEMPLATE = """Generate a single line of dialogue for a {archetype} NPC in a fantasy game.
Mood: {mood}
Topic: {topic}
Keep it under 20 words, in character, no narration — just the spoken line.
Dialogue:"""


def load_curated(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_full_text(row):
    prompt = PROMPT_TEMPLATE.format(
        archetype=row["archetype"], mood=row["mood"], topic=row["topic"]
    )
    return f"{prompt} {row['dialogue']}"


def main():
    curated = load_curated(CURATED_PATH)
    print(f"{len(curated)} curated examples loaded")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    lengths = []
    for row in curated:
        text = build_full_text(row)
        token_ids = tokenizer(text)["input_ids"]
        lengths.append(len(token_ids))

    print(f"Max length: {max(lengths)}")
    print(f"Average length: {sum(lengths) / len(lengths):.1f}")
    print(f"Examples with length > 128: {sum(1 for l in lengths if l > 128)}")


if __name__ == "__main__":
    main()
