"""
Generate seed dialogue data by prompting a local Ollama model over the
archetype x mood x topic schema. Writes one JSON object per line to
data/seed_raw.jsonl, saving incrementally so a crash/timeout doesn't
lose earlier progress (same pattern as the BHL project).
"""

import json
import time
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

OUT_PATH = Path("data/seed_raw.jsonl")
OUT_PATH.parent.mkdir(exist_ok=True)

# Fixed vocabularies — keep these in sync with docs/schema.md
ARCHETYPES = [
    "grumpy_blacksmith",
    "nervous_merchant",
    "wise_old_healer",
    "arrogant_knight",
    "cheerful_innkeeper",
    "suspicious_guard",
]

MOODS = [
    "annoyed",
    "cheerful",
    "fearful",
    "bored",
    "excited",
]

# Keep topics short and varied; expand this list as needed
TOPICS = [
    "broken sword",
    "missing shipment",
    "strange noises at night",
    "a rival business",
    "an approaching storm",
    "rumors of a dragon",
    "unpaid debt",
    "a lost child",
]

PROMPT_TEMPLATE = """Generate a single line of dialogue for a {archetype} NPC in a fantasy game.
Mood: {mood}
Topic: {topic}
Keep it under 20 words, in character, no narration — just the spoken line.
Respond with ONLY the dialogue line, no quotation marks, no explanation."""


def call_ollama(prompt: str, timeout: int = 60) -> str:
    """Call local Ollama with streaming disabled and a wall-clock timeout."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def load_existing_keys(path: Path) -> set:
    """Resume support: skip combos already generated if the script reruns."""
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
    existing = load_existing_keys(OUT_PATH)
    combos = [
        (a, m, t)
        for a in ARCHETYPES
        for m in MOODS
        for t in TOPICS
        if (a, m, t) not in existing
    ]
    print(f"{len(existing)} examples already generated, {len(combos)} remaining")

    with open(OUT_PATH, "a") as f:
        for i, (archetype, mood, topic) in enumerate(combos):
            prompt = PROMPT_TEMPLATE.format(archetype=archetype, mood=mood, topic=topic)
            try:
                dialogue = call_ollama(prompt)
            except requests.exceptions.RequestException as e:
                print(f"[{i}] FAILED ({archetype}, {mood}, {topic}): {e}")
                continue

            row = {
                "archetype": archetype,
                "mood": mood,
                "topic": topic,
                "dialogue": dialogue,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()  # incremental save, same pattern as BHL fetch script
            print(f"[{i}] {archetype}/{mood}: {dialogue}")

            time.sleep(0.2)  # small buffer between calls


if __name__ == "__main__":
    main()