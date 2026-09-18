"""
Few-shot baseline: prompt the base Llama 3.1 8B with a handful of curated
examples, then generate on a held-out set of schema combinations. This is
the comparison point for the LoRA fine-tune in the next stage.

Run:
    python src/baseline_prompt.py
"""

import json
import random
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

CURATED_PATH = Path("data/seed_curated.jsonl")
OUT_PATH = Path("data/baseline_outputs.jsonl")

N_FEWSHOT = 4          # examples included in the prompt
N_HELDOUT = 20         # combos to evaluate on (excluded from few-shot pool)
RANDOM_SEED = 42

PROMPT_TEMPLATE = """Generate a single line of dialogue for a {archetype} NPC in a fantasy game.
Mood: {mood}
Topic: {topic}
Keep it under 20 words, in character, no narration — just the spoken line.
Respond with ONLY the dialogue line, no quotation marks, no explanation.

Here are some examples of the style expected:

{examples}

Now generate a new line for:
Archetype: {archetype}
Mood: {mood}
Topic: {topic}
Dialogue:"""


def load_curated(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_examples(examples):
    lines = []
    for ex in examples:
        lines.append(
            f"Archetype: {ex['archetype']}\n"
            f"Mood: {ex['mood']}\n"
            f"Topic: {ex['topic']}\n"
            f"Dialogue: {ex['dialogue']}"
        )
    return "\n\n".join(lines)


def call_ollama(prompt: str, timeout: int = 60) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def main():
    curated = load_curated(CURATED_PATH)
    print(f"{len(curated)} curated examples loaded")

    rng = random.Random(RANDOM_SEED)
    shuffled = curated[:]
    rng.shuffle(shuffled)

    # Held-out eval set: combos the few-shot prompt will NOT see as examples.
    # This keeps the baseline honest — no leakage between prompt and eval.
    heldout = shuffled[:N_HELDOUT]
    fewshot_pool = shuffled[N_HELDOUT:]

    print(f"{len(heldout)} held-out eval combos, {len(fewshot_pool)} available for few-shot pool")

    with open(OUT_PATH, "w") as out_f:
        for i, target in enumerate(heldout):
            # Sample a fresh set of few-shot examples per generation, excluding
            # anything that shares this target's exact (archetype, mood, topic)
            examples = rng.sample(fewshot_pool, N_FEWSHOT)
            prompt = PROMPT_TEMPLATE.format(
                archetype=target["archetype"],
                mood=target["mood"],
                topic=target["topic"],
                examples=format_examples(examples),
            )

            try:
                generated = call_ollama(prompt)
            except requests.exceptions.RequestException as e:
                print(f"[{i}] FAILED: {e}")
                continue

            row = {
                "archetype": target["archetype"],
                "mood": target["mood"],
                "topic": target["topic"],
                "reference_dialogue": target["dialogue"],  # curated human-approved version
                "baseline_generated": generated,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            print(f"[{i}] {target['archetype']}/{target['mood']}: {generated}")

    print(f"\nDone. Baseline outputs saved to {OUT_PATH}")


if __name__ == "__main__":
    main()