# Game Dialogue LoRA Fine-tune

## Problem
Generating in-character NPC dialogue from structured input 
(archetype, mood, topic) is a common need in game content pipelines. 
This project fine-tunes a small open LLM (Llama 3.1 8B) via LoRA to 
generate dialogue conditioned on this structured input, and compares 
it against a few-shot prompted baseline.

## Status: In Progress

## Steps
- [x] Define structured input/output schema (`docs/schema.md`)
- [x] Generate seed data via LLM-assisted synthesis (`src/generate_seed_data.py`)
- [x] Manually curate seed data
- [x] Build few-shot baseline (`src/baseline_prompt.py`)
- [x] Define evaluation methodology
- [ ] Fine-tune Llama 3.1 8B with LoRA/PEFT (`src/train_lora.py`)
- [ ] Run evaluation on fine-tuned vs baseline outputs
- [ ] Write up results and limitations

## Evaluation
Fine-tuned and baseline outputs are compared on two legs.

**Rule-based checks** (`src/eval_rules.py`), deterministic and automatic:
- word count compliance (20 words or fewer)
- no narration leakage into the spoken line
- memorisation check, flagging generated lines that are near duplicates of a training example

**Blind human-rated rubric** (`src/make_blind_rating_sheet.py`), rated by hand on a 1 to 5 scale:
- archetype fit
- mood fit
- naturalness

Baseline and fine-tuned outputs are pooled and shuffled before rating, with the model identity kept in a separate key file, so scoring is not biased by knowing which model produced which line.

## Setup

```bash
git clone https://github.com/ada-th25/game-dialogue-lora.git
cd game-dialogue-lora
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) running locally with `llama3.1:8b` pulled.

## Data
Seed data is LLM-generated then manually curated, not scraped 
from real game chat/dialogue. This is noted explicitly as a 
limitation in the writeup.

## Limitations
(to be filled in as the project develops)