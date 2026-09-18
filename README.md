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
- [ ] Manually curate seed data
- [ ] Build few-shot baseline (`src/baseline_prompt.py`)
- [ ] Fine-tune Llama 3.1 8B with LoRA/PEFT
- [ ] Evaluate fine-tuned vs baseline
- [ ] Write up results and limitations

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
Seed data is LLM-generated then manually curated — not scraped 
from real game chat/dialogue. This is noted explicitly as a 
limitation in the writeup.

## Limitations
(to be filled in as the project develops)