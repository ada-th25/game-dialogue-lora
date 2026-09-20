"""
LoRA fine-tune of Llama 3.1 8B on curated NPC dialogue data.

Uses the SAME held-out split as baseline_prompt.py (seed=42, N_HELDOUT=20)
so the eval set is identical across baseline / fine-tuned / reference —
this is what makes the three-way comparison fair.

NOTE ON COMPUTE: this needs a CUDA GPU for 4-bit quantized training in a
reasonable time. If you're on a Mac (no CUDA), don't try to run this
locally — push the repo, then run this script on Colab (free/paid T4 or
better) or a cheap cloud GPU instance. Ollama is for inference only; it
isn't used for training.

Run (on a CUDA machine):
    python src/train_lora.py
"""

import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

BASE_MODEL = "meta-llama/Llama-3.1-8B"  # requires HF access request/approval
CURATED_PATH = Path("data/seed_curated.jsonl")
OUTPUT_DIR = Path("checkpoints/lora-dialogue")

N_HELDOUT = 20          # must match baseline_prompt.py
RANDOM_SEED = 42        # must match baseline_prompt.py

# Same instruction format as baseline, but WITHOUT few-shot examples —
# the point of fine-tuning is that the model no longer needs them.
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


def make_split(curated):
    """Reproduces the exact same held-out split as baseline_prompt.py."""
    rng = random.Random(RANDOM_SEED)
    shuffled = curated[:]
    rng.shuffle(shuffled)
    heldout = shuffled[:N_HELDOUT]
    train_pool = shuffled[N_HELDOUT:]
    return train_pool, heldout


def format_example(row):
    prompt = PROMPT_TEMPLATE.format(
        archetype=row["archetype"], mood=row["mood"], topic=row["topic"]
    )
    # Full text = prompt + target, used for causal LM training
    full_text = f"{prompt} {row['dialogue']}"
    return {"text": full_text, "prompt": prompt}


def tokenize_fn(tokenizer, example):
    # Tokenize full text; loss is computed over the whole sequence for
    # simplicity. For cleaner results later, consider masking the prompt
    # tokens (label = -100) so loss only counts the completion.
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():
    curated = load_curated(CURATED_PATH)
    train_pool, heldout = make_split(curated)
    print(f"{len(train_pool)} training examples, {len(heldout)} held out (matches baseline eval set)")

    formatted = [format_example(row) for row in train_pool]
    dataset = Dataset.from_list(formatted)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda ex: tokenize_fn(tokenizer, ex), remove_columns=dataset.column_names)

    # 4-bit quantization to fit an 8B model in a single consumer/Colab GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nDone. LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()