# -*- coding: utf-8 -*-
"""
finetune.py — Fine-tune Qwen3.5-0.8B on Search Query Structured Extraction
===========================================================================
Trains one LoRA adapter per output format (JSON, YAML, TOML, CSV, XML),
then pushes each adapter to the Hugging Face Hub.

Usage:
    python finetune.py
"""

# ──────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ──────────────────────────────────────────────────────────────

DATASET_REPO = "sarthakrastogi/search-expert"
HF_USERNAME = "sarthakrastogi"
PUSH_TO_HUB = True

MAX_SEQ_LENGTH = 512
LOAD_IN_4BIT = True
LORA_R = 16
BATCH_SIZE = 8
GRAD_ACCUM = 2  # effective batch = 16
LEARNING_RATE = 2e-4

ALL_FORMATS = [
    "json_output",
    "yaml_output",
    # "toml_output",
    # "csv_output",
    # "xml_output",
]

MODEL_NAMES = {
    "json_output": "search-expert-json-0.8b",
    "yaml_output": "search-expert-yaml-0.8b",
    "toml_output": "search-expert-toml-0.8b",
    "csv_output": "search-expert-csv-0.8b",
    "xml_output": "search-expert-xml-0.8b",
}

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────

import gc
import time

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from prompts import FORMAT_LABELS, make_training_prompt

# ──────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────


def train_format(fmt: str, raw_train, hf_token: str) -> None:
    label = FORMAT_LABELS[fmt]
    adapter_name = MODEL_NAMES[fmt]

    # ── Load a fresh base model + LoRA ────────────────────────
    print("  Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-0.8B",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_R,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # ── Format training data ───────────────────────────────────
    eos = tokenizer.eos_token

    def format_row(batch, _fmt=fmt, _eos=eos):
        return {
            "text": [
                make_training_prompt(q, o, _fmt, _eos)
                for q, o in zip(batch["query"], batch[_fmt])
            ]
        }

    train_dataset = raw_train.map(
        format_row,
        batched=True,
        remove_columns=raw_train.column_names,
        desc=f"Formatting [{label}]",
    )

    # ── Train ──────────────────────────────────────────────────
    FastLanguageModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=100,
            max_steps=300,
            learning_rate=LEARNING_RATE,
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=f"outputs/{adapter_name}",
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            packing=True,
        ),
    )

    gpu = torch.cuda.get_device_properties(0)
    vram = round(gpu.total_memory / 1024**3, 2)
    print(f"  Training on {len(train_dataset):,} rows | GPU: {gpu.name} ({vram} GB)")

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    peak_vram = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    print(f"  ✓ Trained in {train_secs/60:.1f} min | Peak VRAM: {peak_vram} GB")

    # ── Save locally ───────────────────────────────────────────
    model.save_pretrained(adapter_name)
    tokenizer.save_pretrained(adapter_name)
    print(f"  ✓ Adapter saved → ./{adapter_name}/")

    # ── Push to Hub ────────────────────────────────────────────
    if PUSH_TO_HUB:
        try:
            model.push_to_hub(f"{HF_USERNAME}/{adapter_name}", token=hf_token)
            tokenizer.push_to_hub(f"{HF_USERNAME}/{adapter_name}", token=hf_token)
            print(f"  ✓ Pushed → huggingface.co/{HF_USERNAME}/{adapter_name}")
        except Exception as e:
            print(f"  ⚠ Hub push failed: {e}")

    # ── Free VRAM ──────────────────────────────────────────────
    del model, tokenizer, trainer, train_dataset
    gc.collect()
    torch.cuda.empty_cache()
    print("  VRAM cleared.")


def main():
    try:
        from google.colab import userdata

        hf_token = userdata.get("HF_TOKEN")
        print("TOKEN", hf_token)
    except Exception:
        import os

        hf_token = os.environ.get("HF_TOKEN", "")

    print("Loading dataset...")
    raw_train = load_dataset(DATASET_REPO, split="train", token=hf_token)
    print(f"  Train: {len(raw_train):,} rows")

    for i, fmt in enumerate(ALL_FORMATS):
        label = FORMAT_LABELS[fmt]
        print(f"\n{'='*62}")
        print(f"  FORMAT {i+1}/{len(ALL_FORMATS)}: {label}")
        print(f"{'='*62}")
        train_format(fmt, raw_train, hf_token)

    print("\n✅ All formats trained and saved.")


if __name__ == "__main__":
    main()
