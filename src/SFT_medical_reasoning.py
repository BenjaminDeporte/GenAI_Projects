#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from time import perf_counter


HF_DATASET_ID = "FreedomIntelligence/medical-o1-reasoning-SFT"
HF_CONFIG = "en"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

N_MAX = 1000 # a subset of the dataset to validate pipeline
# N_MAX = 19704 # length of the medical-o1-reasoning-SFT
SPLIT_SEED = 42
VAL_RATIO = 0.1
N_EPOCHS = 3

RANG = 4  # rank for LoRA
N_SAMPLES = 3  # samples for tests
SAMPLE_SEED = 123
MAX_NEW_TOKENS = 2048  # length of generation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2_5_3b_instruct"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "outputs" / "sft-medical"
FINAL_MODEL_DIR = OUTPUT_DIR / "final-model"


def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["Question"]}],
        "completion": [
            {
                "role": "assistant",
                "content": f"<think>{example['Complex_CoT']}</think>{example['Response']}",
            }
        ],
    }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def preprocess_logits_for_metrics(logits, labels):
    del labels
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds[:, :-1].reshape(-1)
    labels = labels[:, 1:].reshape(-1)

    mask = labels != -100
    if mask.sum() == 0:
        return {"token_accuracy": 0.0}

    acc = (preds[mask] == labels[mask]).astype(np.float32).mean().item()
    return {"token_accuracy": acc}


def parse_completion(completion_text):
    """
    Parse completion text to extract reasoning and final answer.
    Format: <think>REASONING</think>FINAL_ANSWER

    Returns: (reasoning_text, final_answer_text)
    """
    if "<think>" in completion_text and "</think>" in completion_text:
        start_idx = completion_text.find("<think>") + len("<think>")
        end_idx = completion_text.find("</think>")
        reasoning = completion_text[start_idx:end_idx].strip()
        final_answer = completion_text[end_idx + len("</think>") :].strip()
    else:
        reasoning = ""
        final_answer = completion_text.strip()

    return reasoning, final_answer

def seconds_to_hms(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_is_empty = not any(DATASET_DIR.iterdir())
    if dataset_is_empty:
        print(f"{DATASET_DIR} is empty -> downloading from Hub and saving to disk...")
        dataset = load_dataset(HF_DATASET_ID, HF_CONFIG)
        dataset.save_to_disk(str(DATASET_DIR))
    else:
        print(f"{DATASET_DIR} is not empty -> loading from disk...")
        dataset = load_from_disk(str(DATASET_DIR))

    print(dataset)

    dataset = dataset.map(
        preprocess_function,
        remove_columns=["Question", "Response", "Complex_CoT"],
    )
    pprint(next(iter(dataset["train"])))

    base_train = dataset["train"]

    if N_MAX <= 0:
        raise ValueError("N_MAX must be > 0")

    n_subset = min(N_MAX, len(base_train))
    subset_train = base_train.shuffle(seed=SPLIT_SEED).select(range(n_subset))
    split = subset_train.train_test_split(
        test_size=VAL_RATIO,
        seed=SPLIT_SEED,
        shuffle=True,
    )

    train_ds = split["train"]
    val_ds = split["test"]

    print(f"Original train size: {len(base_train)}")
    print(f"Subset size:         {len(subset_train)}")
    print(f"Train size:          {len(train_ds)}")
    print(f"Validation size:     {len(val_ds)}")

    model_is_empty = not any(MODEL_DIR.iterdir())
    if model_is_empty:
        print(f"{MODEL_DIR} is empty -> downloading model/tokenizer from Hub and saving to disk...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer.save_pretrained(str(MODEL_DIR))
        model.save_pretrained(str(MODEL_DIR))
    else:
        print(f"{MODEL_DIR} is not empty -> loading model/tokenizer from disk...")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    full_trainable, _ = count_params(model)
    print(f"Full FT trainable: {full_trainable:,}")

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=RANG,
        lora_alpha=2 * RANG,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    lora_model = get_peft_model(model, peft_config)

    lora_trainable, lora_total = count_params(lora_model)
    reduction_vs_full_ft = 100 * (1 - (lora_trainable / full_trainable))
    trainable_pct_of_total = 100 * (lora_trainable / lora_total)

    print(f"Full FT trainable: {full_trainable:,}")
    print(f"LoRA trainable:    {lora_trainable:,}")
    print(f"Reduction:         {reduction_vs_full_ft:.2f}%")
    print(f"LoRA trainable %:  {trainable_pct_of_total:.4f}%")

    sft_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=N_EPOCHS,
        logging_strategy="epoch",
        logging_dir="./logs",
        save_steps=100,
        gradient_checkpointing=True,
        bf16=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_accumulation_steps=8,
    )

    trainer = SFTTrainer(
        model=lora_model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print(f"\nTraining...\n")
    
    t0 = perf_counter()
    trainer.train()
    elapsed = perf_counter() - t0

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
    
    print(f"\nTrained {N_EPOCHS} epochs over dataset of length {N_MAX} in:", seconds_to_hms(elapsed),"\n")
    print(f"Saved trained model artifacts to: {FINAL_MODEL_DIR}\n")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_for_eval = trainer.model
    model_for_eval.eval()

    rng = np.random.default_rng(SAMPLE_SEED)
    n_eval = min(N_SAMPLES, len(val_ds))
    sample_indices = rng.choice(len(val_ds), size=n_eval, replace=False).tolist()

    print(f"Sampling {n_eval} examples from validation set of size {len(val_ds)}")

    results = []
    for idx in sample_indices:
        ex = val_ds[int(idx)]

        messages = ex["prompt"]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_for_eval.device)

        with torch.no_grad():
            generated = model_for_eval.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = generated[0][prompt_len:]
        pred_completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        gt_full = ex["completion"][0]["content"]
        gt_reasoning, gt_final_answer = parse_completion(gt_full)

        pred_reasoning, pred_final_answer = parse_completion(pred_completion)

        results.append(
            {
                "idx": int(idx),
                "prompt": messages[0]["content"],
                "gt_reasoning": gt_reasoning,
                "gt_final_answer": gt_final_answer,
                "pred_reasoning": pred_reasoning,
                "pred_final_answer": pred_final_answer,
                "gt_full": gt_full,
                "pred_full": pred_completion,
            }
        )

    print(f"Built {len(results)} comparisons with parsed components.")

    preview_df = pd.DataFrame(results)[
        [
            "idx",
            "prompt",
            "gt_reasoning",
            "gt_final_answer",
            "pred_reasoning",
            "pred_final_answer",
        ]
    ].copy()

    preview_df["prompt"] = preview_df["prompt"].str.slice(0, 80) + "..."
    preview_df["gt_reasoning"] = preview_df["gt_reasoning"].str.slice(0, 100) + "..."
    preview_df["gt_final_answer"] = preview_df["gt_final_answer"].str.slice(0, 100) + "..."
    preview_df["pred_reasoning"] = preview_df["pred_reasoning"].str.slice(0, 100) + "..."
    preview_df["pred_final_answer"] = preview_df["pred_final_answer"].str.slice(0, 100) + "..."

    print(preview_df)

    for i, r in enumerate(results, start=1):
        print("=" * 120)
        print(f"Sample {i} | Validation index: {r['idx']}")
        print("\n[PROMPT]")
        print(r["prompt"])
        print("\n[GROUND TRUTH - Reasoning]")
        print(r["gt_reasoning"] if r["gt_reasoning"] else "(no reasoning tags)")
        print("\n[GROUND TRUTH - Final Answer]")
        print(r["gt_final_answer"])
        print("\n[PREDICTED - Reasoning]")
        print(r["pred_reasoning"] if r["pred_reasoning"] else "(no reasoning tags)")
        print("\n[PREDICTED - Final Answer]")
        print(r["pred_final_answer"])
        print()


if __name__ == "__main__":
    main()
