#!/usr/bin/env python3
import os
import yaml
import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
# Ensure PatchDPOTrainer is applied before imports
PatchDPOTrainer()
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, DPOTrainer
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Hub token
    hf_token = cfg.get("hub_token") or os.environ.get("HUGGINGFACE_TOKEN")

    # --- Dataset loading ---
    ds_cfg = cfg["datasets"][0]
    ds_type = ds_cfg.get("ds_type", ds_cfg.get("type", "hf"))
    if ds_type.lower() in ("json", "csv", "text"):  # local files
        path = ds_cfg["path"]
        dataset = load_dataset(ds_type.lower(), data_files={"train": path}, split=ds_cfg.get("split", "train"))
    else:
        dataset = load_dataset(ds_cfg["path"], split=ds_cfg.get("split", "train"))

    # --- Train/Validation split if requested ---
    val_size = cfg.get("val_set_size", 0)
    if isinstance(val_size, (int, float)) and val_size > 0:
        splits = dataset.shuffle(seed=cfg.get("seed", 42)).train_test_split(test_size=val_size)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]
        print(f"Split data: {len(train_dataset)} train / {len(eval_dataset)} eval (val_set_size={val_size})")
    else:
        train_dataset = dataset
        eval_dataset = None

    # --- Model & tokenizer ---
    # Pin model to each GPU when using accelerate
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg.get("sequence_len", 2048),
        dtype=None,
        token=hf_token,
        device_map=device_map,
    )

    # --- Optional LoRA / PEFT ---
    if cfg.get("adapter") == "lora":
        r = cfg.get("lora_r", 16)
        alpha = cfg.get("lora_alpha", 16)
        dropout = cfg.get("lora_dropout", 0.0)
        target_modules = cfg.get("target_modules") or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        use_gc = "unsloth" if cfg.get("gradient_checkpointing", False) else False
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            use_gradient_checkpointing=use_gc,
            max_seq_length=cfg.get("sequence_len", 2048),
        )

    # --- Prepare TrainingArguments ---
    lr = cfg.get("learning_rate", 5e-5)
    if isinstance(lr, str): lr = float(lr)
    training_args = TrainingArguments(
        output_dir="/workspace/outputs",
        per_device_train_batch_size=cfg.get("micro_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=cfg.get("num_epochs", 1),
        learning_rate=lr,
        optim=cfg.get("optimizer", "adamw_torch_fused"),
        warmup_steps=cfg.get("warmup_steps", 0),
        max_steps=cfg.get("max_steps", None),
        logging_steps=cfg.get("logging_steps", 100),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=cfg.get("eval_steps", None),
        save_steps=cfg.get("save_steps", None),
        save_total_limit=cfg.get("save_total_limit", None),
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", False)),
        metric_for_best_model=cfg.get("metric_for_best_model", "loss"),
        greater_is_better=bool(cfg.get("greater_is_better", False)),
        weight_decay=cfg.get("weight_decay", 0.0),
        bf16=(cfg.get("bf16") in (True, "auto") and torch.cuda.is_bf16_supported()),
        fp16=bool(cfg.get("fp16", False)),
    )

    # --- Early Stopping Callback ---
    callbacks = []
    if cfg.get("early_stopping", False):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.get("early_stopping_patience", 1),
            )
        )

    # --- Determine which column holds the text ---
    text_field = ds_cfg.get("text_field")
    if text_field is None:
        features = train_dataset.features
        string_cols = [k for k, v in features.items() if getattr(v, "dtype", None) == "string"]
        if not string_cols:
            raise ValueError(
                "Could not infer text_field: no string-typed column found in dataset.features "
                f"{list(features.keys())}"
            )
        text_field = string_cols[0]
        print(f"[Warning] No text_field in config; using '{text_field}' from dataset columns")

    # --- Choose trainer: DPO vs SFT ---
    if cfg.get("rl", "").lower() == "dpo":
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            beta=cfg.get("beta", 0.1),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=cfg.get("sequence_len", 2048),
            max_prompt_length=cfg.get("sequence_len", 2048),
        )
    else:
        trainer_kwargs = dict(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field=text_field,
            max_seq_length=cfg.get("sequence_len", 2048),
            args=training_args,
            callbacks=callbacks,
        )
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

if __name__ == "__main__":
    main()
