#!/usr/bin/env python3
import os
import yaml
import argparse
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p.parse_args()

def main():
    args = parse_args()
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Hugging Face login
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("HF CLI logged in.")

    # W&B login
    wandb_token = os.environ.get("WANDB_TOKEN")
    if wandb_token:
        import wandb; wandb.login(key=wandb_token)
        print("W&B logged in.")

    # Dataset loading
    ds_cfg = cfg.get("dataset", {})
    if ds_cfg.get("type", "hf") == "hf":
        dataset = load_dataset(ds_cfg["path"], split=ds_cfg.get("split", "train"))
    else:
        filename = os.environ["DATASET_FILENAME"]
        path = f"/workspace/input_data/{filename}"
        fmt = ds_cfg.get("format", "json")
        dataset = load_dataset(fmt, data_files={"train": path})

    # Model + tokenizer
    mcfg = cfg["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg["name"],
        max_seq_length=mcfg.get("max_seq_length", 2048),
        dtype=mcfg.get("dtype"),
        load_4bit=mcfg.get("load_in_4bit", False),
        token=hf_token,
    )

    # Optional LoRA / PEFT
    peft_cfg = cfg.get("peft", {})
    if peft_cfg:
        model = FastLanguageModel.get_peft_model(
            model,
            r=peft_cfg.get("r", 16),
            target_modules=peft_cfg.get("target_modules", []),
            lora_alpha=peft_cfg.get("lora_alpha", 16),
            lora_dropout=peft_cfg.get("lora_dropout", 0.0),
            bias=peft_cfg.get("bias", "none"),
            use_gradient_checkpointing=peft_cfg.get("gradient_checkpointing", False),
        )

    # Training arguments
    ta = cfg.get("training_args", {})
    training_args = TrainingArguments(
        output_dir="/workspace/outputs",
        per_device_train_batch_size=ta.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=ta.get("gradient_accumulation_steps", 1),
        max_steps=ta.get("max_steps"),
        num_train_epochs=ta.get("num_train_epochs", 1),
        learning_rate=ta.get("learning_rate", 5e-5),
        optim=ta.get("optim", "adamw_torch_fused"),
        bf16=ta.get("bf16", False),
        fp16=ta.get("fp16", True),
        logging_steps=ta.get("logging_steps", 100),
        save_steps=ta.get("save_steps", None),
        save_total_limit=ta.get("save_total_limit", 1),
    )
    
    # Set up EarlyStoppingCallback if enabled
    callbacks = []
    es_cfg = cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=es_cfg.get("patience", 1),
                early_stopping_threshold=es_cfg.get("threshold", 0.0),
                early_stopping_threshold_mode=es_cfg.get("threshold_mode", "rel"),
            )
        )

    # Trainer init & train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=ds_cfg.get("text_field", "text"),
        max_seq_length=mcfg.get("max_seq_length", 2048),
        args=training_args,
        callbacks=callbacks,
    )
    trainer.train()

if __name__ == "__main__":
    main()
