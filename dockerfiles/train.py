#!/usr/bin/env python3
import os
import yaml
import argparse
import logging
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
import torch

# Optional: require peft for LoRA adapters
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

# Optional: require trl for DPO
try:
    from trl import DPOTrainer
except ImportError:
    DPOTrainer = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config"
    )
    return parser.parse_args()


def load_sft_datasets(cfg, tokenizer):
    ds_cfg = cfg["datasets"][0]
    ds_type = ds_cfg.get("ds_type", ds_cfg.get("type", "hf")).lower()
    if ds_type in ("json", "csv", "text"):
        raw = load_dataset(ds_type, data_files={"train": ds_cfg["path"]}, split=ds_cfg.get("split", "train"))
    else:
        raw = load_dataset(ds_cfg["path"], split=ds_cfg.get("split", "train"))
    val_size = cfg.get("val_set_size", 0)
    if val_size > 0:
        splits = raw.shuffle(seed=cfg.get("seed", 42)).train_test_split(test_size=val_size)
        train_ds, eval_ds = splits["train"], splits["test"]
    else:
        train_ds, eval_ds = raw, None
    text_field = ds_cfg.get("text_field")
    if text_field is None:
        cols = [k for k,v in train_ds.features.items() if getattr(v, 'dtype', None)=='string']
        text_field = cols[0] if cols else None
        if not text_field:
            raise ValueError("No string column found for SFT tokenization")
    def tok(batch): return tokenizer(batch[text_field], truncation=True, max_length=cfg.get("sequence_len",2048))
    train_ds = train_ds.map(tok, batched=True)
    if eval_ds:
        eval_ds = eval_ds.map(tok, batched=True)
    return train_ds, eval_ds


def load_dpo_datasets(cfg, tokenizer):
    ds_cfg = cfg["datasets"][0]
    raw = load_dataset(ds_cfg.get("path"), split=ds_cfg.get("split", "train"))
    def dpomap(ex):
        msgs = ex.get("messages") or ex.get("chat") or []
        query = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        replies = [m["content"] for m in msgs if m.get("role") == "assistant"]
        if len(replies) < 2:
            raise ValueError("Expected at least two assistant replies for DPO ChatML")
        chosen, rejected = replies[0], replies[1]
        q = tokenizer(query, truncation=True, max_length=cfg.get("sequence_len",2048))
        c = tokenizer(chosen, truncation=True, max_length=cfg.get("sequence_len",2048))
        r = tokenizer(rejected, truncation=True, max_length=cfg.get("sequence_len",2048))
        return {
            "input_ids": q["input_ids"], "attention_mask": q["attention_mask"],
            "chosen_input_ids": c["input_ids"], "chosen_attention_mask": c["attention_mask"],
            "rejected_input_ids": r["input_ids"], "rejected_attention_mask": r["attention_mask"],
        }
    train_ds = raw.map(dpomap, batched=False)
    return train_ds, None


def main():
    args = parse_args()
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded configuration from %s", args.config)

    # Initialize W&B
    wandb_project = cfg.get("wandb_project")
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=cfg.get("wandb_run_name"),
            config=cfg,
        )
        logger.info("Initialized W&B project %s", wandb_project)
    else:
        logger.info("No W&B project configured, skipping wandb.init()")

    model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=cfg.get("hub_token"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=cfg.get("hub_token"),
        load_in_8bit=bool(cfg.get("load_in_8bit", False)),
        torch_dtype=torch.bfloat16 if cfg.get("bf16") and torch.cuda.is_bf16_supported() else None,
        device_map="auto" if torch.cuda.device_count()>1 else None,
    )
    if cfg.get("adapter")=="lora":
        if get_peft_model is None:
            raise ImportError("peft required for LoRA")
        if cfg.get("load_in_8bit"):
            model = prepare_model_for_kbit_training(model)
        peft_cfg = LoraConfig(
            r=cfg.get("lora_r",16), lora_alpha=cfg.get("lora_alpha",16),
            target_modules=cfg.get("target_modules",["q_proj","k_proj","v_proj","o_proj"]),
            lora_dropout=cfg.get("lora_dropout",0.0), bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_cfg)

    rl_mode = cfg.get("rl","").lower()=="dpo"
    if rl_mode:
        train_ds, eval_ds = load_dpo_datasets(cfg, tokenizer)
    else:
        train_ds, eval_ds = load_sft_datasets(cfg, tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir","/workspace/outputs"),
        per_device_train_batch_size=cfg.get("micro_batch_size",4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps",1),
        num_train_epochs=cfg.get("num_epochs",1),
        learning_rate=float(cfg.get("learning_rate",5e-5)),
        optim=cfg.get("optimizer","adamw_torch_fused"),
        warmup_steps=cfg.get("warmup_steps",0),
        max_steps=cfg.get("max_steps",-1),
        logging_steps=cfg.get("logging_steps",100),
        eval_strategy="steps" if eval_ds else "no",
        save_strategy="steps", eval_steps=cfg.get("eval_steps"),
        save_steps=cfg.get("save_steps"), save_total_limit=cfg.get("save_total_limit"),
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end",False)),
        metric_for_best_model=cfg.get("metric_for_best_model","loss"),
        greater_is_better=bool(cfg.get("greater_is_better",False)),
        weight_decay=cfg.get("weight_decay",0.0), fp16=bool(cfg.get("fp16",False)),
        report_to=["wandb"] if wandb_project else [],
        logging_dir=cfg.get("logging_dir","./logs"),
        run_name=cfg.get("wandb_run_name"),
    )

    callbacks = []
    if cfg.get("early_stopping",False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stopping_patience",1)))

    if rl_mode:
        if DPOTrainer is None:
            raise ImportError("trl required for DPO training")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=cfg.get("hub_token"),
            device_map="auto" if torch.cuda.device_count()>1 else None,
        )
        ref_model.eval()
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            beta=cfg.get("beta",0.1),
            max_length=cfg.get("sequence_len",2048),
            max_prompt_length=cfg.get("sequence_len",2048),
            callbacks=callbacks,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=callbacks,
            tokenizer=tokenizer,
        )
    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
