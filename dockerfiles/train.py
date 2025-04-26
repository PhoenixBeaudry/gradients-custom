#!/usr/bin/env python3
import os
import yaml
import argparse
import logging
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    SchedulerType,
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
    from trl import DPOTrainer, DPOConfig
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
    ds_type = ds_cfg.get("ds_type", ds_cfg.get("type", "hf")).lower()
    if ds_type in ("json", "csv", "text"):
        raw = load_dataset(ds_type, data_files={"train": ds_cfg["path"]}, split=ds_cfg.get("split", "train"))
    else:
        raw = load_dataset(ds_cfg["path"], split=ds_cfg.get("split", "train"))
    #  ← ADD THIS: rename your CSV headers to what DPOTrainer expects:
    raw = raw.rename_columns({
        "gen_questions": "prompt",
        "Positive":      "chosen",
        "Hard Negative": "rejected",
    })

    # now map & tokenize exactly like your dpomap, and drop those original cols:
    def dpomap(ex):
        q = tokenizer(ex["prompt"],    truncation=True, max_length=cfg["sequence_len"])
        c = tokenizer(ex["chosen"],    truncation=True, max_length=cfg["sequence_len"])
        r = tokenizer(ex["rejected"],  truncation=True, max_length=cfg["sequence_len"])
        return {
            "input_ids":              q["input_ids"],
            "attention_mask":         q["attention_mask"],
            "chosen_input_ids":       c["input_ids"],
            "chosen_attention_mask":  c["attention_mask"],
            "rejected_input_ids":     r["input_ids"],
            "rejected_attention_mask":r["attention_mask"],
        }

    train_ds = raw.map(dpomap,
                       batched=False,
                       remove_columns=["prompt", "chosen", "rejected"])
    return train_ds, None


def main():
    
    args = parse_args()
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded configuration from %s", args.config)

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(cfg.get("wandb_project"), config=cfg)
    model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=cfg.get("hub_token"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        use_auth_token=cfg.get("hub_token"),
        load_in_8bit=bool(cfg.get("load_in_8bit", False)),
        torch_dtype=torch.bfloat16 if cfg.get("bf16") and torch.cuda.is_bf16_supported() else None,
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

    

    callbacks = []
    if cfg.get("early_stopping",False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.get("early_stopping_patience",1)))

    if rl_mode:
        training_args = DPOConfig(
        output_dir=cfg.get("output_dir","/workspace/outputs"),
        per_device_train_batch_size=cfg.get("micro_batch_size",4),
        auto_find_batch_size=True,
        bf16=True,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps",1),
        dataloader_num_workers=8,
        num_train_epochs=cfg.get("num_epochs",1),
        learning_rate=float(cfg.get("learning_rate",5e-5)),
        optim=cfg.get("optimizer","adamw_torch_fused"),
         # warm up the first 500 steps by default (≈1% of most jobs)
        warmup_steps=cfg.get("warmup_steps", 25),
        # use cosine decay after warmup
        lr_scheduler_type=cfg.get("lr_scheduler_type", SchedulerType.COSINE_WITH_RESTARTS),
        max_steps=cfg.get("max_steps",-1),
        logging_steps=cfg.get("logging_steps",100),
        eval_strategy="steps" if eval_ds else "no",
        save_strategy="best", eval_steps=cfg.get("eval_steps"),
        save_steps=cfg.get("save_steps"), save_total_limit=cfg.get("save_total_limit"),
        load_best_model_at_end=False,
        metric_for_best_model=cfg.get("metric_for_best_model","loss"),
        greater_is_better=bool(cfg.get("greater_is_better",False)),
        weight_decay=cfg.get("weight_decay",0.0), fp16=bool(cfg.get("fp16",False)),
        logging_dir=cfg.get("logging_dir","./logs"),
        push_to_hub=True,
        run_name=cfg.get("wandb_run"),
        hub_model_id=cfg.get("hub_model_id"),
        hub_token=cfg.get("hub_token"),
        hub_strategy="every_save",
        use_liger_kernel=True,
    )
        logger.info("Loading DPO Trainer")
        if DPOTrainer is None:
            raise ImportError("trl required for DPO training")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=cfg.get("hub_token"),
        )
        ref_model.eval()
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
    else:
        training_args = TrainingArguments(
        output_dir=cfg.get("output_dir","/workspace/outputs"),
        per_device_train_batch_size=cfg.get("micro_batch_size",4),
        auto_find_batch_size=True,
        bf16=True,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps",1),
        dataloader_num_workers=8,
        num_train_epochs=cfg.get("num_epochs",1),
        learning_rate=float(cfg.get("learning_rate",5e-5)),
        optim=cfg.get("optimizer","adamw_torch_fused"),
         # warm up the first 500 steps by default (≈1% of most jobs)
        warmup_steps=cfg.get("warmup_steps", 25),
        # use cosine decay after warmup
        lr_scheduler_type=cfg.get("lr_scheduler_type", SchedulerType.COSINE_WITH_RESTARTS),
        max_steps=cfg.get("max_steps",-1),
        logging_steps=cfg.get("logging_steps",100),
        eval_strategy="steps" if eval_ds else "no",
        save_strategy="best", eval_steps=cfg.get("eval_steps"),
        save_steps=cfg.get("save_steps"), save_total_limit=cfg.get("save_total_limit"),
        load_best_model_at_end=False,
        metric_for_best_model=cfg.get("metric_for_best_model","loss"),
        greater_is_better=bool(cfg.get("greater_is_better",False)),
        weight_decay=cfg.get("weight_decay",0.0), fp16=bool(cfg.get("fp16",False)),
        logging_dir=cfg.get("logging_dir","./logs"),
        push_to_hub=True,
        run_name=cfg.get("wandb_run"),
        hub_model_id=cfg.get("hub_model_id"),
        hub_token=cfg.get("hub_token"),
        hub_strategy="every_save",
        use_liger_kernel=True,
        )
        logger.info("Loading SFT Trainer")
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            callbacks=callbacks,
            processing_class=tokenizer,
        )
    logger.info("Starting training...")
    accelerate_trainer = accelerator.prepare(trainer)
    accelerate_trainer.train()


if __name__ == "__main__":
    main()
