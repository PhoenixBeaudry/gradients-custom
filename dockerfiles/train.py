#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
from ignite.engine import Engine, Events
from ignite.handlers.lr_finder import FastaiLRFinder
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator
import wandb
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
import time
from transformers import TrainerCallback, TrainerControl, TrainerState
import bitsandbytes as bnb

# Optional imports for LoRA adapters and DPO
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = get_peft_model = prepare_model_for_kbit_training = None

try:
    from trl import DPOTrainer, DPOConfig
except ImportError:
    DPOTrainer = DPOConfig = None

# Disable parallel tokenizer threads to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


###### Custom Callbacks #####
class TimeLimitCallback(TrainerCallback):
    """Stop training after a fixed number of hours."""

    def __init__(self, max_hours: float):
        """
        Args:
            max_hours: training time budget in hours
        """
        self.max_seconds = max_hours * 3600.0 * 0.95
        self.start_time: float | None = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # record the training start time
        self.start_time = time.time()
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # only check once we've started
        if self.start_time is None:
            return control
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            print(f"\n⏱️  Reached time limit of {self.max_seconds/3600:.2f}h — stopping training.")
            control.should_training_stop = True
        return control


def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal LM with SFT or DPO")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    return logging.getLogger(__name__)


def prepare_tokenizer(model_name: str, hub_token: str = None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, use_auth_token=hub_token, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, cfg: dict) -> AutoModelForCausalLM:
    common_kwargs = {
        'use_auth_token': cfg.get('hub_token'),
        'load_in_8bit': bool(cfg.get('load_in_8bit', False)),
        'torch_dtype': torch.bfloat16 if cfg.get('bf16') and torch.cuda.is_bf16_supported() else None,
    }
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation='flash_attention_2',
            **common_kwargs
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)


def apply_lora_adapter(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
    if get_peft_model is None:
        raise ImportError("peft library is required for LoRA adapters.")

    if cfg.get('load_in_8bit', False):
        model = prepare_model_for_kbit_training(model)

    # Determine target modules for LoRA
    targets = cfg.get('target_modules') or []
    if not targets:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(x in name.lower() for x in ('attn', 'attention')):
                targets.append(name.split('.')[-1])
        targets = list(set(targets))
        if not targets:
            raise ValueError("Could not auto-detect attention modules for LoRA. Please set 'target_modules' in config.")

    peft_config = LoraConfig(
        r=int(cfg.get('lora_r', 16)),
        lora_alpha=int(cfg.get('lora_alpha', 16)),
        target_modules=targets,
        lora_dropout=float(cfg.get('lora_dropout', 0.0)),
        bias='none',
        task_type='CAUSAL_LM'
    )
    return get_peft_model(model, peft_config)


def load_sft_datasets(cfg: dict, tokenizer: AutoTokenizer):
    ds_cfg = cfg['datasets'][0]
    ds_type = ds_cfg.get('ds_type', ds_cfg.get('type', 'hf')).lower()
    if ds_type in ('json', 'csv', 'text'):
        raw = load_dataset(ds_type, data_files={'train': ds_cfg['path']}, split=ds_cfg.get('split', 'train'))
    else:
        raw = load_dataset(ds_cfg['path'], split=ds_cfg.get('split', 'train'))

    val_size = cfg.get('val_set_size', 0)
    if val_size > 0:
        splits = raw.shuffle(seed=cfg.get('seed', 42)).train_test_split(test_size=val_size)
        train_ds, eval_ds = splits['train'], splits['test']
    else:
        train_ds, eval_ds = raw, None

    # Identify text column
    text_col = ds_cfg.get('text_field')
    if not text_col:
        text_cols = [k for k, v in train_ds.features.items() if getattr(v, 'dtype', None) == 'string']
        text_col = text_cols[0] if text_cols else None
        if not text_col:
            raise ValueError("No string column found for SFT tokenization.")

    def tokenize_fn(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=int(cfg.get('sequence_len', 2048)))

    train_ds = train_ds.map(tokenize_fn, batched=True)
    if eval_ds:
        eval_ds = eval_ds.map(tokenize_fn, batched=True)

    return train_ds, eval_ds


def load_dpo_datasets(cfg: dict, tokenizer: AutoTokenizer):
    ds_cfg = cfg['datasets'][0]
    ds_type = ds_cfg.get('ds_type', ds_cfg.get('type', 'hf')).lower()
    if ds_type in ('json', 'csv', 'text'):
        raw = load_dataset(ds_type, data_files={'train': ds_cfg['path']}, split=ds_cfg.get('split', 'train'))
    else:
        raw = load_dataset(ds_cfg['path'], split=ds_cfg.get('split', 'train'))

    cols = raw.column_names
    if 'prompt' not in cols and len(cols) > 0:
        raw = raw.rename_column(cols[0], 'prompt')
    if 'chosen' not in cols and len(cols) > 1:
        raw = raw.rename_column(cols[1], 'chosen')
    if 'rejected' not in cols and len(cols) > 2:
        raw = raw.rename_column(cols[2], 'rejected')

    val_size = cfg.get('val_set_size', 0)
    if val_size > 0:
        splits = raw.shuffle(seed=cfg.get('seed', 42)).train_test_split(test_size=val_size)
        train_ds, eval_ds = splits['train'], splits['test']
    else:
        train_ds, eval_ds = raw, None

    return train_ds, eval_ds


def build_trainer(cfg: dict, model, tokenizer, train_ds, eval_ds, callbacks):
    if cfg.get('rl', '').lower() == 'dpo':
        if DPOTrainer is None:
            raise ImportError("trl library is required for DPO training.")

        # ensure pad_token_id exists
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id

        dpo_args = DPOConfig(
            output_dir=cfg.get('output_dir', './outputs'),
            padding_value=pad_id,
            auto_find_batch_size=True,
            bf16=bool(cfg.get('bf16', False)),
            gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps', 2)),
            dataloader_num_workers=int(cfg.get('dataloader_num_workers', 8)),
            num_train_epochs=int(cfg.get('num_epochs', 1)),
            learning_rate=float(cfg.get('learning_rate', 5e-5))/3,
            optim=cfg.get('optimizer', 'lion_8bit'),
            warmup_steps=int(cfg.get('warmup_steps', 25)),
            lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
            max_steps=int(cfg.get('max_steps', -1)),
            logging_steps=int(cfg.get('logging_steps', 100)),
            eval_strategy='steps',
            save_strategy='best',
            eval_steps=int(cfg.get('eval_steps')) if cfg.get('eval_steps') is not None else None,
            save_steps=int(cfg.get('save_steps')) if cfg.get('save_steps') is not None else None,
            save_total_limit=int(cfg.get('save_total_limit')) if cfg.get('save_total_limit') is not None else None,
            load_best_model_at_end=True,
            metric_for_best_model=cfg.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=bool(cfg.get('greater_is_better', False)),
            weight_decay=float(cfg.get('weight_decay', 0.0)),
            fp16=bool(cfg.get('fp16', False)),
            logging_dir=cfg.get('logging_dir', './logs'),
            push_to_hub=True,
            run_name=cfg.get('wandb_run'),
            hub_model_id=cfg.get('hub_model_id'),
            hub_token=cfg.get('hub_token'),
            hub_strategy='every_save',
            use_liger_kernel=True,
        )
        logger = setup_logger()
        logger.info("Initializing DPO Trainer")

        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg['base_model'], use_auth_token=cfg.get('hub_token')
        )
        ref_model.eval()

        return DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

    # ── SFT Trainer branch ────────────────────────────────────────
    tf_args = TrainingArguments(
        output_dir=cfg.get('output_dir', './outputs'),
        auto_find_batch_size=True,
        bf16=bool(cfg.get('bf16', False)),
        gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps', 2)),
        dataloader_num_workers=int(cfg.get('dataloader_num_workers', 8)),
        num_train_epochs=int(cfg.get('num_epochs', 1)),
        learning_rate=float(cfg.get('learning_rate', 5e-5))/3,
        optim=cfg.get('optimizer', 'lion_8bit'),
        warmup_steps=int(cfg.get('warmup_steps', 25)),
        lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
        load_best_model_at_end=True,
        max_steps=int(cfg.get('max_steps', -1)),
        logging_steps=int(cfg.get('logging_steps', 100)),
        eval_strategy='steps' if eval_ds else 'no',
        save_strategy='best',
        eval_steps=int(cfg.get('eval_steps')) if cfg.get('eval_steps') is not None else None,
        save_steps=int(cfg.get('save_steps')) if cfg.get('save_steps') is not None else None,
        save_total_limit=int(cfg.get('save_total_limit')) if cfg.get('save_total_limit') is not None else None,
        metric_for_best_model=cfg.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=bool(cfg.get('greater_is_better', False)),
        weight_decay=float(cfg.get('weight_decay', 0.0)),
        fp16=bool(cfg.get('fp16', False)),
        logging_dir=cfg.get('logging_dir', './logs'),
        push_to_hub=True,
        run_name=cfg.get('wandb_run'),
        hub_model_id=cfg.get('hub_model_id'),
        hub_token=cfg.get('hub_token'),
        hub_strategy='every_save',
        use_liger_kernel=True,
    )
    logger = setup_logger()
    logger.info("Initializing SFT Trainer")

    return Trainer(
        model=model,
        args=tf_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
        callbacks=callbacks,
        processing_class=tokenizer,
    )



def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger()

    # Performance flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    logger.info("Loaded config from %s", args.config)
    accelerator = Accelerator(log_with="wandb", mixed_precision="bf16")
    accelerator.init_trackers(cfg.get('wandb_project'), config=cfg)

    tokenizer = prepare_tokenizer(cfg['base_model'], cfg.get('hub_token'))
    model = load_model(cfg['base_model'], cfg)

    if cfg.get('adapter') == 'lora':
        model = apply_lora_adapter(model, cfg)

    rl_mode = cfg.get('rl', '').lower() == 'dpo'
    load_fn = load_dpo_datasets if rl_mode else load_sft_datasets
    train_ds, eval_ds = load_fn(cfg, tokenizer)

    callbacks = []
    if cfg.get('early_stopping', True):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.get('early_stopping_patience', 8))
        )
    # add time‐limit
    max_hours = int(cfg.get('hours_to_complete'))  # e.g. 4.0
    if max_hours is not None:
        callbacks.append(TimeLimitCallback(max_hours))

    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds, callbacks)

    logger.info("Starting Full Model Training...")

    trainer = accelerator.prepare(trainer)
    trainer.train()



if __name__ == '__main__':
    main()
