#!/usr/bin/env python3
import os
import argparse
import logging
import yaml
from ignite.engine import create_supervised_trainer
from ignite.handlers.lr_finder import FastaiLRFinder
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
        model_name, use_fast=True, use_auth_token=hub_token
    )
    if "Qwen" in model_name:
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

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
        r=cfg.get('lora_r', 16),
        lora_alpha=cfg.get('lora_alpha', 16),
        target_modules=targets,
        lora_dropout=cfg.get('lora_dropout', 0.0),
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
        return tokenizer(batch[text_col], truncation=True, max_length=cfg.get('sequence_len', 2048))

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

def find_lr(cfg, model, train_ds, tokenizer, accelerator):
    # build a small DataLoader for LR search
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.get('micro_batch_size', 4)),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(cfg.get('dataloader_num_workers', 8)),
        pin_memory=True,
    )

    # move model to device
    device = accelerator.device
    model.to(device)

    # optimizer for finder (start very small)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr_finder_start', 1e-7))

    # make a no-op loss (we only need backward())
    def _update(engine, batch):
        model.train()
        inputs = {k: v.to(device) for k, v in batch.items() if k in ('input_ids', 'attention_mask')}
        loss = model(**inputs, labels=inputs['input_ids']).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = create_supervised_trainer(model, optimizer, None, device=device, update_function=_update)

    # attach the finder
    lr_finder = FastaiLRFinder()
    lr_finder.attach(trainer, optimizer, start_lr=cfg.get('lr_finder_start', 1e-7),
                     end_lr=cfg.get('lr_finder_end', 10), num_iter=cfg.get('lr_finder_steps', 100))

    # run only a handful of iterations
    trainer.run(train_loader, max_epochs=1)

    # pick the LR at minimum loss slope (or “steep” region)
    suggested = lr_finder.suggested_lr()
    accelerator.print(f"🔍 LR finder suggests: {suggested:.2e}")
    return suggested


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
            per_device_train_batch_size=int(cfg.get('micro_batch_size', 4)),
            padding_value=pad_id,
            auto_find_batch_size=True,
            bf16=bool(cfg.get('bf16', False)),
            gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps', 1)),
            dataloader_num_workers=int(cfg.get('dataloader_num_workers', 8)),
            num_train_epochs=int(cfg.get('num_epochs', 1)),
            learning_rate=float(cfg.get('learning_rate', 5e-5)),
            optim=cfg.get('optimizer', 'adamw_torch_fused'),
            warmup_steps=int(cfg.get('warmup_steps', 25)),
            lr_scheduler_type=cfg.get('lr_scheduler_type', SchedulerType.COSINE_WITH_RESTARTS),
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
        per_device_train_batch_size=int(cfg.get('micro_batch_size', 4)),
        auto_find_batch_size=True,
        bf16=bool(cfg.get('bf16', False)),
        gradient_accumulation_steps=int(cfg.get('gradient_accumulation_steps', 1)),
        dataloader_num_workers=int(cfg.get('dataloader_num_workers', 8)),
        num_train_epochs=int(cfg.get('num_epochs', 1)),
        learning_rate=float(cfg.get('learning_rate', 5e-5)),
        optim=cfg.get('optimizer', 'adamw_torch_fused'),
        warmup_steps=int(cfg.get('warmup_steps', 25)),
        lr_scheduler_type=cfg.get('lr_scheduler_type', SchedulerType.COSINE_WITH_RESTARTS),
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
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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

    if cfg.get('find_lr', False):
        # run the finder, override cfg['learning_rate']
        lr = find_lr(cfg, model, train_ds, tokenizer, accelerator)
        cfg['learning_rate'] = lr

    callbacks = []
    if cfg.get('early_stopping', True):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.get('early_stopping_patience', 8))
        )

    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds, callbacks)

    logger.info("Starting training...")
    trainer = accelerator.prepare(trainer)
    trainer.train()


if __name__ == '__main__':
    main()
