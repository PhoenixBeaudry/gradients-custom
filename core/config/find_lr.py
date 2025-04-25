#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

def infer_format(path: str):
    ext = Path(path).suffix.lower()
    if ext in {".json", ".jsonl"}:
        return "json"
    if ext == ".csv":
        return "csv"
    if ext in {".txt"}:
        return "text"
    raise ValueError(f"Cannot infer dataset format from {ext}")

def find_and_patch_lr(cfg_path: str):
    yaml = YAML()
    cfg = yaml.load(open(cfg_path))

    ds_path = cfg["datasets"]["dataset_prepared_path"]
    val_frac = float(cfg["datasets"].get("val_set_size", 0) or 0)

    fmt = infer_format(ds_path)
    data_files = {"train": ds_path}

    ds = load_dataset(fmt, data_files=data_files)
    if val_frac > 0:
        split = ds["train"].train_test_split(test_size=val_frac, seed=42)
        train_ds = split["train"]
    else:
        train_ds = ds["train"]

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    model = torch.load  # placeholder; actual model loading is only for LR lookup

    # data collator
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(
        train_ds, batch_size=cfg["micro_batch_size"],
        shuffle=True, collate_fn=collator
    )

    # build a minimal model+optimizer for range test:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"]).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    criterion = lambda m, b: m(**b).loss

    # run LR finder
    finder = LRFinder(
        model, optimizer, criterion,
        device="cuda", memory_cache=False
    )
    finder.range_test(
        loader,
        end_lr=cfg.get("lr_find_end_lr", 10.0),
        num_iter=int(cfg.get("lr_find_iterations", 100))
    )

    # pick best
    losses = finder.history["loss"]
    lrs    = finder.history["lr"]
    best_lr = lrs[losses.index(min(losses))]
    finder.reset()

    print(f"\n✅ Best LR ≃ {best_lr:.2e}")

    # patch YAML
    cfg["learning_rate"] = float(f"{best_lr:.2e}")
    yaml.dump(cfg, open(cfg_path, "w"))
    print(f"✏️  Updated learning_rate in {cfg_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: find_lr_generic.py <config.yml>")
        sys.exit(1)
    find_and_patch_lr(sys.argv[1])
