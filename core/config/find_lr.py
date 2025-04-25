#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM

def find_and_patch_lr(cfg_path: str):
    # --- load config ---
    yaml = YAML()
    cfg = yaml.load(open(cfg_path))

    # --- pull dataset path out of config["datasets"][0]["path"] ---
    datasets = cfg.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("No entries under 'datasets' in config")
    ds_entry = datasets[0]
    ds_path = ds_entry.get("path")
    if ds_path is None:
        raise KeyError("Could not find 'path' in your dataset entry")

    # --- validation fraction & micro batch size from top level ---
    val_frac = float(cfg.get("val_set_size", 0) or 0)
    batch_size = int(cfg.get("micro_batch_size", 1))

    # --- load & split dataset ---
    # infer format from extension
    fmt = Path(ds_path).suffix.lstrip(".").lower()
    ds = load_dataset(fmt, data_files={"train": ds_path})
    train_ds = (
        ds["train"].train_test_split(test_size=val_frac, seed=42)["train"]
        if val_frac > 0
        else ds["train"]
    )

    # --- tokenizer, model, collator, loader ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(cfg["base_model"]).to("cuda")
    collator  = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader    = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)

    # --- set up optimizer & LR‐finder ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    criterion = lambda m, b: m(**b).loss

    finder = LRFinder(model, optimizer, criterion, device="cuda", memory_cache=False)
    finder.range_test(
        loader,
        end_lr=float(cfg.get("lr_find_end_lr", 10.0)),
        num_iter=int(cfg.get("lr_find_iterations", 100)),
    )

    # --- pick best & patch config ---
    losses = finder.history["loss"]
    lrs    = finder.history["lr"]
    best_lr = lrs[losses.index(min(losses))]
    finder.reset()
    print(f"\n✅ Best LR ≃ {best_lr:.2e}")

    cfg["learning_rate"] = float(f"{best_lr:.2e}")
    yaml.dump(cfg, open(cfg_path, "w"))
    print(f"✏️  Wrote learning_rate: {best_lr:.2e} into {cfg_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: find_lr_generic.py <config.yml>")
        sys.exit(1)
    find_and_patch_lr(sys.argv[1])
