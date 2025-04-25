#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, TensorDataset
from torch_lr_finder import LRFinder
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
)

def find_and_patch_lr(cfg_path: str):
    yaml = YAML()
    cfg  = yaml.load(open(cfg_path))

    # --- extract dataset path + val split ---
    ds_entry = cfg["datasets"][0]
    ds_path  = ds_entry["path"]
    val_frac = float(cfg.get("val_set_size", 0) or 0)
    fmt      = Path(ds_path).suffix.lstrip(".").lower()
    raw_ds   = load_dataset(fmt, data_files={"train": ds_path})

    # --- split off validation if any (we only need train for LR finder) ---
    train_ds = (
        raw_ds["train"].train_test_split(test_size=val_frac, seed=42)["train"]
        if val_frac > 0
        else raw_ds["train"]
    )

    # --- tokenizer + model stub for LR finder ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(cfg["base_model"]).to("cuda")

    # --- build a DataLoader over *chosen* sequences ---
    batch_size = int(cfg.get("micro_batch_size", 1))
    seq_len    = int(cfg.get("sequence_len", 512))

    # DPO-style: question + chosen
    if "chosen" in train_ds.column_names and "question" in train_ds.column_names:
        texts = [
            q + c
            for q, c in zip(train_ds["question"], train_ds["chosen"])
        ]
    # Instruct-style: prompt + completion
    elif "prompt" in train_ds.column_names and "completion" in train_ds.column_names:
        texts = [
            p + comp
            for p, comp in zip(train_ds["prompt"], train_ds["completion"])
        ]
    # fallback: assume single text column
    else:
        col = train_ds.column_names[0]
        texts = train_ds[col]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len,
        padding="longest",
        return_tensors="pt",
    )
    ds_for_lr = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    loader    = DataLoader(ds_for_lr, batch_size=batch_size, shuffle=True)

    # --- LR finder setup ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    def criterion(m, batch):
        ids, mask = batch[0].to("cuda"), batch[1].to("cuda")
        return m(input_ids=ids, attention_mask=mask).loss

    finder = LRFinder(model, optimizer, criterion, device="cuda", memory_cache=False)
    finder.range_test(
        loader,
        end_lr=float(cfg.get("lr_find_end_lr", 10.0)),
        num_iter=int(cfg.get("lr_find_iterations", 100)),
    )

    # --- pick best LR & patch config ---
    losses = finder.history["loss"]
    lrs    = finder.history["lr"]
    best_lr = lrs[losses.index(min(losses))]
    finder.reset()

    print(f"\n✅ Best LR ≃ {best_lr:.2e}")
    cfg["learning_rate"] = float(f"{best_lr:.2e}")
    yaml.dump(cfg, open(cfg_path, "w"))
    print(f"✏️  Updated `learning_rate` in {cfg_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: find_lr_generic.py <config.yml>")
        sys.exit(1)
    find_and_patch_lr(sys.argv[1])
