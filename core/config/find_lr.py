#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset
from torch_lr_finder import LRFinder
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_and_patch_lr(cfg_path: str):
    # Load YAML config
    yaml = YAML()
    cfg = yaml.load(open(cfg_path, "r"))

    # --- Extract dataset path and val split ---
    datasets_cfg = cfg.get("datasets", [])
    if not isinstance(datasets_cfg, list) or not datasets_cfg:
        raise ValueError("No entries under 'datasets' in config")
    ds_entry = datasets_cfg[0]
    ds_path = ds_entry.get("path")
    if ds_path is None:
        raise KeyError("Could not find 'path' in your dataset entry")
    val_frac = float(cfg.get("val_set_size", 0) or 0)

    # --- Load and split dataset ---
    fmt = Path(ds_path).suffix.lstrip(".").lower()
    raw_ds = load_dataset(fmt, data_files={"train": ds_path})
    train_ds = (
        raw_ds["train"].train_test_split(test_size=val_frac, seed=42)["train"]
        if val_frac > 0
        else raw_ds["train"]
    )

    # --- Build list of texts for LR finder ---
    if "chosen" in train_ds.column_names and "question" in train_ds.column_names:
        # DPO-style: question + chosen
        texts = [q + c for q, c in zip(train_ds["question"], train_ds["chosen"])]
    elif "prompt" in train_ds.column_names and "completion" in train_ds.column_names:
        # Instruct-style: prompt + completion
        texts = [p + comp for p, comp in zip(train_ds["prompt"], train_ds["completion"])]
    else:
        # Fallback: single text column
        col = train_ds.column_names[0]
        texts = train_ds[col]

    # --- Tokenizer & Model ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"]).to("cuda")

    batch_size = int(cfg.get("micro_batch_size", 1))
    seq_len = int(cfg.get("sequence_len", 512))

    # --- Tokenize for LR finder ---
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len,
        padding="longest",
        return_tensors="pt",
    )

    # --- Custom Dataset for LR finder ---
    class LRRangeDataset(Dataset):
        def __init__(self, encodings):
            self.enc = encodings

        def __len__(self):
            return self.enc["input_ids"].size(0)

        def __getitem__(self, idx):
            item = {
                "input_ids": self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels": self.enc["input_ids"][idx],  # use input_ids as labels
            }
            # torch-lr-finder expects (inputs, labels)
            return item, self.enc["input_ids"][idx]

    loader = DataLoader(
        LRRangeDataset(enc),
        batch_size=batch_size,
        shuffle=True,
    )

    # --- Set up optimizer & LR Finder ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

    def criterion(outputs, labels):
        # we passed labels into the model, so outputs.loss is computed
        return outputs.loss

    finder = LRFinder(model, optimizer, criterion, device="cuda", memory_cache=False)
    finder.range_test(
        loader,
        end_lr=float(cfg.get("lr_find_end_lr", 10.0)),
        num_iter=int(cfg.get("lr_find_iterations", 100)),
    )

    # --- Pick best LR and patch config ---
    losses = finder.history["loss"]
    lrs = finder.history["lr"]
    best_lr = lrs[losses.index(min(losses))]
    finder.reset()

    print(f"\n✅ Best LR ≃ {best_lr:.2e}")
    cfg["learning_rate"] = float(f"{best_lr:.2e}")
    yaml.dump(cfg, open(cfg_path, "w"))
    print(f"✏️  Updated learning_rate in {cfg_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: find_lr.py <config.yml>")
        sys.exit(1)
    find_and_patch_lr(sys.argv[1])
