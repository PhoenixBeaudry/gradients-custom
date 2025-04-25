#!/usr/bin/env python3
import sys, torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from torch_lr_finder import LRFinder

def find_and_patch_lr(cfg_path: str):
    yaml = YAML()
    cfg = yaml.load(open(cfg_path))

    # 1️⃣ Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"]).to("cuda")

    # 2️⃣ Load & tokenize your dataset
    ds = load_dataset(
        cfg["dataset_format"],
        data_files={"train": cfg["train_file"], **({"validation": cfg["valid_file"]} if cfg.get("valid_file") else {})},
    )
    def tok(ex): return tokenizer(ex[cfg["text_column"]],
                                    truncation=True,
                                    max_length=cfg["max_length"])
    tokenized = ds.map(tok, batched=True, remove_columns=ds["train"].column_names)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(
        tokenized["train"],
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collator
    )

    # 3️⃣ Set up optimizer & LR-finder
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    criterion = lambda m, batch: m(**batch).loss

    lr_finder = LRFinder(
        model, optimizer, criterion,
        device="cuda", 
        memory_cache=False
    )
    lr_finder.range_test(
        loader,
        end_lr=cfg["lr_find_end_lr"],
        num_iter=cfg["lr_find_iterations"]
    )

    # 4️⃣ Pick best LR & reset
    losses = lr_finder.history["loss"]
    lrs    = lr_finder.history["lr"]
    best_lr = lrs[losses.index(min(losses))]
    lr_finder.reset()

    print(f"\n✅ Best LR ≃ {best_lr:.2e}")

    # 5️⃣ Patch your YAML
    cfg["learning_rate"] = float(f"{best_lr:.2e}")
    yaml.dump(cfg, open(cfg_path, "w"))
    print(f"✏️  Wrote learning_rate: {best_lr:.2e} into {cfg_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: find_lr_generic.py <config.yml>")
        sys.exit(1)
    find_and_patch_lr(sys.argv[1])
