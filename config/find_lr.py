#!/usr/bin/env python
import sys
from pathlib import Path

from ruamel.yaml import YAML
from torch_lr_finder import LRFinder

from axolotl.train import setup_model_and_trainer 
from axolotl.common.datasets import load_datasets 
from axolotl.cli.args import TrainerCliArgs 


def find_and_patch_lr(config_path: str, num_iter: int = 100, end_lr: float = 10.0):
    yaml = YAML()
    cfg = yaml.load(open(config_path))

    # 1) Load datasets metadata for Trainer
    cli_args = TrainerCliArgs()
    dataset_meta = load_datasets(cfg, cli_args)

    # 2) Build the Axolotl trainer + model
    trainer_builder, model, tokenizer, peft_config, processor = setup_model_and_trainer(
        cfg, dataset_meta
    )

    # 3) Extract the underlying HF Trainer
    #    (HFCausalTrainerBuilder / HFRLTrainerBuilder both expose `.trainer`)
    hf_trainer = trainer_builder.trainer

    # 4) Run LR range test on a handful of batches
    optimizer = hf_trainer.optimizer
    # HuggingFace Trainer uses `compute_loss` under the hood
    criterion = lambda m, batch: hf_trainer.compute_loss(m, batch)[0]
    train_loader = hf_trainer.get_train_dataloader()

    lr_finder = LRFinder(model, optimizer, criterion, device=hf_trainer.args.device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)
    best_lr = lr_finder.history["lr"][lr_finder.history["loss"].index(min(lr_finder.history["loss"]))]
    lr_finder.reset()

    print(f"🔍 Best learning rate ≃ {best_lr:.2e}")

    # 5) Write it back into the YAML
    cfg["learning_rate"] = float(f"{best_lr:.2e}")
    yaml.dump(cfg, open(config_path, "w"))
    print(f"✏️  Patched `{config_path}` with learning_rate: {best_lr:.2e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: find_lr.py <path/to/config.yml>")
        sys.exit(1)
    find_and_patch_lr(sys.argv[1])
