# scripts/find_lr.py
import sys, os
from ruamel.yaml import YAML
from transformers import Trainer
from torch_lr_finder import LRFinder

def find_best_lr(config_path, output_path=None, num_iter=100, end_lr=10):
    yaml = YAML()
    cfg = yaml.load(open(config_path))

    # instantiate Trainer but only for a handful of steps
    trainer = Trainer(cfg, dry_run=True)  
    model, optimizer, criterion, train_loader = (
        trainer.model, trainer.optimizer, trainer.criterion, trainer.train_dataloader()
    )

    lr_finder = LRFinder(model, optimizer, criterion, device=trainer.device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)
    best_lr = lr_finder.history["lr"][lr_finder.history["loss"].index(min(lr_finder.history["loss"]))]
    lr_finder.reset()

    print(f"➡️  Best lr ≃ {best_lr:.2e}")
    # write it back
    cfg["learning_rate"] = float(f"{best_lr:.2e}")
    out = output_path or config_path
    yaml.dump(cfg, open(out, "w"))
    print(f"Updated config at {out}")

if __name__ == "__main__":
    _, cfg_path = sys.argv
    find_best_lr(cfg_path)
