# scripts/find_lr.py

import sys
from ruamel.yaml import YAML
from torch_lr_finder import LRFinder
from transformers import Trainer
from axolotl.train import setup_model_and_tokenizer, setup_reference_model, determine_resume_checkpoint
from axolotl.common.datasets import get_train_eval_datasets  # pseudo-API
from axolotl.utils.trainer import setup_trainer, calculate_total_num_steps

def find_best_lr(config_path, num_iter=100, end_lr=10):
    # 1) load your YAML
    yaml = YAML()
    cfg  = yaml.load(open(config_path))
    # 2) load model/tokenizer/etc
    model, tokenizer, peft_config, processor = setup_model_and_tokenizer(cfg)
    model_ref = setup_reference_model(cfg, tokenizer)
    # 3) load your datasets
    train_ds, eval_ds = get_train_eval_datasets(cfg)
    # 4) figure out total steps
    total_steps = calculate_total_num_steps(cfg, train_ds, update=False)
    # 5) build the Axolotl Trainer
    trainer = setup_trainer(
        cfg, train_ds, eval_ds,
        model, tokenizer, processor,
        total_steps, model_ref, peft_config
    )
    # 6) run the LR range test
    optimizer = trainer.optimizer
    # HF’s Trainer uses `compute_loss` internally
    criterion = trainer.compute_loss
    train_loader = trainer.get_train_dataloader()
    lr_finder = LRFinder(model, optimizer, criterion, device=trainer.args.device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)
    # 7) pick best LR
    losses = lr_finder.history["loss"]
    lrs    = lr_finder.history["lr"]
    best_lr = lrs[losses.index(min(losses))]
    print(f"Best lr ≃ {best_lr:.2e}")
    return best_lr

if __name__ == "__main__":
    cfg_path = sys.argv[1]
    find_best_lr(cfg_path)
