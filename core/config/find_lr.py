# scripts/find_lr.py
import sys
from ruamel.yaml import YAML
from torch_lr_finder import LRFinder
from axolotl.utils.models import load_tokenizer, load_model
from axolotl.utils.trainer import setup_trainer, calculate_total_num_steps
from axolotl.common.datasets import get_train_eval_datasets  # pseudocode for your data loader

def find_best_lr(config_path, num_iter=100, end_lr=10):
    cfg = YAML().load(open(config_path))

    # Load tokenizer & (optional) processor
    tokenizer = load_tokenizer(cfg)
    processor = None
    if getattr(cfg, "is_multimodal", False):
        from axolotl.utils.processor import load_processor
        processor = load_processor(cfg, tokenizer)

    # Load model + PEFT config
    model, peft_config = load_model(cfg, tokenizer, processor=processor)

    # Load your datasets
    train_ds, eval_ds = get_train_eval_datasets(cfg)

    # Compute total steps (for LR schedule, if you need it)
    total_steps = calculate_total_num_steps(cfg, train_ds, update=False)

    # Build the HuggingFace Trainer via Axolotl’s helper
    trainer = setup_trainer(
        cfg, train_ds, eval_ds,
        model, tokenizer, processor,
        total_steps, model_ref=None, peft_config=peft_config
    )

    # Run the LR range‐test
    lr_finder = LRFinder(
        trainer.model,
        trainer.optimizer,
        trainer.compute_loss,
        device=trainer.args.device
    )
    lr_finder.range_test(
        trainer.get_train_dataloader(),
        end_lr=end_lr,
        num_iter=num_iter
    )

    # Pick the LR where loss was minimal
    losses = lr_finder.history["loss"]
    lrs    = lr_finder.history["lr"]
    best_lr = lrs[losses.index(min(losses))]
    print(f"← Best LR ≃ {best_lr:.2e}")

    # Write it back into your config
    cfg["learning_rate"] = float(f"{best_lr:.2e}")
    YAML().dump(cfg, open(config_path, "w"))
    print(f"✔ Updated {config_path}")

if __name__ == "__main__":
    find_best_lr(sys.argv[1])
