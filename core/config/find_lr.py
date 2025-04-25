
import yaml
from axolotl.common.datasets import load_datasets
from axolotl.cli.args import TrainerCliArgs
from axolotl.train import setup_model_and_tokenizer
import sys

def main():
    cfg_path = sys.argv[1]
    # 1. Load the YAML config into a Python dict
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    # 2. Load the data using Axolotl utilities (TrainerCliArgs can be default)
    cli_args = TrainerCliArgs()  # debug=False, etc.
    dataset_meta = load_datasets(cfg, cli_args)  # returns TrainDatasetMeta with train_dataset, eval_dataset

    # 3. Load the model and tokenizer (includes LoRA/QLoRA, CPU offloading, etc., per config)
    model, tokenizer, peft_config, processor = setup_model_and_tokenizer(cfg)

    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Prepare a DataLoader (using the config’s batch size)
    batch_size = cfg.get("micro_batch_size", 1)
    train_loader = DataLoader(
        dataset_meta.train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=default_data_collator
    )

    # LR sweep parameters
    init_lr = 1e-7
    final_lr = 10
    num_steps = 100

    optimizer = AdamW(model.parameters(), lr=init_lr)
    # Compute the exponential multiplier to reach final_lr in num_steps
    mult = (final_lr / init_lr) ** (1 / (num_steps - 1))

    # Run the LR range test
    smoothed_loss = 0.0
    best_loss = float('inf')
    losses = []
    lrs = []
    for step, batch in enumerate(train_loader):
        if step >= num_steps: 
            break
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        loss_item = loss.item()
        # Smooth the loss (fastai style)
        smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss_item
        corrected_loss = smoothed_loss / (1 - 0.98**(step+1))
        # Record LR and (smoothed) loss
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        losses.append(corrected_loss)
        # Track best loss and break if diverging
        if corrected_loss < best_loss:
            best_loss = corrected_loss
        if corrected_loss > 4 * best_loss:
            break
        # Backprop and update
        loss.backward()
        optimizer.step()
        # Increase LR
        optimizer.param_groups[0]['lr'] *= mult

    # Pick the LR corresponding to the minimum loss (or e.g. 10x smaller for safety)
    min_loss_idx = losses.index(min(losses))
    found_lr = lrs[min_loss_idx] * 0.1  # one-tenth of the LR at minimum loss

    # Update the config dict
    cfg["learning_rate"] = found_lr

    # Write it back out (overwriting the file)
    with open("config.yml", "w") as f:
        yaml.safe_dump(cfg, f)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_lr.py <config.yml>")
        sys.exit(1)
    main()
