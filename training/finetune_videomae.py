"""
Fine-tune VideoMAE-base on the violence dataset.

Usage (run from project root):

    python -m training.finetune_videomae

All parameters have defaults wired to config/settings.py → VideoMAEConfig,
so you can override via env vars or CLI flags without touching the source.

Training plan:
  - Replace the Kinetics-400 (400-class) head with a 2-class head.
  - Fine-tune the full model with AdamW + cosine LR schedule.
  - Use gradient checkpointing to fit in 8 GB VRAM at batch_size=2.
  - Class-weighted cross-entropy to compensate for more violent than neutral clips.
  - Save best checkpoint (by val accuracy) to models/violence_videomae/ in
    HuggingFace format — loadable directly by VideoMAESceneClassifier.

Seeds set: numpy, random, torch, PYTHONHASHSEED (requirement: reproducibility).
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running as `python -m training.finetune_videomae` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Windows: DataLoader workers > 0 requires __main__ guard (handled below).
_NUM_WORKERS = 0 if sys.platform == "win32" else 4


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Seed: %d", seed)


def build_loaders(
    dataset_root: str,
    processor,
    clip_len: int,
    batch_size: int,
):
    from training.rwf2000_dataset import ViolenceClipDataset

    train_ds = ViolenceClipDataset(
        dataset_root, split="train", clip_len=clip_len, processor=processor
    )
    val_ds = ViolenceClipDataset(
        dataset_root, split="val", clip_len=clip_len, processor=processor
    )

    # Weighted loss to compensate for class imbalance
    labels = [s[1] for s in train_ds._samples]
    n_neutral = labels.count(0)
    n_violent = labels.count(1)
    total = len(labels)
    w_neutral = total / (2 * n_neutral)
    w_violent = total / (2 * n_violent)
    class_weights = torch.tensor([w_neutral, w_violent], dtype=torch.float)
    logger.info(
        "Class weights — neutral: %.3f  violent: %.3f  (n_neutral=%d, n_violent=%d)",
        w_neutral, w_violent, n_neutral, n_violent,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_weights


def run_epoch(
    model,
    loader: DataLoader,
    criterion,
    optimizer,
    scaler,
    device: str,
    train: bool,
    grad_accum: int = 1,
) -> tuple[float, float]:
    """Run one train or eval epoch. Returns (avg_loss, accuracy)."""
    model.train(train)
    total_loss = correct = total = 0
    optimizer.zero_grad()

    for step, (pixel_values, labels) in enumerate(loader, 1):
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                logits = model(pixel_values=pixel_values).logits
                loss = criterion(logits, labels)
                if train and grad_accum > 1:
                    loss = loss / grad_accum

        if train:
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        batch_loss = loss.item() * (grad_accum if train else 1)
        total_loss += batch_loss * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if train and step % 50 == 0:
            logger.info("  step %d/%d  loss=%.4f", step, len(loader), batch_loss)

    return total_loss / total, correct / total


def main() -> None:
    # ------------------------------------------------------------------
    # CLI / defaults from settings
    # ------------------------------------------------------------------
    from config.settings import VideoMAEConfig

    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE for violence detection")
    parser.add_argument(
        "--dataset",
        default=str(VideoMAEConfig.DATASET_PATH),
        help="Dataset root (contains train/ and val/)",
    )
    parser.add_argument("--epochs",      type=int,   default=VideoMAEConfig.TRAIN_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=VideoMAEConfig.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=VideoMAEConfig.TRAIN_LR)
    parser.add_argument("--clip-len",    type=int,   default=VideoMAEConfig.CLIP_LEN)
    parser.add_argument(
        "--out",
        default=str(VideoMAEConfig.CHECKPOINT_PATH),
        help="Output checkpoint directory (HuggingFace format)",
    )
    parser.add_argument(
        "--base-model",
        default=VideoMAEConfig.BASE_MODEL,
        help="HuggingFace model ID or local path to start from",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    parser.add_argument("--seed", type=int, default=VideoMAEConfig.TRAIN_SEED)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cuda":
        logger.info("GPU: %s  VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ------------------------------------------------------------------
    # Model + processor
    # ------------------------------------------------------------------
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    logger.info("Loading processor from %s", args.base_model)
    processor = VideoMAEImageProcessor.from_pretrained(args.base_model)

    logger.info("Loading model from %s (replacing head with 2-class output) ...", args.base_model)
    model = VideoMAEForVideoClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "neutral", 1: "violent"},
        label2id={"neutral": 0, "violent": 1},
        ignore_mismatched_sizes=True,  # swaps out the Kinetics 400-class head
    )
    # Gradient checkpointing trades ~15% speed for ~40% less activation memory
    model.gradient_checkpointing_enable()
    model.to(device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, class_weights = build_loaders(
        args.dataset, processor, args.clip_len, args.batch_size
    )

    # ------------------------------------------------------------------
    # Training components
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    effective_batch = args.batch_size * args.grad_accum
    logger.info(
        "Training: epochs=%d  batch=%d  grad_accum=%d  effective_batch=%d  lr=%g",
        args.epochs, args.batch_size, args.grad_accum, effective_batch, args.lr,
    )

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        logger.info("── Epoch %d / %d ─────────────────────────────", epoch, args.epochs)

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train=True, grad_accum=args.grad_accum,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device,
            train=False,
        )
        scheduler.step()

        logger.info(
            "Epoch %d  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f",
            epoch, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(out_path))
            processor.save_pretrained(str(out_path))
            logger.info("  ✓ New best checkpoint saved → %s  (val_acc=%.3f)", out_path, best_val_acc)

    logger.info("Training complete. Best val_acc=%.3f  Checkpoint: %s", best_val_acc, out_path)


if __name__ == "__main__":
    main()
