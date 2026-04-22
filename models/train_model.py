"""
EfficientNet-B4 Training Pipeline for Indian Crop Disease Detection
Optimized for NVIDIA RTX 4060 GPU (8GB VRAM)

Crops supported: Tomato, Potato, Maize, Rice, Wheat, Ragi, Sugarcane
Total: ~54+ disease classes from Indian agricultural datasets

Usage:
    python models/train_model.py --data_dir data/raw --epochs 50 --batch_size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "training.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_model")


# ── Training Configuration ───────────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32       # 32 for RTX 4060 8GB VRAM (reduce to 16 if OOM)
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 4
MIXED_PRECISION = True


# ── Data Augmentation ────────────────────────────────────────────────────────
def get_transforms(mode: str = "train"):
    """Get data transforms for training/validation."""
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ── Model Builder ────────────────────────────────────────────────────────────
def build_efficientnet_b4(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B4 with custom classification head."""
    try:
        from efficientnet_pytorch import EfficientNet

        if pretrained:
            model = EfficientNet.from_pretrained("efficientnet-b4")
            logger.info("Loaded pretrained EfficientNet-B4 weights")
        else:
            model = EfficientNet.from_name("efficientnet-b4")
            logger.info("Initialized EfficientNet-B4 from scratch")

        # Replace classifier head
        in_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        logger.info(f"EfficientNet-B4 configured for {num_classes} classes")
        return model

    except ImportError:
        logger.error("efficientnet-pytorch not installed! Install with: pip install efficientnet-pytorch")
        raise


# ── Dataset Preparation ──────────────────────────────────────────────────────
def prepare_datasets(data_dir: str, val_split: float = 0.15):
    """
    Prepare train/val datasets from directory structure.
    Expected structure:
        data_dir/
            crop_disease_class1/
                image1.jpg
                image2.jpg
            crop_disease_class2/
                ...
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Get all subdirectories as classes
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories found in {data_dir}")

    logger.info(f"Found {len(class_dirs)} classes in {data_dir}")
    for d in class_dirs[:10]:
        count = len(list(d.glob("*")))
        logger.info(f"  {d.name}: {count} images")
    if len(class_dirs) > 10:
        logger.info(f"  ... and {len(class_dirs)-10} more classes")

    # Create train/val datasets
    full_dataset = datasets.ImageFolder(str(data_path), transform=get_transforms("train"))
    num_classes = len(full_dataset.classes)

    logger.info(f"Total images: {len(full_dataset)}, Classes: {num_classes}")

    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply val transforms to validation set
    val_dataset.dataset = datasets.ImageFolder(str(data_path), transform=get_transforms("val"))

    # Weighted sampler for class imbalance
    targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = np.bincount(targets, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)  # +1 to avoid div by zero
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    logger.info(f"Train: {train_size}, Val: {val_size}")
    logger.info(f"Class distribution: min={class_counts.min()}, max={class_counts.max()}, "
                f"mean={class_counts.mean():.0f}")

    return train_dataset, val_dataset, full_dataset.classes, sampler


# ── Training Loop ────────────────────────────────────────────────────────────
def train_model(
    data_dir: str,
    output_dir: str = "models/checkpoints/efficientnet_b4_indian",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    resume_from: str = "",
):
    """
    Train EfficientNet-B4 on Indian crop disease dataset.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Prepare data
    train_dataset, val_dataset, class_names, sampler = prepare_datasets(data_dir)
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Build model
    model = build_efficientnet_b4(num_classes, pretrained=True)
    model = model.to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=MIXED_PRECISION)

    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accs = []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    logger.info(f"Classes: {num_classes}, Model: EfficientNet-B4")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                batch_acc = 100.0 * correct / total
                logger.info(
                    f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | Acc: {batch_acc:.1f}%"
                )

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(enabled=MIXED_PRECISION):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.0f}s"
        )

        # ── Save best model ──────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save PyTorch checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "num_classes": num_classes,
            }, output_path / "best_model.pth")

            logger.info(f"  ✅ New best model saved! Val Acc: {val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info(f"  ⏹ Early stopping after {epoch+1} epochs (patience={EARLY_STOPPING_PATIENCE})")
                break

    # ── Save final artifacts ─────────────────────────────────────────────────

    # Save class labels mapping
    class_labels = {}
    for idx, name in enumerate(class_names):
        parts = name.lower().replace(" ", "_").split("___")
        if len(parts) >= 2:
            crop = parts[0]
            disease = parts[1]
        else:
            crop = name.split("_")[0] if "_" in name else "unknown"
            disease = name

        class_labels[str(idx)] = {
            "class_name": name,
            "crop": crop,
            "disease": disease.replace("_", " ").title(),
        }

    labels_path = output_path / "class_labels.json"
    labels_path.write_text(json.dumps(class_labels, indent=2), encoding="utf-8")
    logger.info(f"Saved class labels to {labels_path}")

    # Save training curves
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(train_losses, label="Train Loss")
        ax1.plot(val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(val_accs, label="Val Accuracy", color="green")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"Validation Accuracy (Best: {best_val_acc:.1f}%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        fig.savefig(output_path / "training_curves.png", dpi=150)
        plt.close()
        logger.info("Training curves saved")
    except Exception as e:
        logger.warning(f"Could not save training curves: {e}")

    logger.info(f"Training complete! Best Val Acc: {best_val_acc:.1f}%")
    logger.info(f"Model saved to: {output_path}")

    return model, class_names


# ── ONNX Export ──────────────────────────────────────────────────────────────
def export_to_onnx(
    checkpoint_path: str,
    output_path: str = "models/checkpoints/efficientnet_b4_indian/model.onnx",
    num_classes: int = None,
):
    """Export trained model to ONNX for offline inference."""
    logger.info(f"Exporting model to ONNX: {output_path}")

    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if num_classes is None:
        num_classes = ckpt.get("num_classes", len(ckpt.get("class_names", [])))

    model = build_efficientnet_b4(num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={"image": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=14,
    )

    logger.info(f"ONNX model exported to {output_path}")

    # Save class labels alongside ONNX
    class_names = ckpt.get("class_names", [])
    if class_names:
        labels_path = Path(output_path).parent / "class_labels.json"
        if not labels_path.exists():
            labels = {str(i): name for i, name in enumerate(class_names)}
            labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 for Indian Crop Disease Detection")
    parser.add_argument("--data_dir", type=str, default="data/unified",
                        help="Path to unified dataset directory")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/efficientnet_b4_indian",
                        help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint path")
    parser.add_argument("--export_onnx", action="store_true",
                        help="Export best model to ONNX after training")

    args = parser.parse_args()

    model, class_names = train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_from=args.resume,
    )

    if args.export_onnx:
        ckpt_path = Path(args.output_dir) / "best_model.pth"
        onnx_path = Path(args.output_dir) / "model.onnx"
        export_to_onnx(str(ckpt_path), str(onnx_path))
