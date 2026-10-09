import argparse
from pathlib import Path
import sys

# Ensure src directory is in Python path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

try:
    from src.config import BATCH_SIZE, EPOCHS, IMAGE_SIZE, LEARNING_RATE, OUTPUT_DIR
    from src.dataset import LandslideDataset
    from src.model import UNet
except ImportError:
    from config import BATCH_SIZE, EPOCHS, IMAGE_SIZE, LEARNING_RATE, OUTPUT_DIR
    from dataset import LandslideDataset
    from model import UNet


def dice_score(logits, masks, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(logits.detach(), masks).item()

    return total_loss / len(loader), total_dice / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score(logits, masks).item()

    return total_loss / len(loader), total_dice / len(loader)


def main():
    default_img_dir = Path(__file__).resolve().parent.parent / "data" / "images"
    default_msk_dir = Path(__file__).resolve().parent.parent / "data" / "masks"

    parser = argparse.ArgumentParser(description="Train project_lithos U-Net model.")
    parser.add_argument("--images", default=str(default_img_dir), help="Path to training image folder.")
    parser.add_argument("--masks", default=str(default_msk_dir), help="Path to training mask folder.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--channels", type=int, default=3)
    args = parser.parse_args()

    dataset = LandslideDataset(args.images, args.masks, image_size=args.image_size, channels=args.channels)
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two image/mask pairs.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=args.channels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
            f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"Saved best model with validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
