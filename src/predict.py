import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from src.config import IMAGE_SIZE, OUTPUT_DIR, THRESHOLD
    from src.model import UNet
except ImportError:
    from config import IMAGE_SIZE, OUTPUT_DIR, THRESHOLD
    from model import UNet


def load_model(model_path, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path, image_size=IMAGE_SIZE):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        pil_img = Image.open(path).convert("RGB")
        image = np.array(pil_img)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_shape = image.shape[:2]
    resized = cv2.resize(image, (image_size, image_size))
    tensor = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)
    return image, tensor, original_shape


@torch.no_grad()
def predict(model, device, image_tensor, original_shape, threshold=THRESHOLD):
    logits = model(image_tensor.to(device))
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    probs_resized = cv2.resize(probs, (original_shape[1], original_shape[0]))
    mask = (probs_resized > threshold).astype(np.uint8) * 255
    return probs_resized, mask


def main():
    default_model = OUTPUT_DIR / "best_model.pt"

    parser = argparse.ArgumentParser(description="Run landslide prediction on a satellite image.")
    parser.add_argument("--image", required=True, help="Path to input image file.")
    parser.add_argument("--model", default=str(default_model), help="Path to trained model weights.")
    parser.add_argument("--output", default="prediction_mask.png", help="Path to save output prediction mask.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Binary threshold.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.model}")
    model = load_model(args.model, device)

    print(f"Processing image: {args.image}")
    orig_img, img_tensor, orig_shape = preprocess_image(args.image)

    probs, mask = predict(model, device, img_tensor, orig_shape, threshold=args.threshold)

    output_path = Path(args.output)
    cv2.imwrite(str(output_path), mask)
    print(f"Saved prediction mask to: {output_path.resolve()}")

    landslide_coverage = (mask > 0).mean() * 100
    print(f"Landslide risk score (avg prob): {probs.mean():.4f}")
    print(f"Predicted landslide coverage area: {landslide_coverage:.2f}%")


if __name__ == "__main__":
    main()
