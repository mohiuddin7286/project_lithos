from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image


print("--- YOLOV8 DATA PREPARATION SEQUENCE ---")

ROOT_DIR = Path(__file__).resolve().parent
RAW_IMAGES_DIRS = [ROOT_DIR / "data" / "images" / "images", ROOT_DIR / "data" / "images"]
RAW_JSON_DIR = ROOT_DIR / "data" / "images" / "annotations"
OUTPUT_ROOT = ROOT_DIR / "datasets" / "lithos_vision"
OUTPUT_IMAGES_DIR = OUTPUT_ROOT / "images"
OUTPUT_LABELS_DIR = OUTPUT_ROOT / "labels"
TRAIN_IMAGES_DIR = OUTPUT_IMAGES_DIR / "train"
VAL_IMAGES_DIR = OUTPUT_IMAGES_DIR / "val"
TRAIN_LABELS_DIR = OUTPUT_LABELS_DIR / "train"
VAL_LABELS_DIR = OUTPUT_LABELS_DIR / "val"
VAL_RATIO = 0.2
RANDOM_SEED = 42
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def find_image_path(base_name: str) -> Path | None:
    for raw_dir in RAW_IMAGES_DIRS:
        for extension in SUPPORTED_IMAGE_EXTENSIONS:
            candidate = raw_dir / f"{base_name}{extension}"
            if candidate.exists():
                return candidate
    return None


def discover_class_names() -> list[str]:
    labels: set[str] = set()

    for json_path in sorted(RAW_JSON_DIR.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for shape in data.get("shapes", []):
            label_name = str(shape.get("label", "")).strip()
            if label_name:
                labels.add(label_name)

    return sorted(labels)


def convert_bbox_to_yolo(image_size: tuple[int, int], box: list[float]) -> tuple[float, float, float, float]:
    width, height = image_size
    x_min, y_min, x_max, y_max = box

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    box_width = x_max - x_min
    box_height = y_max - y_min

    return (
        x_center / width,
        y_center / height,
        box_width / width,
        box_height / height,
    )


def extract_bbox(shape: dict) -> list[float] | None:
    points = shape.get("points", [])
    if len(points) < 2:
        return None

    xs = [float(point[0]) for point in points if len(point) >= 2]
    ys = [float(point[1]) for point in points if len(point) >= 2]
    if not xs or not ys:
        return None

    return [min(xs), min(ys), max(xs), max(ys)]


class_names = discover_class_names()
class_map = {name: index for index, name in enumerate(class_names)}

print(f"Discovered {len(class_names)} classes: {class_map}")

if not RAW_JSON_DIR.exists():
    raise FileNotFoundError(f"Missing annotation directory: {RAW_JSON_DIR}")

OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
VAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)

json_files = sorted(RAW_JSON_DIR.glob("*.json"))
random.Random(RANDOM_SEED).shuffle(json_files)

if not json_files:
    print("No JSON annotations were found. Nothing to convert.")
    raise SystemExit(0)

val_count = max(1, round(len(json_files) * VAL_RATIO)) if len(json_files) > 1 else 0
val_files = set(json_files[:val_count])
processed_count = 0
skipped_missing_images = 0

for json_path in json_files:
    base_name = json_path.stem
    image_path = find_image_path(base_name)
    if image_path is None:
        skipped_missing_images += 1
        continue

    split_images_dir = VAL_IMAGES_DIR if json_path in val_files else TRAIN_IMAGES_DIR
    split_labels_dir = VAL_LABELS_DIR if json_path in val_files else TRAIN_LABELS_DIR
    output_image_path = split_images_dir / image_path.name
    output_label_path = split_labels_dir / f"{base_name}.txt"

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    shutil.copy2(image_path, output_image_path)

    with output_label_path.open("w", encoding="utf-8") as output_handle:
        for shape in data.get("shapes", []):
            label_name = str(shape.get("label", "")).strip()
            if label_name not in class_map:
                continue

            bbox = extract_bbox(shape)
            if bbox is None:
                continue

            class_id = class_map[label_name]
            x_center, y_center, box_width, box_height = convert_bbox_to_yolo((image_width, image_height), bbox)
            output_handle.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
            )

    processed_count += 1
    if processed_count % 500 == 0:
        print(f"Processed {processed_count} files...")

print(f"✅ Conversion complete. Generated {processed_count} YOLO label files.")
if skipped_missing_images:
    print(f"Skipped {skipped_missing_images} annotations because the matching image was missing.")
