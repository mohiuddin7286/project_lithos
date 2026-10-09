from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".h5", ".hdf5"}


def _first_dataset(group):
    for value in group.values():
        if isinstance(value, h5py.Dataset):
            return value[()]
        if isinstance(value, h5py.Group):
            nested = _first_dataset(value)
            if nested is not None:
                return nested
    return None


def _read_h5(path, preferred_keys):
    with h5py.File(path, "r") as file:
        for key in preferred_keys:
            if key in file:
                return file[key][()]
        data = _first_dataset(file)
        if data is None:
            raise ValueError(f"No dataset found in HDF5 file: {path}")
        return data


def _normalize_image(image):
    image = image.astype(np.float32)
    normalized = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[-1]):
        values = image[:, :, channel]
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        if max_value > min_value:
            normalized[:, :, channel] = (values - min_value) / (max_value - min_value)

    return np.nan_to_num(normalized)


class LandslideDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256, channels=3):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.channels = channels
        self.image_paths = sorted(
            [
                path
                for path in self.image_dir.iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def _read_image(self, image_path):
        if image_path.suffix.lower() in {".h5", ".hdf5"}:
            image = _read_h5(image_path, preferred_keys=("img", "image", "images", "data"))
            image = np.asarray(image)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            if image.ndim != 3:
                raise ValueError(f"Expected 2D or 3D image data in: {image_path}")
            if image.shape[0] <= 20 and image.shape[-1] > 20:
                image = np.moveaxis(image, 0, -1)
            image = image[:, :, : self.channels]
            if image.shape[-1] < self.channels:
                repeat_count = self.channels - image.shape[-1]
                image = np.concatenate([image, np.repeat(image[:, :, -1:], repeat_count, axis=-1)], axis=-1)
            image = cv2.resize(image, (self.image_size, self.image_size))
            return _normalize_image(image)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        return image.astype(np.float32) / 255.0

    def _read_mask(self, mask_path):
        if mask_path.suffix.lower() in {".h5", ".hdf5"}:
            mask = _read_h5(mask_path, preferred_keys=("mask", "label", "labels", "y", "data"))
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = np.squeeze(mask)
            mask = cv2.resize(mask.astype(np.float32), (self.image_size, self.image_size))
            return (mask > 0).astype(np.float32)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        return (mask > 127).astype(np.float32)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name

        if not mask_path.exists():
            alt_name = image_path.name.replace("image_", "mask_").replace("image", "mask").replace("img_", "mask_")
            mask_path = self.mask_dir / alt_name

        if not mask_path.exists():
            stem_number = "".join(filter(str.isdigit, image_path.stem))
            if stem_number:
                matches = list(self.mask_dir.glob(f"*{stem_number}*"))
                if matches:
                    mask_path = matches[0]

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image {image_path.name}")

        image = self._read_image(image_path)
        mask = self._read_mask(mask_path)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask


if __name__ == "__main__":
    import sys

    root_dir = Path(__file__).resolve().parent.parent
    img_dir = root_dir / "data" / "images"
    msk_dir = root_dir / "data" / "masks"

    print(f"Checking dataset at:\n  Images: {img_dir}\n  Masks:  {msk_dir}")
    if not img_dir.exists() or not msk_dir.exists():
        print("Dataset directories do not exist.")
        sys.exit(1)

    dataset = LandslideDataset(img_dir, msk_dir)
    print(f"Found {len(dataset)} image/mask pairs.")
    if len(dataset) > 0:
        img, msk = dataset[0]
        print(f"Sample 0 - Image shape: {img.shape}, Mask shape: {msk.shape}")

