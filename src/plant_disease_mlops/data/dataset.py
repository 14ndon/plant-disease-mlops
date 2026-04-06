"""PyTorch Dataset over processed ``class_*`` image folders."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from plant_disease_mlops.data.transforms import get_eval_transforms, get_train_transforms


def discover_num_classes(processed_root: os.PathLike, split: str = "train") -> int:
    """Infer class count from ``class_*`` directories (contiguous 0..N-1)."""
    root = Path(processed_root) / split
    if not root.is_dir():
        return 0
    best = -1
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("class_"):
            try:
                idx = int(p.name.split("_", 1)[1])
            except ValueError:
                continue
            best = max(best, idx)
    return best + 1 if best >= 0 else 0


class PlantDiseaseDataset(Dataset):
    """Loads ``data/processed/{split}/class_{i}/*.jpg`` written by ``preprocess_data``."""

    def __init__(
        self,
        data_dir: os.PathLike,
        split: str = "train",
        transform=None,
        metadata_df=None,
        use_metadata: bool = False,
    ):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.use_metadata = use_metadata
        self.metadata_df = metadata_df

        self.samples: list[tuple[str, int]] = []
        if not os.path.isdir(self.data_dir):
            return

        class_dirs = sorted(
            d for d in os.listdir(self.data_dir) if d.startswith("class_")
        )
        for class_dir_name in class_dirs:
            class_dir = os.path.join(self.data_dir, class_dir_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = int(class_dir_name.split("_", 1)[1])
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(class_dir, img_name), class_idx)
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.use_metadata and self.metadata_df is not None:
            meta = self._get_metadata(img_path)
            return image, meta, label
        return image, label

    def _get_metadata(self, img_path: str) -> torch.Tensor:
        del img_path
        return torch.tensor([20.0, 60.0, 6.5], dtype=torch.float32)


def get_dataloaders(
    data_dir: os.PathLike,
    batch_size: int = 32,
    num_workers: int = 4,
):
    from torch.utils.data import DataLoader

    train_dataset = PlantDiseaseDataset(
        data_dir=data_dir,
        split="train",
        transform=get_train_transforms(),
    )
    test_dataset = PlantDiseaseDataset(
        data_dir=data_dir,
        split="test",
        transform=get_eval_transforms(),
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )
