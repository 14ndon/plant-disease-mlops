"""Torchvision transforms for training/eval; PIL helpers for ingestion scripts."""

from __future__ import annotations

from typing import Literal

from PIL import Image
from torchvision import transforms

DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def inference_image_size() -> int:
    return DEFAULT_IMAGE_SIZE


def preprocess_pil_image(img: Image.Image) -> Image.Image:
    """RGB, bicubic resize for on-disk processed JPEGs (ingestion)."""
    return img.convert("RGB").resize(
        (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        Image.Resampling.BICUBIC,
    )


def get_train_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_transforms(split: Literal["train", "test"] = "train", image_size: int = DEFAULT_IMAGE_SIZE):
    """Alias matching the proposal naming (train vs test/eval)."""
    if split == "train":
        return get_train_transforms(image_size)
    return get_eval_transforms(image_size)
