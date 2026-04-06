"""Tests for PIL checks and Great Expectations manifest validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image
from PIL import Image as PILImage

from plant_disease_mlops.data.dataset import PlantDiseaseDataset, discover_num_classes
from plant_disease_mlops.data.validation import (
    build_manifest_dataframe,
    validate_manifest_with_great_expectations,
    validate_pil_image,
)


def test_validate_pil_image_accepts_rgb():
    img = PILImage.new("RGB", (100, 100), color=(10, 20, 30))
    ok, msg = validate_pil_image(img)
    assert ok and msg == "Valid"


def test_validate_pil_image_rejects_too_small():
    img = PILImage.new("RGB", (10, 10))
    ok, msg = validate_pil_image(img)
    assert not ok
    assert "small" in msg.lower()


def test_great_expectations_manifest(tmp_path: Path):
    df = pd.DataFrame(
        {
            "label": [0, 1, 2],
            "width": [224, 300, 128],
            "height": [224, 300, 128],
            "mode": ["RGB", "RGB", "RGB"],
        }
    )
    result = validate_manifest_with_great_expectations(df, num_classes=3)
    assert result.success


@pytest.fixture()
def tiny_dataset_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    paths: list[str] = []
    labels: list[int] = []
    for i in range(6):
        p = images_dir / f"im_{i}.png"
        PILImage.new("RGB", (120, 120), color=(i * 30, 10, 10)).save(p)
        paths.append(str(p))
        labels.append(i % 2)

    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(num_classes=2),
        }
    )
    ds = Dataset.from_dict({"image": paths, "label": labels}, features=features)
    dsd = DatasetDict(train=ds, test=ds.select([0, 1]))
    dsd.save_to_disk(str(raw))
    return raw


def test_preprocess_and_dataset_roundtrip(tiny_dataset_dir: Path, tmp_path: Path):
    from datasets import load_from_disk

    from plant_disease_mlops.data.preprocess import preprocess_dataset

    out = tmp_path / "processed"
    dsd = load_from_disk(str(tiny_dataset_dir))
    preprocess_dataset(dsd, out, val_fraction=0.34, seed=0)

    assert discover_num_classes(out, "train") == 2
    assert discover_num_classes(out, "test") == 2

    ds_torch = PlantDiseaseDataset(out, split="train", transform=None)
    assert len(ds_torch) > 0
    img, label = ds_torch[0]
    assert isinstance(label, int)
    assert img.size == (224, 224)


def test_build_manifest_and_validate_subset(tiny_dataset_dir: Path):
    from datasets import load_from_disk

    dsd = load_from_disk(str(tiny_dataset_dir))
    df = build_manifest_dataframe(dsd["train"], limit=4)
    res = validate_manifest_with_great_expectations(df, num_classes=2)
    assert res.success
