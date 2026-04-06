"""Download PlantVillage images from Hugging Face Hub archives (pinned revisions)."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image
from huggingface_hub import hf_hub_download

from plant_disease_mlops.data.constants import (
    HF_REPO_ID,
    REVISION_DATA_ZIP,
    REVISION_SPLITS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    ZIP_FILENAME,
)


def _find_extract_root(dest: Path) -> Path:
    for raw in dest.rglob("raw"):
        if raw.is_dir() and (raw / "color").is_dir():
            return raw.parent
    raise FileNotFoundError(
        f"No directory raw/color found under {dest} after extracting {ZIP_FILENAME}."
    )


def _list_class_names(color_dir: Path) -> list[str]:
    names = [p.name for p in color_dir.iterdir() if p.is_dir()]
    return sorted(names)


def _read_split_lines(split_file: Path) -> list[str]:
    text = split_file.read_text(encoding="utf-8")
    return [ln.strip().replace("\\", "/") for ln in text.splitlines() if ln.strip()]


def _build_split_dataset(
    extract_root: Path,
    split_file: Path,
    class_names: list[str],
    max_samples: int | None,
) -> Dataset:
    lines = _read_split_lines(split_file)
    if max_samples is not None:
        lines = lines[:max_samples]

    image_paths: list[str] = []
    labels: list[str] = []

    for rel in lines:
        full = (extract_root / rel).resolve()
        if not full.is_file():
            continue
        class_name = full.parent.name
        if class_name not in class_names:
            continue
        image_paths.append(str(full))
        labels.append(class_name)

    features = Features(
        {
            "image": Image(),
            "label": ClassLabel(names=class_names),
        }
    )
    label_ids = [class_names.index(lb) for lb in labels]
    return Dataset.from_dict(
        {"image": image_paths, "label": label_ids},
        features=features,
    )


def download_plantvillage(
    output_dir: Path,
    *,
    cache_dir: Path | None = None,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetDict:
    """
    Download ``data.zip``, official color split lists, build a ``DatasetDict`` with
    ``train`` and ``test`` splits, and ``save_to_disk`` under ``output_dir``.

    ``output_dir`` is the Hugging Face datasets disk folder (e.g. ``data/raw/plantvillage``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work = output_dir / "_download_work"
    work.mkdir(parents=True, exist_ok=True)
    archive_dir = work / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=ZIP_FILENAME,
            repo_type="dataset",
            revision=REVISION_DATA_ZIP,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )

    extract_marker = archive_dir / ".extract_complete"
    if not extract_marker.is_file():
        print(f"Extracting {zip_path.name} (this may take several minutes)...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(archive_dir)
        extract_marker.write_text("ok", encoding="utf-8")
    else:
        print("Using existing extracted archive (delete _download_work to re-extract).")

    extract_root = _find_extract_root(archive_dir)
    color_dir = extract_root / "raw" / "color"
    class_names = _list_class_names(color_dir)

    train_split_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=SPLIT_TRAIN,
            repo_type="dataset",
            revision=REVISION_SPLITS,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    test_split_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=SPLIT_TEST,
            repo_type="dataset",
            revision=REVISION_SPLITS,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )

    train_ds = _build_split_dataset(
        extract_root, train_split_path, class_names, max_train_samples
    )
    test_ds = _build_split_dataset(
        extract_root, test_split_path, class_names, max_test_samples
    )

    ds = DatasetDict(train=train_ds, test=test_ds)
    ds.save_to_disk(str(output_dir))
    return ds


def dataset_disk_exists(path: Path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "dataset_dict.json").is_file()
