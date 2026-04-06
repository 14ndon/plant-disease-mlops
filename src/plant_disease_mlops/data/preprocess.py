"""Write processed JPEGs under ``train`` / ``validation`` / ``test`` class folders."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from plant_disease_mlops.data.transforms import preprocess_pil_image


def _write_split(
    dataset,
    split_name: str,
    out_root: Path,
    seed: int,
) -> None:
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    for i, idx in enumerate(tqdm(indices, desc=f"write {split_name}")):
        ex = dataset[idx]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        label = int(ex["label"])
        class_dir = split_dir / f"class_{label}"
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{i}.jpg"
        preprocess_pil_image(img).save(out_path, format="JPEG", quality=95)


def preprocess_dataset(
    dataset_dict,
    output_dir: Path,
    *,
    val_fraction: float,
    seed: int,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = dataset_dict["train"]
    if val_fraction <= 0:
        _write_split(train_ds, "train", output_dir, seed)
    else:
        n = len(train_ds)
        n_val = int(n * val_fraction)
        n_val = max(1, min(n - 1, n_val))
        idxs = list(range(n))
        random.Random(seed).shuffle(idxs)
        val_set = set(idxs[:n_val])
        train_indices = [i for i in idxs if i not in val_set]
        val_indices = sorted(val_set)

        train_part = train_ds.select(train_indices)
        val_part = train_ds.select(val_indices)

        _write_split(train_part, "train", output_dir, seed)
        _write_split(val_part, "validation", output_dir, seed + 1)

    _write_split(dataset_dict["test"], "test", output_dir, seed + 2)


def sync_to_s3(local_dir: Path, bucket_name: str, s3_prefix: str) -> None:
    """
    Sync ``local_dir`` to ``s3://{bucket}/{prefix}/`` using the AWS CLI (multipart,
    skips unchanged objects — good for resume and large trees).
    """
    if not shutil.which("aws"):
        raise RuntimeError(
            "The AWS CLI (`aws`) must be installed and on PATH. "
            "See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        )

    local_dir = Path(local_dir).resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    bucket_name = bucket_name.strip().strip("/")
    prefix = (s3_prefix or "").strip().strip("/")
    dest = f"s3://{bucket_name}/{prefix}/" if prefix else f"s3://{bucket_name}/"

    # Trailing slash: sync directory contents into the destination prefix.
    source = str(local_dir).rstrip("/") + "/"

    cmd = ["aws", "s3", "sync", source, dest]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())
