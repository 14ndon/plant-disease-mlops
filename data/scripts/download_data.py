#!/usr/bin/env python3
"""Download PlantVillage images and optional IEEE metadata (see proposal)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root: plant-disease-mlops/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from plant_disease_mlops.data.hf_plantvillage import (  # noqa: E402
    dataset_disk_exists,
    download_plantvillage,
)


def download_metadata(metadata_path: Path) -> bool:
    """Return True if IEEE multimodal metadata is present (manual download)."""
    marker = metadata_path / "annotations.csv"
    if marker.is_file():
        print("Metadata found:", marker)
        return True
    print(
        """
        Multi-modal metadata not found. Optional manual steps:
        1. Visit: https://ieee-dataport.org/documents/context-aware-multimodal-augmented-plantvillage-dataset
        2. Download the archive and extract to:
        """
        f"           {metadata_path}\n"
        "        3. Ensure annotations.csv exists, then re-run this script.\n"
        "        Proceeding without metadata is fine for image-only ingestion.\n"
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "data" / "raw" / "plantvillage",
        help="HuggingFace datasets save_to_disk directory",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=_ROOT / "data" / "raw" / "metadata",
        help="Directory for IEEE metadata (manual)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap train split size (debug / CI)",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Cap test split size (debug / CI)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and rebuild even if output already exists",
    )
    args = parser.parse_args()

    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    cache_path = Path(cache_dir) if cache_dir else None

    if dataset_disk_exists(args.output) and not args.force:
        print(f"Dataset already on disk: {args.output} (use --force to rebuild)")
    else:
        if args.force and args.output.exists():
            import shutil

            shutil.rmtree(args.output)

        ds = download_plantvillage(
            args.output,
            cache_dir=cache_path,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        print("Train samples:", len(ds["train"]))
        print("Test samples:", len(ds["test"]))
        names = ds["train"].features["label"].names
        print("Classes:", len(names))

    download_metadata(args.metadata_dir)


if __name__ == "__main__":
    main()
